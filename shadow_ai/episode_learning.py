"""Persist human matches and make small outcome-based policy updates off-thread."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
from uuid import uuid4

import numpy as np
import torch


_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="match-learning")
_SAVE_LOCK = threading.Lock()


class EpisodeLearning:
    def __init__(self, model, mode: str, directory: Path):
        self.model = model
        self.mode = mode
        self.directory = directory
        self.samples = []
        self.events = []
        self.result = {"status": "RECORDING"}
        self.future = None
        self.finished = False

    def record(self, observation, action: int, actor: str = "G0"):
        self.samples.append({"observation": np.asarray(observation).tolist(),
                             "action": int(action), "actor": actor})

    def finish(self, outcome: str, level: int, seed: int):
        if self.finished:
            return
        self.finished = True
        episode_id = uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=True)
        schema = self.directory / f"{self.mode}_{episode_id}.json"
        payload = {"version": 1, "mode": self.mode, "outcome": outcome,
                   "level": level, "seed": seed, "samples": self.samples,
                   "events": self.events, "training": "PENDING"}
        schema.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self.result = {"status": "TRAINING", "schema": str(schema)}
        # The worker gets independent weights: live inference is never mutated.
        self.future = _WORKER.submit(self._train, deepcopy(self.model), payload, schema)

    def _train(self, model, payload, schema):
        try:
            samples = payload["samples"]
            if not samples:
                raise ValueError("No tactical decisions recorded in this match")
            # Keep every match in the schema; bound each update to 2048 decisions.
            indices = np.linspace(0, len(samples) - 1, min(2048, len(samples)), dtype=int)
            batch = [samples[i] for i in indices]
            observations = torch.as_tensor(np.asarray([s["observation"] for s in batch]),
                                           dtype=torch.float32, device=model.device)
            actions = torch.as_tensor([s["action"] for s in batch], device=model.device)
            outcome_reward = 1.0 if payload["outcome"] in {"CAPTURED", "NEUTRALIZED"} else -1.0
            # Bounded outcome-weighted policy gradient with a fitted value baseline.
            # This is incremental learning from a human match, not a PPO rollout.
            targets = torch.full((len(batch),), outcome_reward, device=model.device)
            model.policy.set_training_mode(True)
            optimizer = model.policy.optimizer
            for group in optimizer.param_groups:
                group["lr"] = min(float(group["lr"]), 1e-5)
            for _ in range(2):
                for start in range(0, len(batch), 64):
                    values, log_prob, entropy = model.policy.evaluate_actions(
                        observations[start:start + 64], actions[start:start + 64])
                    expected = targets[start:start + 64]
                    advantage = (expected - values.flatten().detach()).clamp(-1, 1)
                    loss = -(advantage * log_prob).mean()
                    loss += 0.5 * torch.nn.functional.mse_loss(values.flatten(), expected)
                    if entropy is not None:
                        loss -= 0.01 * entropy.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                    optimizer.step()
            checkpoint = self.directory / f"{schema.stem}.zip"
            model.save(checkpoint)
            latest = self.directory / f"{self.mode}_latest.zip"
            with _SAVE_LOCK:
                temporary = self.directory / f"{schema.stem}_latest.tmp"
                temporary.write_bytes(checkpoint.read_bytes())
                os.replace(temporary, latest)
            payload["training"] = "TRAINED"
            payload["checkpoint"] = str(checkpoint)
            schema.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return {"status": "TRAINED", "schema": str(schema), "checkpoint": str(latest),
                    "samples": len(samples)}
        except Exception as exc:
            payload["training"] = "FAILED"
            payload["error"] = str(exc)
            schema.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return {"status": "FAILED", "schema": str(schema), "error": str(exc)}

    def state(self):
        if self.future is not None and self.future.done():
            try:
                self.result = self.future.result()
            except Exception as exc:
                self.result = {"status": "FAILED", "error": str(exc)}
        return dict(self.result)
