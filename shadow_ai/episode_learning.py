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


def compatible_policy_spaces(candidate, reference):
    """Episode updates must retain the legacy policy's input/action contract."""
    for name in ('observation_space', 'action_space'):
        expected = getattr(reference, name, None)
        actual = getattr(candidate, name, None)
        if expected is None:
            continue
        if actual is None or actual != expected:
            return False
    return True


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
        self.player_path = []
        self.elapsed = 0.
        self.skipped_decisions = {'overridden': 0, 'incompatible': 0}

    def record(self, observation, action: int, actor: str = "G0", position=None, executed=True):
        if self.finished:
            return False
        if not executed:
            self.skipped_decisions['overridden'] += 1
            return False
        observation = np.asarray(observation)
        space = getattr(self.model, 'observation_space', None)
        action_space = getattr(self.model, 'action_space', None)
        if ((space is not None and observation.shape != space.shape)
                or (action_space is not None and not action_space.contains(int(action)))):
            self.skipped_decisions['incompatible'] += 1
            return False
        self.samples.append({"observation": observation.tolist(),
                             "action": int(action), "actor": actor, "time": self.elapsed,
                             "position": None if position is None else np.asarray(position).tolist(),
                             'executed': True})
        return True

    def record_path(self, position, dt):
        if self.finished:
            return
        self.elapsed += dt
        self.player_path.append(dict(time=round(self.elapsed, 4), x=float(position[0]), y=float(position[1])))

    @staticmethod
    def route_rewards(payload):
        """Retrospective interception feedback; no route is sent to live sensors.

        Compare each agent's next decision position against the player's actual
        route over the following two seconds. Approaching that route is rewarded;
        moving away is penalized. Every recorded path point can supply a target.
        """
        samples, route = payload['samples'], payload.get('player_path', [])
        rewards = np.zeros(len(samples), dtype=np.float32)
        if not route:
            return rewards
        times = np.array([point['time'] for point in route])
        positions = np.array([[point['x'], point['y']] for point in route])
        following = {}
        for index in range(len(samples) - 1, -1, -1):
            sample = samples[index]
            actor = sample['actor']
            later = following.get(actor)
            following[actor] = sample
            if sample.get('position') is None or later is None or later.get('position') is None:
                continue
            target_index = min(len(route) - 1, int(np.searchsorted(times, sample.get('time', 0.) + 2.)))
            target = positions[target_index]
            before = np.linalg.norm(np.asarray(sample['position']) - target)
            after = np.linalg.norm(np.asarray(later['position']) - target)
            rewards[index] = np.clip((before - after) / max(1., before), -1., 1.)
        return rewards

    def finish(self, outcome: str, level: int, seed: int):
        if self.finished:
            return
        self.finished = True
        episode_id = uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=True)
        schema = self.directory / f"{self.mode}_{episode_id}.json"
        payload = {"version": 3, "mode": self.mode, "outcome": outcome,
                   "level": level, "seed": seed, "samples": self.samples,
                   "events": self.events, "player_path": self.player_path, "duration": self.elapsed,
                   'skipped_decisions': dict(self.skipped_decisions),
                   "training": "PENDING"}
        if not self.samples:
            payload['training'] = 'SKIPPED'
            payload['reason'] = 'No executed legacy policy decisions in this match'
            schema.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            self.result = dict(status='SKIPPED', schema=str(schema), level=level,
                               path_points=len(self.player_path), reason=payload['reason'])
            return
        schema.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self.result = {"status": "TRAINING", "schema": str(schema), "level": level, "path_points": len(self.player_path)}
        # The worker gets independent weights: live inference is never mutated.
        self.future = _WORKER.submit(self._train, deepcopy(self.model), payload, schema)

    def _train(self, model, payload, schema):
        try:
            # Version 1/2 schemas lack this flag. New schemas explicitly record
            # only actions whose intent survived the runtime's mandatory rules.
            samples = [sample for sample in payload['samples'] if sample.get('executed', True)]
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
            route_rewards = self.route_rewards(dict(payload, samples=samples))
            targets = torch.as_tensor(np.clip(.75 * outcome_reward + .5 * route_rewards[indices], -1., 1.),
                                      dtype=torch.float32, device=model.device)
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
            payload['route_training'] = dict(points=len(payload.get('player_path', [])),
                decisions_with_feedback=int(np.count_nonzero(route_rewards)))
            schema.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return {"status": "TRAINED", "schema": str(schema), "checkpoint": str(latest),
                    "episode_checkpoint": str(checkpoint), "samples": len(samples),
                    "path_points": len(payload.get('player_path', [])), "level": payload['level']}
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
