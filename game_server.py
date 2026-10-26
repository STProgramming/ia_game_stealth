from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
    import torch
    # Tiny MLP inference is often faster with one CPU worker than with
    # thread-pool scheduling overhead.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from shadow_ai.interactive_runtime import (
    InputState,
    InteractiveGuardRuntime,
    InteractiveSwatRuntime,
)
from shadow_ai.episode_learning import EpisodeLearning
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


def first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def discover_guard_model() -> Optional[Path]:
    env = os.getenv("SHADOW_GUARD_MODEL")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            ROOT / "models_v7_1_guard" / "guard_v7_1_final.zip",
            ROOT / "models_v6_3_2" / "best_stable" / "best_model.zip",
            ROOT / "models_v6_3" / "best_stable" / "best_model.zip",
        ]
    )
    return first_existing(candidates)


def discover_swat_model() -> Optional[Path]:
    env = os.getenv("SHADOW_SWAT_MODEL")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            ROOT / "models_v7_1_swat" / "swat_v7_1_final.zip",
            ROOT / "models_v7_swat" / "swat_shared_policy_v7.zip",
        ]
    )
    return first_existing(candidates)


class GameSession:
    def __init__(self, guard_model: Optional[Path], swat_model: Optional[Path]):
        self.guard_model = guard_model
        self.swat_model = swat_model
        self.runtime = None
        self.mode = "guard" if guard_model else "swat"
        self.input = InputState()
        self.static_key = None
        self.revision = 0
        self.set_mode(self.mode)
        self._start_episode_learning()

    def _start_episode_learning(self):
        directory = ROOT / ".runtime" / "episodes"
        checkpoint = directory / f"{self.mode}_latest.zip"
        if checkpoint.exists():
            self.runtime.model = PPO.load(checkpoint, device="cpu")
        self.learning = EpisodeLearning(self.runtime.model, self.mode, directory)

    def set_mode(self, mode: str) -> None:
        if mode == "guard" and self.guard_model is not None:
            self.runtime = InteractiveGuardRuntime(self.guard_model)
            self.mode = "guard"
            return
        if mode == "swat" and self.swat_model is not None:
            self.runtime = InteractiveSwatRuntime(self.swat_model)
            self.mode = "swat"
            return
        if self.guard_model is not None:
            self.runtime = InteractiveGuardRuntime(self.guard_model)
            self.mode = "guard"
            return
        if self.swat_model is not None:
            self.runtime = InteractiveSwatRuntime(self.swat_model)
            self.mode = "swat"
            return
        raise RuntimeError(
            "No model found. Expected a Guard or SWAT .zip checkpoint."
        )

    def handle(self, data: dict) -> None:
        typ = data.get("type")
        if typ in {"mode", "reset", "scenario", "next_level"}:
            self.input = InputState()
            self.static_key = None
            self.revision += 1
        if typ == "next_level":
            if self.mode == "guard" and self.runtime.advance_level():
                self._start_episode_learning()
            return
        if typ == "input":
            incoming = InputState.from_dict(data)
            # Key-down pulses can arrive between two 30 Hz simulation ticks.
            # Keep them latched until the runtime consumes the next frame.
            incoming.merge_transients(self.input)
            self.input = incoming
        elif typ == "mode":
            self.set_mode(str(data.get("mode", "")))
            self._start_episode_learning()
        elif typ == "reset":
            scenario = str(data.get("scenario") or "")
            self.runtime.reset(scenario)
            self._start_episode_learning()
        elif typ == "scenario":
            scenario = str(data.get("scenario") or "")
            self.runtime.reset(scenario)
            self._start_episode_learning()

    def tick(self) -> dict:
        self.runtime.tick(self.input)
        for observation, action, actor in getattr(self.runtime, "learning_decisions", []):
            self.learning.record(observation, action, actor)
        for event in getattr(self.runtime, "learning_events", []):
            self.learning.events.append(event)
        if self.runtime.status in {"CAPTURED", "ESCAPED", "NEUTRALIZED"}:
            try:
                self.learning.finish(self.runtime.status, getattr(self.runtime, "level", 1), self.runtime.seed)
            except Exception as exc:
                self.learning.result = {"status": "FAILED", "error": str(exc)}
        self.input.clear_transients()

        key = (
            self.mode,
            getattr(self.runtime, "status", ""),
            getattr(
                getattr(self.runtime, "env", None),
                "training_scenario",
                getattr(getattr(self.runtime, "core", None), "scenario", ""),
            ),
        )
        include_static = key != self.static_key
        self.static_key = key

        state = self.runtime.state(include_static=include_static)
        state["revision"] = self.revision
        state["learning"] = self.learning.state()
        state["level"] = getattr(self.runtime, "level", None)
        state["available_modes"] = {
            "guard": self.guard_model is not None,
            "swat": self.swat_model is not None,
        }
        state["models"] = {
            "guard": None if self.guard_model is None else str(self.guard_model),
            "swat": None if self.swat_model is None else str(self.swat_model),
        }
        return state


app = FastAPI(title="Shadow Protocol AI Playground")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/status")
async def status():
    guard = discover_guard_model()
    swat = discover_swat_model()
    return {
        "guard_model": None if guard is None else str(guard),
        "swat_model": None if swat is None else str(swat),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    guard = discover_guard_model()
    swat = discover_swat_model()

    try:
        session = GameSession(guard, swat)
    except Exception as exc:
        await ws.send_json({"type": "fatal", "message": str(exc)})
        await ws.close()
        return

    async def receive_inputs():
        while True:
            raw = await ws.receive_text()
            session.handle(json.loads(raw))

    receiver = asyncio.create_task(receive_inputs())

    try:
        # Fixed 30 Hz authoritative physics. Input receiving is independent,
        # so a packet never waits behind the simulation/send loop.
        frame_time = 1.0 / 30.0
        loop = asyncio.get_running_loop()
        next_frame = loop.time()

        while True:
            state = session.tick()
            await ws.send_json({"type": "state", "state": state})

            next_frame += frame_time
            await asyncio.sleep(max(0.0, next_frame - loop.time()))

    except WebSocketDisconnect:
        return
    finally:
        receiver.cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--guard-model", type=Path)
    parser.add_argument("--swat-model", type=Path)
    args = parser.parse_args()

    if args.guard_model:
        os.environ["SHADOW_GUARD_MODEL"] = str(args.guard_model)
    if args.swat_model:
        os.environ["SHADOW_SWAT_MODEL"] = str(args.swat_model)

    print()
    print("Shadow Protocol AI Playground")
    print("-----------------------------")
    print("Guard:", discover_guard_model() or "NOT FOUND")
    print("SWAT: ", discover_swat_model() or "NOT FOUND")
    print(f"Open: http://{args.host}:{args.port}")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
