"""Cooperative wall-clock budget and file-only timing evidence."""
import json
import math
import os
import time
from pathlib import Path


class Budget:
    def __init__(self, output, seconds=None):
        value = seconds if seconds is not None else os.environ.get("REAXKIT_OPTIMIZATION_TIME_LIMIT_SECONDS")
        self.seconds = float(value) if value is not None else None
        if self.seconds is not None and (not math.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError("Optimization time limit must be positive and finite")
        self.path = Path(os.environ.get("REAXKIT_TIMING_FILE", str(Path(output) / "optimization-timing.json")))
        self.created = time.monotonic()
        self.started = None
        self.reason = None
        self.count = 0
        self.publish("initializing")

    def begin(self):
        if self.started is None:
            self.started = time.monotonic()
            self.publish("running")

    def should_stop(self):
        self.begin()
        if self.seconds is not None and time.monotonic() - self.started >= self.seconds:
            self.reason = "time_limit"
        stop = os.environ.get("REAXKIT_STOP_FILE")
        if stop and Path(stop).exists():
            self.reason = self.reason or "stop_requested"
        self.publish("stopping" if self.reason else "running")
        return self.reason is not None

    def checkpoint(self, score=None):
        self.count += 1
        if score is not None:
            value = float(score)
            record = dict(completed=self.count, elapsed_seconds=time.monotonic() - self.started,
                          score=value if math.isfinite(value) else None)
            history = self.path.with_name("optimization-events.jsonl")
            history.parent.mkdir(parents=True, exist_ok=True)
            with history.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
        self.publish("running")

    def finish(self, reason=None):
        self.reason = reason or self.reason or "optimizer_criteria"
        self.publish("completed", graceful=True)

    def publish(self, state, graceful=False):
        current = time.monotonic()
        record = dict(schema_version=1, state=state, termination_reason=self.reason,
                      graceful=graceful, time_limit_seconds=self.seconds,
                      initialization_seconds=None if self.started is None else self.started-self.created,
                      optimization_elapsed_seconds=None if self.started is None else current-self.started,
                      completed_units=self.count, clock_scope="optimizer_entrypoint",
                      updated_at_epoch=time.time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.path)
