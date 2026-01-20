# src/tracking/tracker_manager.py
from typing import Optional
from .base_tracker import BaseTracker

class TrackerManager:
    _tracker: Optional[BaseTracker] = None

    @classmethod
    def set_tracker(cls, tracker: BaseTracker):
        """Set the global tracker (e.g., WandbTracker, MLflowTracker)."""
        cls._tracker = tracker

    @classmethod
    def init(cls, **kwargs):
        if cls._tracker:
            cls._tracker.init(**kwargs)

    @classmethod
    def log(cls, metrics: dict, step: int = None):
        if cls._tracker:
            cls._tracker.log(metrics, step=step)

    @classmethod
    def finish(cls):
        if cls._tracker:
            cls._tracker.finish()
