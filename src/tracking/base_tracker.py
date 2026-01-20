# src/tracking/base_tracker.py
from abc import ABC, abstractmethod

class BaseTracker(ABC):
    @abstractmethod
    def init(self, **kwargs):
        """Initialize tracking session (e.g., wandb.init())."""
        pass

    @abstractmethod
    def log(self, metrics: dict, step: int = None):
        """Log metrics (scalars, histograms, images, etc.)."""
        pass

    @abstractmethod
    def finish(self):
        """Clean up resources (e.g., wandb.finish())."""
        pass
