# src/tracking/__init__.py
from .tracker_manager import TrackerManager

# convenience functions
set_tracker = TrackerManager.set_tracker
init = TrackerManager.init
log = TrackerManager.log
finish = TrackerManager.finish
