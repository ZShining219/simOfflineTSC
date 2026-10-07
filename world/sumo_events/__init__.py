"""Stable SUMO event API: lane_blockage, road_closure and global_rain.

Use install_events(world, schedule_path) for the existing LibSignal World.
Use SumoEventRuntime(net_file, Schedule(...)) for another SUMO connection.
Reports are immediate confirmed facts; audit output is privileged evidence.
See function docstrings and configs/events/hz4x4.yml for the calling contract.
"""
from .schema import Event, Report, Schedule, load_schedule
from .runtime import SumoEventRuntime
from .integration import install_events, validate_routes
from .grounding import ReportGrounder
from .plans import EpisodePlan, PlanController, install_event_plan, load_plan

__all__ = ['Event', 'Report', 'Schedule', 'load_schedule', 'SumoEventRuntime',
           'install_events', 'validate_routes', 'ReportGrounder',
           'EpisodePlan', 'PlanController', 'install_event_plan', 'load_plan']
