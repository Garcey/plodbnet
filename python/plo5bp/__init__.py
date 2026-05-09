"""plo5bp — PLO5 double-board bomb-pot self-play PPO pipeline."""

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM, encode_observation
from plo5bp.env import BombPotEnv
from plo5bp.actions import ACTION_NAMES, NUM_ACTIONS

__all__ = [
    "ACTION_NAMES",
    "BombPotEnv",
    "GameConfig",
    "NUM_ACTIONS",
    "OBS_DIM",
    "TrainingConfig",
    "encode_observation",
]
