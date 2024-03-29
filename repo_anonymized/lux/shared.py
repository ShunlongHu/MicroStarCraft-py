import math
from typing import Tuple, Union

import numpy as np
from luxai_s2.config import EnvConfig
from luxai_s2.factory import Factory
from luxai_s2.map.position import Position
from luxai_s2.state import State
from luxai_s2.unit import Unit

from repo_anonymized.lux.kit.config import EnvConfig as KitEnvConfig
from repo_anonymized.lux.kit.factory import Factory as KitFactory
from repo_anonymized.lux.kit.kit import GameState
from repo_anonymized.lux.kit.unit import Unit as KitUnit

LuxEnvConfig = Union[EnvConfig, KitEnvConfig]
LuxGameState = Union[State, GameState]
LuxFactory = Union[Factory, KitFactory]
LuxUnit = Union[Unit, KitUnit]


def pos_to_numpy(pos: Union[Position, np.ndarray]) -> np.ndarray:
    if isinstance(pos, Position):
        return pos.pos
    return pos


def pos_to_idx(pos: Union[Position, np.ndarray], map_size: int) -> int:
    pos = pos_to_numpy(pos)
    return pos[0] * map_size + pos[1]


def idx_to_pos(idx: int, map_size: int) -> Tuple[int, int]:
    return (idx // map_size, idx % map_size)


def factory_water_cost(
    factory: LuxFactory, state: LuxGameState, env_cfg: LuxEnvConfig
) -> int:
    if isinstance(factory, Factory):
        return factory.water_cost(env_cfg)
    elif isinstance(factory, KitFactory):
        return factory.water_cost(state)
    else:
        raise NotImplementedError(f"{factory.__class__.__name__} unsupported")


def move_power_cost(unit: LuxUnit, rubble_at_target: int) -> int:
    return math.floor(
        unit.unit_cfg.MOVE_COST + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
    )


def agent_id(unit: LuxUnit) -> str:
    if isinstance(unit, Unit):
        return unit.team.agent
    elif isinstance(unit, KitUnit):
        return unit.agent_id
    else:
        raise NotImplementedError(f"{unit.__class__.__name__} unsupported")
