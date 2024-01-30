from ctypes import *


class InitParam(Structure):
    _fields_ = [("w", c_int), ("h", c_int), ("numWorkers", c_int)]


class Observation(Structure):
    _fields_ = [("data", POINTER(c_byte)),
                ("size", c_int),
                ("reward", POINTER(c_int)),
                ("rewardSize", c_int),
                ("mask", POINTER(c_byte)),
                ("maskSize", c_int),
                ]


class Action(Structure):
    _fields_ = [("data", POINTER(c_byte)), ("size", c_int)]


class TotalObservation(Structure):
    _fields_ = [("ob1", Observation), ("ob2", Observation)]


class TotalAction(Structure):
    _fields_ = [("action1", Action), ("action2", Action)]


class Reward:
    GAME_TIME = 0
    IS_END = 1
    VICTORY_SIDE = 2
    NEW_WORKER_CNT = 3
    NEW_LIGHT_CNT = 4
    NEW_RANGED_CNT = 5
    NEW_HEAVY_CNT = 6
    NEW_BASE_CNT = 7
    NEW_BARRACK_CNT = 8

    DEAD_WORKER_CNT = 9
    DEAD_LIGHT_CNT = 10
    DEAD_RANGED_CNT = 11
    DEAD_HEAVY_CNT = 12
    DEAD_BASE_CNT = 13
    DEAD_BARRACK_CNT = 14

    NEW_WORKER_KILLED = 15
    NEW_LIGHT_KILLED = 16
    NEW_RANGED_KILLED = 17
    NEW_HEAVY_KILLED = 18
    NEW_BASE_KILLED = 19
    NEW_BARRACK_KILLED = 20
    NEW_NET_INCOME = 21
    NEW_HIT_CNT = 22


REWARD_SIZE = Reward.NEW_HIT_CNT + 1


class ActionPlane:
    ACTION = 0
    MOVE_PARAM = 1
    GATHER_PARAM = 2
    RETURN_PARAM = 3
    PRODUCE_DIRECTION_PARAM = 4
    PRODUCE_TYPE_PARAM = 5
    RELATIVE_ATTACK_POSITION = 6


class ActionType:
    NOOP = 0
    MOVE = 1
    GATHER = 2
    RETURN = 3
    PRODUCE = 4
    ATTACK = 5


# action, move dir, gather dir, return dir, prod dir, prod type, attack dir
ACTION_SIZE = [6, 4, 4, 4, 4, 6, 49]
ACTION_MASK_SIZE_PER_UNIT = sum(ACTION_SIZE)

GAME_W = 32
GAME_H = 32


class ObPlane:
    HP_1 = 0
    HP_2 = 1
    HP_3 = 2
    HP_4 = 3
    HP_5 = 4
    HP_6_PLUS = 5
    RES_1 = 6
    RES_2 = 7
    RES_3 = 8
    RES_4 = 9
    RES_5 = 10
    RES_6_PLUS = 11
    OWNER_1 = 12
    OWNER_NONE = 13
    OWNER_2 = 14
    OBSTACLE = 15
    GATHERING = 16
    IS_TERRAIN = 17
    IS_MINERAL = 18
    IS_BASE = 19
    IS_BARRACK = 20
    IS_WORKER = 21
    IS_LIGHT = 22
    IS_HEAVY = 23
    IS_RANGED = 24
    IS_NOOP = 25
    IS_MOVE = 26
    IS_GATHER = 27
    IS_RETURN = 28
    IS_PRODUCE = 29
    IS_ATTACK = 30


class ObjType:
    TERRAIN = 0
    MINERAL = 1
    BASE = 2
    BARRACK = 3
    WORKER = 4
    LIGHT = 5
    HEAVY = 6
    RANGED = 7


OBSERVATION_PLANE_NUM = 31


class AIActionMask:
    ACTION_TYPE_MASK = 0
    MOVE_PARAM_MASK = ACTION_TYPE_MASK + ActionType.ATTACK + 1
    GATHER_PARAM_MASK = MOVE_PARAM_MASK + 4
    RETURN_PARAM_MASK = GATHER_PARAM_MASK + 4
    PRODUCE_DIRECTION_PARAM_MASK = RETURN_PARAM_MASK + 4
    PRODUCE_TYPE_PARAM_MASK = PRODUCE_DIRECTION_PARAM_MASK + 4
    RELATIVE_ATTACK_POSITION_MASK = PRODUCE_TYPE_PARAM_MASK + 6
