import time
from repo_anonymized.micro_sc.vec_env.game_types import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())
import torch
import gym
from gym.spaces.box import Box
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.discrete import Discrete


class VecEnvSc:
    ENV_ID = 1
    def __init__(self, num_workers: int, device: torch.device,
                 seed: int,
                 isRotSym: bool,
                 isAxSym: bool,
                 terrainProb: float,
                 expansionCnt: int,
                 clusterPerExpansion: int,
                 mineralPerCluster: int):
        import os
        self.obj = cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../cpp_lib/rts_engine_shared{VecEnvSc.ENV_ID}.dll"))
        VecEnvSc.ENV_ID += 1
        self.obj.Init.argtypes = [InitParam]
        self.obj.Reset.argtypes = [c_int, c_bool, c_bool, c_double, c_int, c_int, c_int]
        self.obj.Reset.restype = TotalObservation
        self.obj.Step.argtypes = [TotalAction]
        self.obj.Step.restype = TotalObservation
        self.unwrapped = self
        self.device = device
        self.num_workers = num_workers
        initParam = InitParam(c_int(GAME_W), c_int(GAME_H), c_int(self.num_workers))
        self.obj.Init(initParam)
        self.reward_weight1 = torch.tensor([0,           # GAME_TIME
                               0,           # IS_END
                               -1000,       # VICTORY_SIDE
                               2 + 1,       # NEW_WORKER_CNT
                               1 + 4,       # NEW_LIGHT_CNT
                               2 + 8,       # NEW_RANGED_CNT
                               4 + 16,      # NEW_HEAVY_CNT
                               16,          # NEW_BASE_CNT
                               6 + 1,       # NEW_BARRACK_CNT
                               0,           # DEAD_WORKER_CNT
                               0,           # DEAD_LIGHT_CNT
                               0,           # DEAD_RANGED_CNT
                               0,           # DEAD_HEAVY_CNT
                               0,           # DEAD_BASE_CNT
                               0,           # DEAD_BARRACK_CNT
                               2 + 1,       # NEW_WORKER_KILLED
                               1 + 4,       # NEW_LIGHT_KILLED
                               2 + 8,       # NEW_RANGED_KILLED
                               4 + 16,      # NEW_HEAVY_KILLED
                               16,          # NEW_BASE_KILLED
                               6 + 1,       # NEW_BARRACK_KILLED
                               1,           # NEW_NET_INCOME,
                               1,])         # NEW_HIT_CNT
        self.reward_weight2 = self.reward_weight1.clone()
        self.reward_weight2[Reward.VICTORY_SIDE] = 1000
        self.reward_weight1 = self.reward_weight1.reshape(-1, 1).type(torch.FloatTensor)
        self.reward_weight2 = self.reward_weight2.reshape(-1, 1).type(torch.FloatTensor)

        self.seed = seed
        self.isRotSym = isRotSym
        self.isAxSym = isAxSym
        self.terrainProb = terrainProb
        self.expansionCnt = expansionCnt
        self.clusterPerExpansion = clusterPerExpansion
        self.mineralPerCluster = mineralPerCluster

        self.observation_space = Box(low=-1.0, high=1.0, shape=(OBSERVATION_PLANE_NUM, GAME_H, GAME_W))
        self.action_plane_space = gym.spaces.MultiDiscrete(ACTION_SIZE)
        self.action_space = MultiDiscrete(np.array([ACTION_SIZE] * GAME_H * GAME_H).flatten().tolist())

    def reset(self) -> ((torch.tensor, torch.tensor), (torch.tensor, torch.tensor)):
        totalObs = self.obj.Reset(self.seed, self.isRotSym, self.isAxSym, self.terrainProb, self.expansionCnt, self.clusterPerExpansion, self.mineralPerCluster)
        assert totalObs.ob1.size == self.num_workers * OBSERVATION_PLANE_NUM * GAME_H * GAME_W
        assert totalObs.ob2.size == self.num_workers * OBSERVATION_PLANE_NUM * GAME_H * GAME_W
        ob1 = torch.from_numpy(np.ctypeslib.as_array(totalObs.ob1.data, [self.num_workers, OBSERVATION_PLANE_NUM, GAME_H, GAME_W])).type(torch.FloatTensor)
        ob2 = torch.from_numpy(np.ctypeslib.as_array(totalObs.ob2.data, [self.num_workers, OBSERVATION_PLANE_NUM, GAME_H, GAME_W])).type(torch.FloatTensor)
        ob1 = ob1 * 2 - 1
        ob2 = ob2 * 2 - 1
        ob1 = ob1.detach().to(self.device)
        ob2 = ob2.detach().to(self.device)
        mask1 = torch.from_numpy(np.ctypeslib.as_array(totalObs.ob1.mask, [self.num_workers, ACTION_MASK_SIZE_PER_UNIT, GAME_H, GAME_W])).type(torch.FloatTensor).detach().to(self.device)
        mask2 = torch.from_numpy(np.ctypeslib.as_array(totalObs.ob2.mask, [self.num_workers, ACTION_MASK_SIZE_PER_UNIT, GAME_H, GAME_W])).type(torch.FloatTensor).detach().to(self.device)

        self.seed += self.num_workers
        return (ob1, ob2), (mask1, mask2)

    def step(self, action1: torch.tensor, action2: torch.tensor) -> ((torch.tensor, torch.tensor), (torch.tensor, torch.tensor), torch.tensor, str):
        actionData1 = action1.reshape(self.num_workers * len(ACTION_SIZE) * GAME_H * GAME_W).type(torch.uint8)
        actionData2 = action2.reshape(self.num_workers * len(ACTION_SIZE) * GAME_H * GAME_W).type(torch.uint8)
        actionObj1 = actionData1.numpy().ctypes.data_as(POINTER(c_byte))
        actionObj2 = actionData2.numpy().ctypes.data_as(POINTER(c_byte))
        actionStruct1 = Action(actionObj1, c_int(actionData1.size(0)))
        actionStruct2 = Action(actionObj2, c_int(actionData2.size(0)))

        totalObs = self.obj.Step(TotalAction(actionStruct1, actionStruct2))
        assert totalObs.ob1.size == self.num_workers * OBSERVATION_PLANE_NUM * GAME_H * GAME_W
        assert totalObs.ob2.size == self.num_workers * OBSERVATION_PLANE_NUM * GAME_H * GAME_W
        ob1 = torch.from_numpy(np.ctypeslib.as_array(totalObs.ob1.data, [self.num_workers, OBSERVATION_PLANE_NUM, GAME_H, GAME_W])).type(torch.FloatTensor)
        ob2 = torch.from_numpy(np.ctypeslib.as_array(totalObs.ob2.data, [self.num_workers, OBSERVATION_PLANE_NUM, GAME_H, GAME_W])).type(torch.FloatTensor)
        ob1 = ob1 * 2 - 1
        ob2 = ob2 * 2 - 1
        ob1 = ob1.detach().to(self.device)
        ob2 = ob2.detach().to(self.device)

        # print("time = ", totalObs.ob1.reward[Reward.GAME_TIME])
        reTensor1 = torch.from_numpy(np.ctypeslib.as_array(totalObs.ob1.reward, [self.num_workers, REWARD_SIZE])).type(torch.FloatTensor)
        reTensor2 = torch.from_numpy(np.ctypeslib.as_array(totalObs.ob2.reward, [self.num_workers, REWARD_SIZE])).type(torch.FloatTensor)
        isEnd = reTensor1[:, Reward.IS_END].flatten()
        re1 = torch.matmul(reTensor1, self.reward_weight1).flatten()
        re2 = torch.matmul(reTensor2, self.reward_weight2).flatten()

        re1 = re1.detach().to(self.device)
        re2 = re2.detach().to(self.device)

        mask1 = torch.from_numpy(np.ctypeslib.as_array(totalObs.ob1.mask, [self.num_workers, ACTION_MASK_SIZE_PER_UNIT, GAME_H, GAME_W])).type(torch.FloatTensor).detach().to(self.device)
        mask2 = torch.from_numpy(np.ctypeslib.as_array(totalObs.ob2.mask, [self.num_workers, ACTION_MASK_SIZE_PER_UNIT, GAME_H, GAME_W])).type(torch.FloatTensor).detach().to(self.device)
        return (ob1, ob2), (mask1, mask2),  (re1, re2), isEnd, totalObs.ob1.reward[Reward.GAME_TIME]


if __name__ == "__main__":
    # env = VecEnv(128, torch.device("cpu"), 0, False, True, 1, 5, 5, 100)
    env = VecEnvSc(128, torch.device("cpu"), 0, False, True, 0, 1, 1, 100)
    baseCoord = [None] * env.num_workers
    ob, mask = env.reset()
    action1 = torch.zeros((env.num_workers, len(ACTION_SIZE), GAME_H, GAME_W)).type(torch.int8)
    for w in range(env.num_workers):
        for y in range(GAME_H):
            for x in range(GAME_W):
                if ob[0][w, ObPlane.IS_BASE, y, x] != -1 and ob[0][w, ObPlane.OWNER_1, y, x] != -1:
                    action1[w, ActionPlane.ACTION, y, x] = ActionType.PRODUCE
                    action1[w, ActionPlane.PRODUCE_TYPE_PARAM, y, x] = ObjType.WORKER - ObjType.BASE
                    action1[w, ActionPlane.PRODUCE_DIRECTION_PARAM, y, x] = 2
                    baseCoord[w] = (y, x)
    for w in range(env.num_workers):
        for y in range(baseCoord[w][0], GAME_H):
            action1[w, ActionPlane.ACTION, y, baseCoord[w][1]] = ActionType.MOVE
            action1[w, ActionPlane.MOVE_PARAM, y, baseCoord[w][1]] = 2
            if ob[0][w, ObPlane.IS_BASE, y, baseCoord[w][1]] != -1 and ob[0][w, ObPlane.OWNER_2, y, baseCoord[w][1]] != -1:
                action1[w, ActionPlane.ACTION, y - 1, baseCoord[w][1]] = ActionType.ATTACK
                action1[w, ActionPlane.RELATIVE_ATTACK_POSITION, y - 1, baseCoord[w][1]] = 7 * 4 + 3
        action1[w, ActionPlane.ACTION, baseCoord[w][0], baseCoord[w][1]] = ActionType.PRODUCE

    action2 = torch.zeros((env.num_workers, len(ACTION_SIZE), GAME_H, GAME_W)).type(torch.int8)
    for i in range(5 + 2 * 32 + 60 * 2): # 5 step produce, 2*32 step move, 60 * 2 step attack
        o, mask, r, isEnd, t = env.step(action1, action2)
        print(r[0][0], r[1][0], isEnd[0])
        for j in range(ACTION_MASK_SIZE_PER_UNIT):
            fileName = 'masks/' + str(j) + '_time=' + str(t) + '.png'
            plt.imsave(fileName, mask[0][-1, j])

    plt.imshow(o[1][-1, ObPlane.IS_BASE])
    plt.show()
