import math
from scipy.spatial.distance import pdist, squareform
import numpy as np
from multiagent.core import World, Agent, Landmark, LandmarkType
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self, evaluate=False):
        self.R = 1.0  # communication range
        self.deta = 0.1
        self.r = 0.1  # perception range
        self.num = 4  # number of agents
        self.agentSize = 0.03  # agent's size
        self.safeLearning = False
        self.evaluate = evaluate  # parameter for whether the initial states is same
        # load the map
        self.pos = None  # position of coverpoint
        self.energy = None # energy of coverpoint
        self.obstacle = None # obstacle pos

        self.dx = 0.1
        self.dy = 0.1
        self.connected = True
        self.collision = False
        self.coverCount = 5  # 待覆盖点位数量
        self.coverpointE = 5  # 待覆盖次数
        self.doneRew = 100.0  # 点位完成覆盖后给与的奖励
        self.coverRew = 10.0  # 单次覆盖奖励

    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 0
        num_agents = self.num
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        # 待覆盖点+四个障碍物（长条，正方，不规则（由2个大小不一方形组合而成））
        # world.landmarks = [Landmark() for i in range(self.coverCount + 3)]
        world.landmarks = [Landmark() for i in range(self.coverCount)]

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False  # This parameter is used to apply the force of the collision
            agent.silent = True
            agent.size = self.agentSize
            agent.R = self.R
            agent.deta = self.deta
            agent.r = self.r
            agent.coverage = False

        for i, landmarks in enumerate(world.landmarks):
            landmarks.name = 'energyPoint %d' %i

            # 障碍物单独处理
            if i < self.coverCount:
                landmarks.type = LandmarkType.OBSTACLE
                landmarks.r = 0.01
            # else:
            #     landmarks.type = LandmarkType.ENERGY_POINT

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for _, agent in enumerate(world.agents):
            agent.color = np.array([128/255, 118/255, 105/255])
        # set random initial states
        init_pos_range = 0.95

        # 随机生成待覆盖点位位置并更新待覆盖点能量
        self.pos = np.random.uniform(-0.6, +0.6, (self.coverCount, 2))
        self.energy = np.array(
            [self.coverpointE for _ in range(self.coverCount)])

        # 生成圆形障碍物位置，当前为固定模式
        for i, landmarks in enumerate(world.landmarks):
            if i < self.coverCount:
                # 覆盖点
                landmarks.state.p_pos = self.pos[i]
                landmarks.state.coverpointE = self.coverpointE
                landmarks.state.energy = self.coverpointE
            else:
                # 障碍物
                landmarks.state.p_pos = self.pos[i]
                landmarks.energy = self.energy[i]


        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(
                -init_pos_range, +init_pos_range, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        while True:
            lamed2 = self.cal_laplace(world)
            if lamed2 >= 0.3:
                getpos = np.array([])
                for k in world.agents:
                    getpos = np.append(getpos, k.state.p_pos)
                getpos = getpos.reshape(self.num, 2)
                min_dist = min(pdist(getpos, 'euclidean'))
                if min_dist > 2 * self.agentSize:
                    break
                else:
                    for agent in world.agents:
                        agent.state.p_pos = np.random.uniform(
                            -init_pos_range, +init_pos_range, world.dim_p)
            else:
                for agent in world.agents:
                    agent.state.p_pos = np.random.uniform(
                        -init_pos_range, +init_pos_range, world.dim_p)
        if self.evaluate:
            while True:
                lamed2 = self.cal_laplace(world)
                if lamed2 >= 0.1:
                    getpos = np.array([])
                    for k in world.agents:
                        getpos = np.append(getpos, k.state.p_pos)
                    getpos = getpos.reshape(self.num, 2)
                    min_dist = min(pdist(getpos, 'euclidean'))
                    if min_dist > 2 * self.agentSize:
                        break
                    else:
                        for agent in world.agents:
                            agent.state.p_pos = np.random.uniform(
                                -init_pos_range, +init_pos_range, world.dim_p)
                else:
                    for agent in world.agents:
                        agent.state.p_pos = np.random.uniform(
                            -init_pos_range, +init_pos_range, world.dim_p)
            # world.agents[0].state.p_pos = np.array([-0.8, 0.8])
            # world.agents[0].state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
            # world.agents[1].state.p_pos = world.agents[0].state.p_pos + np.array([0.15, 0.0])
            # world.agents[2].state.p_pos = world.agents[0].state.p_pos - np.array([0.15, 0.0])
            # world.agents[3].state.p_pos = world.agents[0].state.p_pos + np.array([0.0, 0.15])
            # world.agents[0].state.p_pos = np.array([-0.8, 0.8])
            # world.agents[1].state.p_pos = world.agents[0].state.p_pos + np.array([0.1, 0.0])
            # world.agents[2].state.p_pos = world.agents[0].state.p_pos + np.array([0.2, 0.0])
            # world.agents[3].state.p_pos = world.agents[0].state.p_pos + np.array([0.3, 0.0])

    def cal_laplace(self, world):
        # calculate the connectivity of the graph as well as its eigenvector
        getpos = np.array([])
        for k in world.agents:
            getpos = np.append(getpos, k.state.p_pos)
        getpos = getpos.reshape(self.num, 2)
        value = -(self.R ** 2 / math.log(self.deta))
        dist = squareform(pdist(getpos, 'euclidean'))
        adjacency = np.zeros((self.num, self.num))
        for i in range(self.num):
            for j in range(i + 1, self.num):
                if dist[i, j] <= self.R:
                    adjacency[i, j] = math.exp(-dist[i, j] ** 2 / value)
                adjacency[j, i] = adjacency[i, j]

        degree = np.diag(np.sum(adjacency, axis=1))
        laplacian = degree - adjacency
        eig_value, eig_vector = np.linalg.eig(laplacian)
        eig_value_sort = eig_value[np.argsort(eig_value)]
        lamed2 = eig_value_sort[1]
        return lamed2

    # 计算当前agent覆盖的点位，更新点位信息
    def cover(self, agent, world):
        # 计算距离信息
        diff = np.linalg.norm(self.pos - agent.state.p_pos, axis=1)

        # 得到在覆盖范围内的点位
        inRangeIndex = diff < self.r
        print(np.sum(inRangeIndex), diff)
        # 得到有效覆盖位置（能量不为0，在覆盖范围内）
        needToBeCover = np.logical_and(self.energy != 0, inRangeIndex)

        # 对覆盖完全的点位进行额外奖励
        doneCount = np.sum(self.energy[needToBeCover] == 1)

        # 更新覆盖点能量
        self.energy[needToBeCover] -= 1
        # 更新待覆盖点状态，方便进行查看
        for i, _ in enumerate(needToBeCover):
            if needToBeCover[i]:
                world.landmarks[i].state.energy -= 1

        # 返回此次覆盖的点位数量
        return np.sum(needToBeCover) * self.coverRew + doneCount * self.doneRew

    def reward(self, agent, world):
        self.connected = True
        self.collision = False
        # Agents are rewarded based on the strength covered, penalized for collisions and unconnected
        rew = 0

        if self.out(agent):
            return -1000

        if self.done(agent, world):
            return 1000

        # 找到最近的待覆盖点位，并给与负奖励
        dists = np.sum(np.square(self.pos - agent.state.p_pos), axis=1)
        sorted_index = np.argsort(dists)

        for i in sorted_index:
            if self.energy[i] > 0:
                # 加大数量级
                rew -= dists[i] * 10
                break

        lamed2 = self.cal_laplace(world)
        lamed2_th = 0.1  # lamed2's threshold
        if lamed2 >= lamed2_th:
            rew -= 0
        elif lamed2 > 0:
            rew -= 5
        else:
            rew -= 10
            self.connected = False

        # 获得覆盖奖励
        rew += self.cover(agent, world)

        for a in world.agents:
            if a is not agent:
                dist = np.sqrt(
                    np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                if dist >= 2*self.r:
                    rew += 0
                elif dist >= 2*self.agentSize:
                    rew -= 5
                else:
                    rew -= 10
                    self.collision = True
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        other_pos = []
        other_speed = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_speed.append(other.state.p_vel - agent.state.p_vel)

        # 计算距离信息
        diff = np.linalg.norm(self.pos - agent.state.p_pos, axis=1)
        result = [agent.state.p_vel] + [agent.state.p_pos] + \
            other_pos + other_speed + [self.energy] + [diff]

        return np.concatenate(result)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos)

    def info(self, agent, world):
        return {'connected': self.connected, 'collision': self.collision, 'coverage': self.coverage}

    # 检测是否出界
    def out(self, agent):
        if (abs(agent.state.p_pos) > 1).any():
            return True
        return False

    # 检测是否完成覆盖任务
    def done(self, agent, world):
        self.coverage = 1 - np.sum(self.energy) / \
            (self.coverCount * self.coverpointE)
        if self.coverage == 1.0:
            return True
        return False
        # if (abs(agent.state.p_pos) > 1).any():
        #     return True
        # return False
