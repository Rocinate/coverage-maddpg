import math

import gym
from gym import spaces
from scipy.spatial.distance import pdist, squareform
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from qpsolvers import solve_qp
from scipy.optimize import minimize


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True, safe_control=False):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.second_order_dynamic = False
        # if true, action is a_x and a_y
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        self.sensitivity = 2.0  # 5.0
        self.safe_control = safe_control
        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                if self.second_order_dynamic:
                    u_action_space = spaces.Discrete(world.dim_p * 2)
                else:
                    u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if not agent.silent:
                if self.discrete_action_space:
                    c_action_space = spaces.Discrete(world.dim_c)
                else:
                    c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
                agent.action.c = np.zeros(self.world.dim_c)
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def fun(self, udes):
        def v(u):
            return np.sum(np.square(u - udes))

        return v

    def connect_contriant(self, beta, speed_n, trans_mat, dt, h0):
        def v(u):
            return np.dot(beta.T, speed_n) + dt * np.dot(np.dot(beta.T, trans_mat), u) + h0
        return v

    def collision_contriant(self, G_p, h_p):
        def v(u):
            return np.dot(G_p, u) + h_p
        return v

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.agents = self.world.policy_agents
        if self.safe_control:
            # safe control
            # cal the beta and h0
            num = len(self.agents)
            pos_n = np.array([])
            speed_n = np.array([])
            s = np.array([])
            udes = np.array([])
            trans_mat = np.zeros((2 * num, 4 * num))
            for i in range(num):
                pos_n = np.append(pos_n, self.agents[i].state.p_pos)
                speed_n = np.append(speed_n, self.agents[i].state.p_vel)
                s = np.append(s, self.agents[i].state.p_pos + self.world.dt * self.agents[i].state.p_vel)
                udes = np.append(udes, action_n[i][1:])
                trans_mat[2 * i, 4 * i] = self.sensitivity
                trans_mat[2 * i + 1, 4 * i + 2] = self.sensitivity
                trans_mat[2 * i, 4 * i + 1] = -self.sensitivity
                trans_mat[2 * i + 1, 4 * i + 3] = -self.sensitivity
            value = -(self.agents[0].R ** 2 / math.log(self.agents[0].deta))
            dist = squareform(pdist(s.reshape(num, 2), 'euclidean'))
            adjacency = np.zeros((num, num))
            for i in range(num):
                for j in range(i + 1, num):
                    if dist[i, j] <= self.agents[0].R:
                        adjacency[i, j] = math.exp(-dist[i, j] ** 2 / value)
                    adjacency[j, i] = adjacency[i, j]

            degree = np.diag(np.sum(adjacency, axis=1))
            laplacian = degree - adjacency
            eig_value, eig_vector = np.linalg.eig(laplacian)
            eig_value_sort = eig_value[np.argsort(eig_value)]
            eig_vector_sort = eig_vector[:, np.argsort(eig_value)]
            lamed2 = eig_value_sort[1]
            vector = eig_vector_sort[:, 1]
            beta = np.zeros((2 * num, 1))
            s = s.reshape(num, 2)
            for i in range(num):
                a2x = 0
                a2y = 0
                idx = [idx for (idx, val) in enumerate(adjacency[i]) if val > 0]
                for j in range(len(idx)):
                    agent = idx[j]
                    a2x = a2x + (-(adjacency[i, agent]) / (0.5 * value)) * ((s[i, 0]) - (s[agent, 0])) * \
                          ((vector[i] - vector[agent]) ** 2)
                    a2y = a2y + (-(adjacency[i, agent]) / (0.5 * value)) * ((s[i, 1]) - (s[agent, 1])) * \
                          ((vector[i] - vector[agent]) ** 2)
                beta[2 * i] = a2x
                beta[2 * i + 1] = a2y
            h0 = lamed2 - 0.05
            cons = ({'type': 'ineq', 'fun': self.connect_contriant(beta, speed_n, trans_mat, self.world.dt, h0)},)
            speed_n = speed_n.reshape(num, 2)
            for i in range(num - 1):
                for j in range(i + 1, num):
                    if dist[i, j] >= 0.2:
                        continue
                    else:
                        h_coll = dist[i, j] ** 2 - 4 * self.agents[0].size ** 2
                        deta_s = s[i, :] - s[j, :]
                        deta_speed = speed_n[i, :] - speed_n[j, :]
                        deta_trans = deta_s[0] * (trans_mat[2 * i, :] - trans_mat[2 * j, :]) + \
                                     deta_s[1] * (trans_mat[2 * i + 1, :] - trans_mat[2 * j + 1, :])
                        cons = cons + ({'type': 'ineq', 'fun': self.collision_contriant(deta_trans * self.world.dt,
                                                                                        np.dot(deta_s, deta_speed) +
                                                                                        0.5 * h_coll)},)
            res = minimize(self.fun(udes), udes, constraints=cons)
            if not res.success:
                safe_action_n = udes
                # print('1')
            else:
                safe_action_n = res.x
            # min 1/2x^T*P*x + q^T*x
            # s.t. Gx <= h

            # P = 2 * np.diag(np.ones(4 * num))
            # q = -udes.reshape((4 * num, 1))
            # h_c = np.dot(beta.T, speed_n.reshape(2 * num, 1)) + h0
            # G_c = -self.world.dt * np.dot(beta.T, trans_mat)
            #
            # # cal out of range
            # h_out_range_up = 0.98 - s
            # h_out_range_up = h_out_range_up.reshape(2 * num, 1)
            # h_out_range_down = s + 0.98
            # h_out_range_down = h_out_range_down.reshape(2 * num, 1)
            # G_o = np.concatenate((self.world.dt * trans_mat, -self.world.dt * trans_mat), axis=0)
            # h_o = np.concatenate((h_out_range_up - speed_n.reshape((2 * num, 1)),
            #                       h_out_range_down + speed_n.reshape((2 * num, 1))), axis=0)
            # # cal h_ij
            # h_p = np.array([])
            # G_p = np.array([])
            # speed_n = speed_n.reshape(num, 2)
            # for i in range(num - 1):
            #     for j in range(i + 1, num):
            #         if dist[i, j] >= 0.2:
            #             continue
            #         else:
            #             h_coll = dist[i, j] ** 2 - 4 * self.agents[0].size ** 2
            #             deta_s = s[i, :] - s[j, :]
            #             deta_speed = speed_n[i, :] - speed_n[j, :]
            #             h_p = np.append(h_p, np.dot(deta_s, deta_speed) + 0.5 * h_coll)
            #             deta_trans = deta_s[0] * (trans_mat[2 * i, :] - trans_mat[2 * j, :]) + \
            #                          deta_s[1] * (trans_mat[2 * i + 1, :] - trans_mat[2 * j + 1, :])
            #             G_p = np.append(G_p, -deta_trans * self.world.dt)
            # constraint_num = int(len(G_p)/(4*num))
            # if constraint_num > 0:
            #     G_p = G_p.reshape(constraint_num, 4*num)
            #     h_p = h_p.reshape(constraint_num, 1)
            #     G = np.concatenate((G_c, G_o, G_p), axis=0)
            #     h = np.concatenate((h_c, h_o, h_p), axis=0)
            # else:
            #     G = np.concatenate((G_c, G_o), axis=0)
            #     h = np.concatenate((h_c, h_o), axis=0)
            # safe_action_n = solve_qp(P, q, G, h, solver="quadprog")
            # if safe_action_n is None:
            #     # print('1')
            #     safe_action_n = udes
            # set action for each agent
            for i, agent in enumerate(self.agents):
                action_n[i][1:] = safe_action_n[4 * i:4 * i + 4]
                self._set_action(action_n[i], agent, self.action_space[i])
        else:
            for i, agent in enumerate(self.agents):
                self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n.append(self._get_info(agent))
        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return action_n, obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    if self.second_order_dynamic:
                        agent.action.u[0] += action[0][0]
                        agent.action.u[1] += action[0][1]
                    else:
                        agent.action.u[0] += action[0][1] - action[0][2]
                        agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            if agent.accel is not None:
                self.sensitivity = agent.accel
            agent.action.u *= self.sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        # if mode == 'human':
        #     alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        #     message = ''
        #     for agent in self.world.agents:
        #         comm = []
        #         for other in self.world.agents:
        #             if other is agent: continue
        #             if np.all(other.state.c == 0):
        #                 word = '_'
        #             else:
        #                 word = alphabet[np.argmax(other.state.c)]
        #             message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
        #     print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700, 700)  # 700 is the windows size

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []  # the entities drawn everytime
            self.render_geoms_xform = []
            # set the environment range
            self.render_geoms.append(rendering.make_line((-1, -1), (1, -1)))
            self.render_geoms.append(rendering.make_line((1, -1), (1, 1)))
            self.render_geoms.append(rendering.make_line((1, 1), (-1, 1)))
            self.render_geoms.append(rendering.make_line((-1, 1), (-1, -1)))

            for entity in self.world.agents:
                if entity.r is not None:
                    # coverage range
                    geom2 = rendering.make_circle(entity.r)
                    xform2 = rendering.Transform()
                    geom2.set_color(*np.array([128 / 255, 138 / 255, 135 / 255]), alpha=0.2)
                    geom2.add_attr(xform2)
                    self.render_geoms.append(geom2)
                    self.render_geoms_xform.append(xform2)
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            for landmark in self.world.landmarks:
                # 绘制待覆盖点
                if landmark.r is not None:
                    geom = rendering.make_circle(landmark.r)
                    xform = rendering.Transform()
                    geom.set_color(*np.array([255, 0, 255]), alpha=1.0)
                    geom.add_attr(xform)
                    self.render_geoms.append(geom)
                    self.render_geoms_xform.append(xform)
                # 绘制障碍点
                else:
                    pass

            # 不移动的点位只需要设定一次位置就行了
            n = len(self.world.agents)
            for e, landmark in enumerate(self.world.landmarks):
                # 更新覆盖点位置信息
                if landmark.r is not None:
                    self.render_geoms_xform[2 * n + e].set_translation(*landmark.state.p_pos)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        # calculate the network
        for viewer in self.viewers:
            getpos = np.array([])
            for k in self.world.agents:
                getpos = np.append(getpos, k.state.p_pos)

            num = len(self.agents)
            getpos = getpos.reshape(num, 2)
            dist = squareform(pdist(getpos, 'euclidean'))
            for i in range(num):
                for j in range(i + 1, num):
                    if dist[i, j] <= self.agents[0].R:
                        viewer.draw_line(getpos[i, :], getpos[j, :])

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1.2
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, agent in enumerate(self.world.agents):
                self.render_geoms_xform[2 * e].set_translation(*agent.state.p_pos)
                self.render_geoms_xform[2 * e + 1].set_translation(*agent.state.p_pos)

            # 更新覆盖点颜色
            n = len(self.world.agents)
            for e, landmark in enumerate(self.world.landmarks):
                self.render_geoms[2 * n + e + 4].set_color(*np.array([255, 0, 255]), alpha=landmark.state.energy/landmark.state.coverpointE)

            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i + env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
