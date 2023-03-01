# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
# File func: main func
import os
import pickle
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from arguments import parse_args
from replay_buffer import ReplayBuffer
import multiagent.scenarios as scenarios
from model import openai_actor, openai_critic
from multiagent.environment import MultiAgentEnv


def make_env(scenario_name, arglist):
    """ 
    create the environment from script 
    """
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if arglist.benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done, info_callback=scenario.info)
        # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    actors_cur = [None for _ in range(env.n)]
    critics_cur = [None for _ in range(env.n)]
    actors_tar = [None for _ in range(env.n)]
    critics_tar = [None for _ in range(env.n)]
    optimizers_c = [None for _ in range(env.n)]
    optimizers_a = [None for _ in range(env.n)]

    if arglist.restore:  # restore the model
        for i in range(env.n):
            actors_cur[i] = (torch.load(arglist.old_model_name + 'a_c_{}'.format(i))).to(arglist.device)
            critics_cur[i] = (torch.load(arglist.old_model_name + 'c_c_{}'.format(i))).to(arglist.device)
            actors_tar[i] = (torch.load(arglist.old_model_name + 'a_t_{}'.format(i))).to(arglist.device)
            critics_tar[i] = (torch.load(arglist.old_model_name + 'c_t_{}'.format(i))).to(arglist.device)
            optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
            optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    else:
        for i in range(env.n):
            actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
            critics_cur[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
            actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
            critics_tar[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
            optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
            optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actors_tar = update_train_tar(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_train_tar(critics_cur, critics_tar, 1.0)  # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def update_train_tar(agents_cur, agents_tar, tau):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tau + \
                                (1 - tau) * state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar


def agents_train(arglist, game_step, update_cnt, memory, obs_size, action_size, actors_cur, actors_tar, critics_cur,
                 critics_tar, optimizers_a, optimizers_c):
    """
    use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update
    """
    # update all trainers, if not in display or benchmark mode
    if game_step > arglist.learning_start_step and \
            (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0:
            print('Start training ...')
        # update the target par using the cur
        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            # sample the experience
            _obs_n, _action_n, _rew_n, _obs_n_t, _done_n = memory.sample(
                arglist.batch_size, agent_idx)  # Note_The func is not the same as others

            # --use the data to update the CRITIC
            # set the data to gpu
            rew = torch.tensor(_rew_n, dtype=torch.float, device=arglist.device)
            done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)
            action_n = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n = torch.from_numpy(_obs_n).to(arglist.device, torch.float)
            obs_n_t = torch.from_numpy(_obs_n_t).to(arglist.device, torch.float)
            # cal the loss
            action_tar = torch.cat([a_t(obs_n_t[:, obs_size[idx][0]:obs_size[idx][1]]).detach()
                                    for idx, a_t in enumerate(actors_tar)], dim=1)  # get the action in next state
            q = critic_c(obs_n, action_n).reshape(-1)  # q value in current state
            q_t = critic_t(obs_n_t, action_tar).reshape(-1)  # q value in next state
            tar_value = q_t * arglist.gamma * done_n + rew  # q_*gamma*done + reward
            loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
            # update the parameters
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
            opt_c.step()

            # --use the data to update the ACTOR
            # There is no need to cal other agent's action
            model_out, policy_c_new = actor_c(
                obs_n[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the action of this agent
            action_n[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n, action_n)))

            opt_a.zero_grad()
            (1e-3 * loss_pse + loss_a).backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
            opt_a.step()

        # update the target network
        actors_tar = update_train_tar(actors_cur, actors_tar, arglist.tau)
        critics_tar = update_train_tar(critics_cur, critics_tar, arglist.tau)

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar


def train(arglist):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env(arglist.scenario_name, arglist)

    print('=============================')
    print('=1 Env {} is right ...'.format(arglist.scenario_name))
    print('=============================')

    """step2: create agents"""
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    action_shape_n = [env.action_space[i].n for i in range(env.n)]  # no need for stop bit
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers(env, obs_shape_n, action_shape_n, arglist)
    memory = ReplayBuffer(arglist.memory_size)

    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0  # game steps until now
    update_cnt = 0  # numbers of updating the models
    agent_info = [[[]]]  # placeholder for benchmarking info
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    game_step_per_episode = []
    is_collision_h = []
    is_connected_h = []
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a

    print('=3 starting iterations ...')
    print('=============================')
    start_time = time.time()
    obs_n = env.reset()

    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data
        # end_time = time.time()
        # print(int(end_time - start_time))
        print(episode_gone)
        if game_step > 1 and episode_gone % 500 == 0:
            end_time = time.time()
            mean_ep_r = round(np.mean(episode_rewards[-500:-1]), 3)
            mean_step = round(np.mean(game_step_per_episode[-500:-1]), 3)
            mean_connected_rate = round(np.mean(is_connected_h[-500:-1]), 3)
            mean_collision_rate = round(np.mean(is_collision_h[-500:-1]), 3)
            print('=Training: steps:{} episode:{} reward:{} step:{} connected:{} collision:{} time:{}'
                  .format(game_step, episode_gone, mean_ep_r, mean_step, mean_connected_rate, mean_collision_rate,
                          int(end_time - start_time)))
            # mean_agents_r = [round(np.mean(agent_rewards[idx][-500:-1]), 2) for idx in range(env.n)]
            # print('Agent reward:{}'.format(mean_agents_r))
        game_step_cur = 0
        is_collision = False
        is_connected = True
        for episode_cnt in range(arglist.max_episode_len):
            # get action
            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy()
                        for agent, obs in zip(actors_cur, obs_n)]

            # interact with env
            obs_n_t, rew_n, done_n, info_n = env.step(action_n)

            # save the experience
            memory.add(obs_n, np.concatenate(action_n), rew_n, obs_n_t, done_n)
            episode_rewards[-1] += np.sum(rew_n)
            for i, rew in enumerate(rew_n):
                agent_rewards[i][-1] += rew

            # train our agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train(
                arglist, game_step, update_cnt, memory, obs_size, action_size,
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)

            # update the obs_n
            game_step += 1
            game_step_cur +=1
            obs_n = obs_n_t
            done = all(done_n)
            is_collision = is_collision or any([info_n[i]['collision'] for i in range(env.n)])
            is_connected = is_connected and all([info_n[i]['connected'] for i in range(env.n)])
            terminal = (episode_cnt >= arglist.max_episode_len - 1)
            # env.render()
            # time.sleep(0.01)
            if done or terminal:
                obs_n = env.reset()
                agent_info.append([[]])
                episode_rewards.append(0)
                game_step_per_episode.append(game_step_cur)
                is_connected_h.append(int(is_connected))
                is_collision_h.append(int(is_collision))
                for a_r in agent_rewards:
                    a_r.append(0)
                break

        # save the model
        if episode_gone > 0 and (episode_gone + 1) % arglist.fre_save_model == 0:
            time_now = time.strftime('%m%d_%H%M%S')
            print('=time:{} episode:{} step:{}        save'.format(time_now, episode_gone + 1, game_step))
            model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format(
                arglist.scenario_name, time_now, (episode_gone + 1)))
            if not os.path.exists(model_file_dir):  # make the path
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                    enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))
    # save the curves
    rew_file_name = arglist.scenario_name + '_rewards.pkl'
    with open(rew_file_name, 'wb') as fp:
        pickle.dump(episode_rewards, fp)
    game_step_file_name = arglist.scenario_name + '_game_steps.pkl'
    with open(game_step_file_name, 'wb') as fp:
        pickle.dump(game_step_per_episode, fp)
    connected_file_name = arglist.scenario_name + '_connected.pkl'
    with open(connected_file_name, 'wb') as fp:
        pickle.dump(is_connected_h, fp)
    collision_file_name = arglist.scenario_name + '_collision.pkl'
    with open(collision_file_name, 'wb') as fp:
        pickle.dump(is_collision_h, fp)

if __name__ == '__main__':
    arg = parse_args()
    train(arg)
