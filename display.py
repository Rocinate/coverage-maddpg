import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

import multiagent.scenarios as scenarios
from arguments import parse_args
from multiagent.environment import MultiAgentEnv


def make_env(scenario_name, arglist):
    """
    create the environment from script
    """
    scenario = scenarios.load(scenario_name + ".py").Scenario(evaluate=True)
    world = scenario.make_world()
    if arglist.benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done, info_callback=scenario.info, safe_control=False)
    return env


def get_trainers(env, arglist):
    """ load the model """
    actors_tar = [torch.load(arglist.model_name + 'a_c_{}.pt'.format(agent_idx), map_location=arglist.device)
                  for agent_idx in range(env.n)]

    return actors_tar


def enjoy(arglist):
    """
    This func is used for testing the model
    """
    np.random.seed(1)
    episode_step = 0
    """ init the env """
    env = make_env(arglist.scenario_name, arglist)

    """ init the agents """
    actors_tar = get_trainers(env, arglist)

    """ interact with the env """
    obs_n = env.reset()
    is_collision = False
    is_connected = True
    test_time = 100
    connect_time = 0
    collision_time = 0
    episode = 0
    trajectory = []
    coverage_time = 0
    coverage_time_h = []
    while True:
        # update the episode step number
        episode_step += 1

        # get action
        action_n = []
        for actor, obs in zip(actors_tar, obs_n):
            model_out, _ = actor(torch.from_numpy(obs).to(arglist.device, torch.float), model_original_out=True)
            action_n.append(F.softmax(model_out, dim=-1).detach().cpu().numpy())
        if arglist.save_pos:
            pos = np.array([])
            for k in env.world.agents:
                pos = np.append(pos, k.state.p_pos)
            trajectory.append(pos)
        # interact with env
        _, obs_n, rew_n, done_n, info_n = env.step(action_n)

        # update the flag
        is_collision = is_collision or any([info_n[i]['collision'] for i in range(env.n)])
        is_connected = is_connected and all([info_n[i]['connected'] for i in range(env.n)])
        if all([info_n[i]['coverage'] for i in range(env.n)]):
            coverage_time = coverage_time + 1
        if any([info_n[i]['collision'] for i in range(env.n)]):
            collision_time = collision_time + 1
        if all([info_n[i]['connected'] for i in range(env.n)]):
            connect_time = connect_time + 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)

        # reset the env
        if done or terminal:
            episode = episode + 1
            episode_step = 0
            obs_n = env.reset()
            coverage_time_h.append(coverage_time)
            print('collision:{} connected:{} coverage:{}'.format(is_collision, is_connected, coverage_time))
            coverage_time = 0
            is_collision = False
            is_connected = True
        # render the env
        if arglist.display:
            env.render()
            time.sleep(0.1)
        if episode >= test_time:
            break
    if arglist.save_pos:
        file_name = arglist.plots_dir + 'trajectory_save' + '.pkl'
        with open(file_name, 'wb') as fp:
            pickle.dump(trajectory, fp)
    print('Total collision:{} connected:{} coverage:{} in {} test'.format(collision_time/(episode*arglist.max_episode_len),
                                                                          connect_time/(episode*arglist.max_episode_len),
                                                                          np.mean(coverage_time_h),
                                                                          episode))
if __name__ == '__main__':
    arglist = parse_args()
    enjoy(arglist)
