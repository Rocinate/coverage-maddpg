# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch

import time
import torch
import argparse

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
time_now = time.strftime('%y%m_%d%H%M')


def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning experiments for multi-agent environments")

    # environment
    parser.add_argument("--scenario_name", type=str, default="simple_coverage", help="name of the scenario script")
    parser.add_argument("--start_time", type=str, default=time_now, help="the time when start the game")
    parser.add_argument("--max_episode_len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--max_episode", type=int, default=100000, help="maximum episode length")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")

    # core training parameters
    parser.add_argument("--device", default=device, help="torch device ")
    parser.add_argument("--safe_control", type=bool, default=True, help="adopt the CBF ")
    parser.add_argument("--learning_start_step", type=int, default=10000, help="learning start steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--learning_fre", type=int, default=100, help="learning frequency")
    parser.add_argument("--tau", type=int, default=0.01, help="how depth we exchange the par of the nn")
    parser.add_argument("--lr_a", type=float, default=0.002, help="learning rate for adam optimizer")
    parser.add_argument("--lr_c", type=float, default=0.002, help="learning rate for adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--memory_size", type=int, default=1e6, help="number of data stored in the memory")
    parser.add_argument("--num_units_1", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--num_units_2", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--num_units_openai", type=int, default=128, help="number of units in the mlp")

    # checkpointing
    parser.add_argument("--fre_save_model", type=int, default=2000,
                        help="the number of the episode for saving the model")
    parser.add_argument("--save_dir", type=str, default="models", help="directory in which training state and model \
    should be saved")
    parser.add_argument("--old_model_name", type=str, default="models/simple_coverage_0307_141936_20000/",
                        help="directory in which training state and model are loaded")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    # evaluation
    parser.add_argument("--model_name", type=str, default="models/simple_coverage_0302_115811_20000/",
                        help="directory in which evaluated model is loaded")
    parser.add_argument("--save_pos", type=bool, default=False)
    parser.add_argument("--display", type=bool, default=True)
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()
