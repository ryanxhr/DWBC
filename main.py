import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import wandb
import time

import utils
import get_dataset
from algos import DWBC, WBC, ORIL


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Env: {env_name}, Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--algorithm", default="DWBC")  # Policy name
    parser.add_argument('--env', default="ant-medium-replay-v2")  # environment name
    parser.add_argument("--split_x", default=2, type=int)  # percentile X used to select the dataset
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e5, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    # DWBC
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--alpha", default=20.0, type=float)
    parser.add_argument("--pu_learning", default=False, type=bool)
    parser.add_argument("--eta", default=0.5, type=float)
    parser.add_argument("--normalize", default=True, type=bool)
    args = parser.parse_args()

    file_name = f"{args.algorithm}_{args.env}_{args.split_x}_{args.seed}_{args.pu_learning}_{args.eta}"
    print("---------------------------------------")
    print(f"Algorithm: {args.algorithm}, env: {args.env}, X: {args.split_x}, "
          f"Seed: {args.seed}, PU-learning: {args.pu_learning}, Eta: {args.eta}")
    print("---------------------------------------")

    # wandb
    # wandb.init(project="OffIL", entity="ryanxhr", name=file_name, config=args)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env_e = gym.make(args.env)
    env_id = args.env.split('-')[0]
    if env_id in {'hopper', 'halfcheetah', 'walker2d', 'ant'}:
        env_o = gym.make(f'{env_id}-random-v2')
        exp_num = 10
    else:
        env_o = gym.make(f'{env_id}-cloned-v1')
        exp_num = 100

    # Set seeds
    env_e.seed(args.seed)
    env_e.action_space.seed(args.seed)
    env_o.seed(args.seed)
    env_o.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env_e.observation_space.shape[0]
    action_dim = env_e.action_space.shape[0]

    # Initialize policy
    if args.algorithm == 'DWBC':
        policy = DWBC.DWBC(state_dim, action_dim, args.alpha, args.pu_learning, args.eta)
    elif args.algorithm == 'WBC':
        policy = WBC.WBC(state_dim, action_dim, args.alpha)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    # Load dataset
    if "replay" in args.env:  # setting 1
        dataset_e_raw = env_e.get_dataset()
        split_x = args.split_x
        dataset_e, dataset_o = get_dataset.dataset_setting1(dataset_e_raw, split_x)

    else:  # setting 2 or 3
        dataset_e_raw = env_e.get_dataset()
        dataset_o_raw = env_o.get_dataset()
        split_x = int(exp_num * args.split_x / 100)
        dataset_e, dataset_o = get_dataset.dataset_setting2(dataset_e_raw, dataset_o_raw, split_x, exp_num)

    states_e = dataset_e['observations']
    states_o = dataset_o['observations']
    states_b = np.concatenate([states_e, states_o]).astype(np.float32)

    print('# {} of expert demonstraions'.format(states_e.shape[0]))
    print('# {} of imperfect demonstraions'.format(states_o.shape[0]))

    replay_buffer_e = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer_o = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer_e.convert_D4RL(dataset_e)
    replay_buffer_o.convert_D4RL(dataset_o)

    if args.normalize:
        shift = np.mean(states_b, 0)
        scale = np.std(states_b, 0) + 1e-3
    else:
        shift, scale = 0, 1
    replay_buffer_e.normalize_states(mean=shift, std=scale)
    replay_buffer_o.normalize_states(mean=shift, std=scale)

    evaluations = []
    for t in range(int(args.max_timesteps)):
        policy.train(replay_buffer_e, replay_buffer_o, args.batch_size)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            evaluations.append(eval_policy(policy, args.env, args.seed, shift, scale))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
