import numpy as np


def qlearning_dataset(dataset, terminate_on_end=False):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    """
    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        final_timestep = dataset['timeouts'][i]
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def dataset_setting1(dataset, split_x, terminate_on_end=False):
    """
    Returns D_e and D_o of setting 1 in the paper.
    """
    N = dataset['rewards'].shape[0]
    return_traj = []
    obs_traj = [[]]
    next_obs_traj = [[]]
    action_traj = [[]]
    reward_traj = [[]]
    done_traj = [[]]

    for i in range(N-1):
        obs_traj[-1].append(dataset['observations'][i].astype(np.float32))
        next_obs_traj[-1].append(dataset['observations'][i+1].astype(np.float32))
        action_traj[-1].append(dataset['actions'][i].astype(np.float32))
        reward_traj[-1].append(dataset['rewards'][i].astype(np.float32))
        done_traj[-1].append(bool(dataset['terminals'][i]))

        final_timestep = dataset['timeouts'][i]
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            return_traj.append(np.sum(reward_traj[-1]))
            obs_traj.append([])
            next_obs_traj.append([])
            action_traj.append([])
            reward_traj.append([])
            done_traj.append([])

    # select top 20% return trajectories
    inds_all = np.argsort(return_traj)[::-1]
    succ_num = int(len(inds_all) * 0.2)
    inds_top20 = inds_all[:succ_num]
    inds_e = inds_top20[::split_x]
    inds_e = list(inds_e)
    inds_all = list(inds_all)
    inds_o = set(inds_all) - set(inds_e)
    inds_o = list(inds_o)

    print('# select {} trajs in mixed dataset as D_e'.format(len(inds_e)))
    print('# select {} trajs in mixed dataset as D_o'.format(len(inds_o)))

    obs_traj_e = [obs_traj[i] for i in inds_e]
    next_obs_traj_e = [next_obs_traj[i] for i in inds_e]
    action_traj_e = [action_traj[i] for i in inds_e]
    reward_traj_e = [reward_traj[i] for i in inds_e]
    done_traj_e = [done_traj[i] for i in inds_e]

    obs_traj_o = [obs_traj[i] for i in inds_o]
    next_obs_traj_o = [next_obs_traj[i] for i in inds_o]
    action_traj_o = [action_traj[i] for i in inds_o]
    reward_traj_o = [reward_traj[i] for i in inds_o]
    done_traj_o = [done_traj[i] for i in inds_o]

    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)

    dataset_e = {
        'observations': concat_trajectories(obs_traj_e),
        'actions': concat_trajectories(action_traj_e),
        'next_observations': concat_trajectories(next_obs_traj_e),
        'rewards': concat_trajectories(reward_traj_e),
        'terminals': concat_trajectories(done_traj_e),
    }

    dataset_o = {
        'observations': concat_trajectories(obs_traj_o),
        'actions': concat_trajectories(action_traj_o),
        'next_observations': concat_trajectories(next_obs_traj_o),
        'rewards': concat_trajectories(reward_traj_o),
        'terminals': concat_trajectories(done_traj_o),
    }

    return dataset_e, dataset_o


def dataset_setting2(dataset1, dataset2, split_x, terminate_on_end=False):
    """
    Returns D_e and D_o of setting 2 in the paper.
    """
    dataset_o = dataset_T_trajs(dataset2, 1000)
    dataset_e, dataset_o_extra = dataset_split_expert(dataset1, split_x)
    for key in dataset_o.keys():
        dataset_o[key] = np.concatenate([dataset_o[key], dataset_o_extra[key]], 0)
    return dataset_e, dataset_o


def dataset_split_expert(dataset, split_x, terminate_on_end=False):
    """
    Returns D_e and expert data in D_o of setting 2 in the paper.
    """
    N = dataset['rewards'].shape[0]
    return_traj = []
    obs_traj = [[]]
    next_obs_traj = [[]]
    action_traj = [[]]
    reward_traj = [[]]
    done_traj = [[]]

    for i in range(N-1):
        obs_traj[-1].append(dataset['observations'][i].astype(np.float32))
        next_obs_traj[-1].append(dataset['observations'][i+1].astype(np.float32))
        action_traj[-1].append(dataset['actions'][i].astype(np.float32))
        reward_traj[-1].append(dataset['rewards'][i].astype(np.float32))
        done_traj[-1].append(bool(dataset['terminals'][i]))

        final_timestep = dataset['timeouts'][i]
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            return_traj.append(np.sum(reward_traj[-1]))
            obs_traj.append([])
            next_obs_traj.append([])
            action_traj.append([])
            reward_traj.append([])
            done_traj.append([])

    # select 10 trajectories
    inds_all = list(range(len(obs_traj)))
    succ_num = 10
    inds_10 = inds_all[:succ_num]
    inds_e = inds_10[::split_x]
    inds_e = list(inds_e)
    inds_10 = list(inds_10)
    inds_o = set(inds_10) - set(inds_e)
    inds_o = list(inds_o)

    print('# select {} trajs in expert dataset as D_e'.format(len(inds_e)))
    print('# select {} trajs in expert dataset as expert data in D_o'.format(len(inds_o)))

    obs_traj_e = [obs_traj[i] for i in inds_e]
    next_obs_traj_e = [next_obs_traj[i] for i in inds_e]
    action_traj_e = [action_traj[i] for i in inds_e]
    reward_traj_e = [reward_traj[i] for i in inds_e]
    done_traj_e = [done_traj[i] for i in inds_e]

    obs_traj_o = [obs_traj[i] for i in inds_o]
    next_obs_traj_o = [next_obs_traj[i] for i in inds_o]
    action_traj_o = [action_traj[i] for i in inds_o]
    reward_traj_o = [reward_traj[i] for i in inds_o]
    done_traj_o = [done_traj[i] for i in inds_o]

    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)

    dataset_e = {
        'observations': concat_trajectories(obs_traj_e),
        'actions': concat_trajectories(action_traj_e),
        'next_observations': concat_trajectories(next_obs_traj_e),
        'rewards': concat_trajectories(reward_traj_e),
        'terminals': concat_trajectories(done_traj_e),
    }

    dataset_o = {
        'observations': concat_trajectories(obs_traj_o),
        'actions': concat_trajectories(action_traj_o),
        'next_observations': concat_trajectories(next_obs_traj_o),
        'rewards': concat_trajectories(reward_traj_o),
        'terminals': concat_trajectories(done_traj_o),
    }

    return dataset_e, dataset_o


def dataset_T_trajs(dataset, T, terminate_on_end=False):
    """
    Returns T trajs from dataset.
    """
    N = dataset['rewards'].shape[0]
    return_traj = []
    obs_traj = [[]]
    next_obs_traj = [[]]
    action_traj = [[]]
    reward_traj = [[]]
    done_traj = [[]]

    for i in range(N-1):
        obs_traj[-1].append(dataset['observations'][i].astype(np.float32))
        next_obs_traj[-1].append(dataset['observations'][i+1].astype(np.float32))
        action_traj[-1].append(dataset['actions'][i].astype(np.float32))
        reward_traj[-1].append(dataset['rewards'][i].astype(np.float32))
        done_traj[-1].append(bool(dataset['terminals'][i]))

        final_timestep = dataset['timeouts'][i]
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            return_traj.append(np.sum(reward_traj[-1]))
            obs_traj.append([])
            next_obs_traj.append([])
            action_traj.append([])
            reward_traj.append([])
            done_traj.append([])

    # select T trajectories
    inds_all = list(range(len(obs_traj)))
    inds = inds_all[:T]
    inds = list(inds)

    print('# select {} trajs in the dataset'.format(T))

    obs_traj = [obs_traj[i] for i in inds]
    next_obs_traj = [next_obs_traj[i] for i in inds]
    action_traj = [action_traj[i] for i in inds]
    reward_traj = [reward_traj[i] for i in inds]
    done_traj = [done_traj[i] for i in inds]

    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)

    return {
        'observations': concat_trajectories(obs_traj),
        'actions': concat_trajectories(action_traj),
        'next_observations': concat_trajectories(next_obs_traj),
        'rewards': concat_trajectories(reward_traj),
        'terminals': concat_trajectories(done_traj),
    }
