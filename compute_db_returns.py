import numpy as np
import os.path as osp

def get_db_stats(datafile):
    gamma = 0.99

    data = np.load(datafile, allow_pickle=True)

    # Grab 100 trajectories
    data_len = np.where(data['dones'])[0][-1]
    starts = np.concatenate([[0], np.where(data['dones'])[0][:-2] + 1], axis=0) # get the starts
    dones = data['dones'][:data_len]
    rewards = data['rewards'][:data_len]
    returns = np.zeros(data_len)

    scores, ep_reward = [], 0
    for d, r in zip(dones, rewards):
        ep_reward += r
        if d:
            scores.append(ep_reward)
            ep_reward = 0

    returns[-1] = rewards[-1]
    for n in range(data_len-2, -1, -1):
        returns[n] = rewards[n] + gamma*returns[n+1]*(1-dones[n])
    discounted_rew = returns[starts]

    def get_stats(x):
        mean, minimum, maximum, std = np.mean(x), np.min(x), np.max(x), np.std(x)
        return dict(mean=mean, min=minimum, max=maximum, std=std)

    undiscounted_stats = get_stats(scores)
    discounted_stats = get_stats(discounted_rew)

    for k in undiscounted_stats.keys():
        print(f'{k} Discounted: {discounted_stats[k]}')
        print(f'{k} Undiscounted: {undiscounted_stats[k]}')

if __name__ == '__main__':
    target_datafile = 'data/cheetah_run/d_target/data.npz'
    offline_datafile = 'data/cheetah_run/d_offline/data.npz'

    print('Target DB Stats')
    get_db_stats(target_datafile)

    print('Offline DB Stats')
    get_db_stats(offline_datafile)
