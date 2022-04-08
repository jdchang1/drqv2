import numpy as np
import os.path as osp

data_dir = '/home/jdc396/offline_representation/drq_data/humanoid-stand'
target_dir = 'd_target'
#datafile = 'data/cheetah_run/d_target/data.npz'
datafile = osp.join(data_dir, target_dir, 'data.npz')
gamma = 0.999

data = np.load(datafile, allow_pickle=True)

# Grab 100 trajectories
data_len = np.where(data['dones'])[0][99]
starts = np.concatenate([[0], np.where(data['dones'])[0][:99] + 1], axis=0) # get the starts
dones = data['dones'][:data_len]
rewards = data['rewards'][:data_len]
returns = np.zeros(data_len)

scores, ep_reward = [], 0
for d, r in zip(dones, rewards):
    ep_reward += r
    if d:
        scores.append(ep_reward)
        ep_reward = 0
print(np.mean(scores))

returns[-1] = rewards[-1]
for n in range(data_len-2, -1, -1):
    returns[n] = rewards[n] + gamma*returns[n+1]*(1-dones[n])

sample_idxs = []
for idx in starts:
    sample_idxs.append(np.arange(0, 50) + idx) # NOTE: 50 since we have action repeat 2
sample_idxs = np.stack(sample_idxs).flatten()
eval_idxs = np.random.choice(sample_idxs, 100, replace=False) # We want 100 samples for rmse

print(f'Mean Discounted Return of 100 samples: {returns[eval_idxs].mean()}')
print(f'STD Discounted Return of 100 samples: {returns[eval_idxs].std()}')

idx_mapping = data['idx_mapping'].item()

obs_idxs = [idx_mapping[k] for k in eval_idxs]
actions = data['actions'][eval_idxs]
targets = returns[eval_idxs]

#with open('data/cheetah_run/eval.npz', 'wb') as f:
with open(osp.join(data_dir, target_dir, 'eval_999.npz'), 'wb') as f:
    np.savez(f, observations=obs_idxs, actions=actions, returns=targets)

