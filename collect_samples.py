import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

from tqdm import tqdm
import numpy as np
from PIL import Image
import os.path as osp
import yaml
import cv2

torch.backends.cudnn.benchmark = True

def get_score(agent, task_name, ep_greedy, n_samples):
    total_rewards, scores  = 0, []
    frame_stack = 3
    action_repeat = 2

    env = dmc.make(task_name, frame_stack, action_repeat, 0)

    spec = env.action_spec()
    action_shape = spec.shape
    action_min = spec.minimum
    action_max = spec.maximum

    time_step = env.reset()
    for step in tqdm(range(n_samples)):

        if ep_greedy > np.random.rand():
            action = np.random.uniform(low=action_min, high=action_max, size=action_shape)
        else:
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation, step, eval_mode=True)

        time_step = env.step(action)

        total_rewards += time_step.reward
        if time_step.last():
            time_step = env.reset()
            scores.append(total_rewards)
            total_rewards = 0
    print(f'Eps: {ep_greedy} | Mean score: {np.mean(scores)}')

def get_samples(
    agent,
    save_dir,
    task_name,
    ep_greedy,
    db_type,
    n_samples,
):
    total_rewards, scores  = 0, []
    frame_stack = 3
    action_repeat = 2


    actions, rewards, dones, next_actions = [], [], [], []
    idx2frame = {} # create mapping
    frame_idx, pointer, frames, raw_frames = 0, 0, [], []

    env = dmc.make(task_name, frame_stack, action_repeat, 0)

    spec = env.action_spec()
    action_shape = spec.shape
    action_min = spec.minimum
    action_max = spec.maximum

    time_step = env.reset()
    for step in tqdm(range(n_samples)):
        # Add frames
        if len(frames) == 0:
            frames = np.array_split(time_step.observation, frame_stack, axis=0)
        else:
            frames.append(time_step.observation[-3:])

        # for the larget 100 x 100 images (for random crops)
        if len(raw_frames) == 0:
            raw_frames = np.array_split(time_step.raw, frame_stack, axis=0)
        else:
            raw_frames.append(time_step.raw[-3:])

        if ep_greedy > np.random.rand():
            action = np.random.uniform(low=action_min, high=action_max, size=action_shape)
        else:
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation, step, eval_mode=True)

        time_step = env.step(action)

        with torch.no_grad(), utils.eval_mode(agent):
            next_action = agent.act(time_step.observation, step, eval_mode=True)
        next_actions.append(next_action)

        actions.append(action)
        rewards.append(time_step.reward)
        dones.append(time_step.last())
        total_rewards += time_step.reward
        idx2frame[step] = frame_idx + pointer
        pointer += 1
        if time_step.last():
            frames.append(time_step.observation[-3:])
            raw_frames.append(time_step.raw[-3:])
            assert(len(frames) == len(raw_frames))
            for i, (frame, raw_frame) in enumerate(zip(frames, raw_frames)):
                im = Image.fromarray(np.moveaxis(frame, 0, -1))
                raw_im = Image.fromarray(np.moveaxis(raw_frame, 0, -1))
                im.save(osp.join(save_dir, db_type, 'frames', f'{frame_idx}.png'))
                raw_im.save(osp.join(save_dir, db_type, 'big_frames', f'{frame_idx}.png'))
                frame_idx += 1
            frames = []
            raw_frames = []
            time_step = env.reset()
            scores.append(total_rewards)
            pointer, total_rewards = 0, 0


    # Save rest of the frames
    assert(len(frames) == len(raw_frames))
    if len(frames) > 0:
        frames.append(time_step.observation[-3:])
        raw_frames.append(time_step.raw[-3:])
        for i, (frame, raw_frame) in enumerate(zip(frames, raw_frames)):
            im = Image.fromarray(np.moveaxis(frame, 0, -1))
            raw_im = Image.fromarray(np.moveaxis(raw_frame, 0, -1))
            im.save(osp.join(save_dir, db_type, 'frames', f'{frame_idx}.png'))
            raw_im.save(osp.join(save_dir, db_type, 'big_frames', f'{frame_idx}.png'))
            frame_idx += 1

    num_traj = len(scores)
    avg_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    std_score = np.std(scores)
    print(avg_score)
    print(num_traj)
    with open(osp.join(save_dir, db_type, 'data.npz'), 'wb') as f:
        np.savez(
            f,
            actions=np.stack(actions),
            dones=np.array(dones),
            rewards=np.array(rewards),
            idx_mapping=idx2frame,
            num_traj=num_traj,
            avg_score=avg_score)

    # Given our Target Policy, get pi(s')
    with open(osp.join(save_dir, db_type, 'next_actions.npy'), 'wb') as f:
        np.save(f, np.stack(next_actions))
    return {
        'num_trajs': num_traj,
        'avg_undiscounted': float(avg_score),
        'min_undiscounted': float(min_score),
        'max_undiscounted': float(max_score),
        'std_undiscounted': float(std_score)
    }

def get_eval(data_dir, target_dir):
    #data_dir = '/home/jdc396/offline_representation/drq_data/cheetah-run'
    #target_dir = 'd_offline_62'
    #datafile = 'data/cheetah_run/d_target/data.npz'
    datafile = osp.join(data_dir, target_dir, 'data.npz')
    gamma = 0.99

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

    #sample_idxs = []
    #for idx in starts:
    #    sample_idxs.append(np.arange(0, 50) + idx) # NOTE: 50 since we have action repeat 2
    #sample_idxs = np.stack(sample_idxs).flatten()
    #eval_idxs = np.random.choice(sample_idxs, 100, replace=False) # We want 100 samples for rmse
    eval_idxs = np.random.choice(starts, 100, replace=False) # We want 500 samples for rmse
    #eval_idxs = np.arange(starts[10], starts[11])
    print(len(eval_idxs))

    print(f'Mean Discounted Return of 100 samples: {returns[eval_idxs].mean()}')
    print(f'STD Discounted Return of 100 samples: {returns[eval_idxs].std()}')
    #print(f'Mean Discounted Return of 100 samples: {returns[eval_idxs].mean()}')
    #print(f'STD Discounted Return of 100 samples: {returns[eval_idxs].std()}')

    idx_mapping = data['idx_mapping'].item()

    obs_idxs = [idx_mapping[k] for k in eval_idxs]
    actions = data['actions'][eval_idxs]
    targets = returns[eval_idxs]

    #with open('data/cheetah_run/eval.npz', 'wb') as f:
    with open(osp.join(data_dir, target_dir, 'eval_init.npz'), 'wb') as f:
        np.savez(f, observations=obs_idxs, actions=actions, returns=targets)

    data = returns[eval_idxs]
    return {
        'avg_eval': float(data.mean()),
        'min_eval': float(data.min()),
        'max_eval': float(data.max()),
        'std_eval': float(data.std())
    }

def min_max_traj(root_dir, target_dir, task='cheetah-run'):
    datafile = osp.join(root_dir, target_dir, 'data.npz')
    gamma = 0.99

    data = np.load(datafile, allow_pickle=True)

    # Grab 100 trajectories
    data_len = np.where(data['dones'])[0]
    starts = np.concatenate([[0], np.where(data['dones'])[0][:-1] + 1], axis=0) # get the starts
    dones = data['dones']
    rewards = data['rewards']

    scores, ep_reward = [], 0
    for d, r in zip(dones, rewards):
        ep_reward += r
        if d:
            scores.append(ep_reward)
            ep_reward = 0
    min_score, max_score, mean_score = np.min(scores), np.max(scores), np.mean(scores)
    min_idx, max_idx = np.argmin(scores), np.argmax(scores)
    min_idx_range = list(range(starts[min_idx], starts[min_idx+1]))
    max_idx_range = list(range(starts[max_idx], starts[max_idx+1]))

    idx_mapping = data['idx_mapping'].item()

    def get_traj(idx_range):
        traj = []
        for idx in idx_range:
            im = np.asarray(Image.open(osp.join(root_dir, target_dir, 'big_frames', f'{idx}.png')))
            traj.append(im)
        return np.stack(traj)

    def create_mp4(frames, video_name, fps=30):
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        size = frames[0].shape[:-1] # assuming (H, W, C)
        grayscale = frames[0].shape[-1] == 1
        video = cv2.VideoWriter(osp.join(root_dir, target_dir, f'{video_name}.mp4'), codec, fps, size)
        for frame in frames:
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame) # Note assumes non grayscale is BGR.....should convert
        video.release()

    min_traj = get_traj(min_idx_range)
    max_traj = get_traj(max_idx_range)
    meta_data = {'min_score': min_score, 'max_score': max_score, 'mean_score': mean_score}

    with open(osp.join(root_dir, target_dir, 'trajs.npz'), 'wb') as f:
        np.savez(f, min_traj=min_traj, max_traj=max_traj, meta=meta_data)
    print(min_score, max_score, mean_score)
    print(min_traj.shape)
    print(max_traj.shape)

    print('Creating Videos')
    create_mp4(min_traj, video_name=f'{task}_{int(min_score)}')
    create_mp4(max_traj, video_name=f'{task}_{int(max_score)}')

def get_db_stats(root_dir, target_dir):
    datafile = osp.join(root_dir, target_dir, 'data.npz')
    gamma = 0.99

    data = np.load(datafile, allow_pickle=True)

    data_len = np.where(data['dones'])[0][-1]
    starts = np.concatenate([[0], np.where(data['dones'])[0][:-2] + 1], axis=0) # get the starts
    dones = data['dones'][:data_len]
    rewards = data['rewards'][:data_len]
    returns = np.zeros(data_len)

    returns[-1] = rewards[-1]
    for n in range(data_len-2, -1, -1):
        returns[n] = rewards[n] + gamma*returns[n+1]*(1-dones[n])
    discounted_rew = returns[starts]

    def get_stats(x):
        mean, minimum, maximum, std = np.mean(x), np.min(x), np.max(x), np.std(x)
        return dict(
            mean_discounted=float(mean),
            min_discounted=float(minimum),
            max_discounted=float(maximum),
            std_discounted=float(std))

    discounted_stats = get_stats(discounted_rew)
    return discounted_stats

def main(save_dir, checkpoint, task_name, n_samples=int(1e5)):
    if not osp.isdir(save_dir):
        os.mkdir(save_dir)
    agent = torch.load(checkpoint)['agent']
    torch.save({'actor': agent.actor.state_dict(),
                'encoder': agent.encoder.state_dict(),
                'critic': agent.critic.state_dict()}, osp.join(save_dir, 'chkpt.pt'))
    #eps = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    #eps = [0.0, 0.2, 0.6, 1.0]
    eps = [0.0, 0.015, 0.05, 0.2]
    #names = ['d_target'] + [f'd_offline_{i}' for i in range(1,7)]
    names = ['d_target'] + [f'd_offline_{i}' for i in [2,4,6]]
    metadata = {}
    for ep, db_name in zip(eps, names):

        # Create Directories
        if not osp.isdir(osp.join(save_dir, db_name)):
            os.makedirs(osp.join(save_dir, db_name, 'frames'))
            os.mkdir(osp.join(save_dir, db_name, 'big_frames'))

        # collect samples
        info = get_samples(agent, save_dir, task_name, ep, db_name, n_samples)
        discounted_info = get_db_stats(save_dir, db_name)
        info.update(discounted_info)
        # create eval data
        eval_info = get_eval(save_dir, db_name)
        # create videos
        min_max_traj(save_dir, db_name, task=task_name)
        metadata[db_name] = {'db': info, 'eval': eval_info}

    with open(osp.join(save_dir, 'metadata.yml'), 'w+') as f:
        yaml.dump(metadata, f, default_flow_style=False)

def main_score(checkpoint, task_name, n_samples=5000):
    agent = torch.load(checkpoint)['agent']
    #eps = [0.01, 0.025, 0.05, 0.075, 0.1]
    eps = [0.015]
    for ep in eps:
        get_score(agent, task_name, ep, n_samples)

def main_eval(save_dir, checkpoint, task_name):
    if not osp.isdir(save_dir):
        os.mkdir(save_dir)
    agent = torch.load(checkpoint)['agent']
    #torch.save({'actor': agent.actor.state_dict(),
    #            'encoder': agent.encoder.state_dict(),
    #            'critic': agent.critic.state_dict()}, osp.join(save_dir, 'chkpt.pt'))
    #eps = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    #eps = [0.0, 0.2, 0.6, 1.0]
    eps = [0.0, 0.015, 0.05, 0.2]
    #names = ['d_target'] + [f'd_offline_{i}' for i in range(1,7)]
    names = ['d_target'] + [f'd_offline_{i}' for i in [2,4,6]]
    metadata = {}
    for ep, db_name in zip(eps, names):

        # Create Directories
        if not osp.isdir(osp.join(save_dir, db_name)):
            os.makedirs(osp.join(save_dir, db_name, 'frames'))
            os.mkdir(osp.join(save_dir, db_name, 'big_frames'))

        # create eval data
        eval_info = get_eval(save_dir, db_name)

if __name__ == '__main__':
    #checkpoint = 'exp_local/2021.11.23/121842_task=cheetah_run/snapshot.pt'
    #checkpoint = 'exp_local/2021.12.04/111419_task=cheetah_run/snapshot.pt'
    #checkpoint = 'exp_local/2021.12.18/133623_task=quadruped_walk/snapshot.pt'
    #checkpoint = 'exp_local/2021.12.18/133831_task=finger_turn_hard/snapshot.pt'
    #checkpoint = 'exp_local/2021.12.19/015554_task=cartpole_swingup/snapshot.pt'
    #checkpoint = 'exp_local/2021.12.19/025259_task=walker_walk/snapshot.pt'
    #checkpoint = 'exp_local/2021.12.19/051258_task=cup_catch/snapshot.pt'
    checkpoint = 'exp_local/2021.12.18/133539_task=humanoid_stand/snapshot.pt'
    #checkpoint = 'exp_local/2021.12.18/133530_task=humanoid_run/snapshot.pt'
    save_dir = '/home/jdc396/offline_representation/drq_data/humanoid-stand'
    #save_dir = '/home/jdc396/offline_representation/drq_data/cheetah-run'
    task_name = 'humanoid_stand'
    #task_name = 'cheetah_run'
    #n_samples = int(1e5)

    #main(save_dir, checkpoint, task_name, n_samples=n_samples)
    #main_score(checkpoint, task_name)
    main_eval(save_dir, checkpoint, task_name)

