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
torch.backends.cudnn.benchmark = True

def main(task, checkpoint, save_dir, db_dirs, eps):
    assert(len(db_dirs) == len(eps))
    agent = torch.load(checkpoint)['agent'] # assuming gpu if not just change map_location
    frame_stack = 3
    action_repeat = 2
    env = dmc.make(task, frame_stack, action_repeat, 0)

    spec = env.action_spec()
    action_shape = spec.shape
    action_min = spec.minimum
    action_max = spec.maximum

    def get_next_actions(db_dir, eps_greedy):
        data = np.load(osp.join(save_dir, db_dir, 'data.npz'), allow_pickle=True)
        next_actions = []
        n_data = data['actions'].shape[0]
        idx_mapping = data['idx_mapping'].item()

        for i in tqdm(range(n_data)):
            next_obs_idxs = [idx_mapping[i] + j for j in range(1, frame_stack+1)] # get the next_obs frames
            next_obs = []
            for noi in next_obs_idxs:
                im = np.asarray(Image.open(osp.join(save_dir, db_dir, 'frames', f'{noi}.png')))
                next_obs.append(np.moveaxis(im, -1, 0)) # move channel to the first dim (pytorch convention)
            next_obs = np.concatenate(next_obs, axis=0)

            if eps_greedy > np.random.rand():
                action = np.random.uniform(low=action_min, high=action_max, size=action_shape)
            else:
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(next_obs, i, eval_mode=True)

            next_actions.append(action)

        with open(osp.join(save_dir, db_dir, f'next_actions_comp.npy'), 'wb') as f:
            np.save(f, np.stack(next_actions))

    # get next actions with appropriate policy 
    for db, ep in zip(db_dirs, eps):
        print(f'DB: {db} | Eps: {ep}')
        get_next_actions(db, ep)

def next_action_for_offline(offline_db, task, checkpoint, save_dir, db_dirs, eps):
    assert(len(db_dirs) == len(eps))
    agent = torch.load(checkpoint)['agent'] # assuming gpu if not just change map_location
    frame_stack = 3
    action_repeat = 2
    env = dmc.make(task, frame_stack, action_repeat, 0)

    spec = env.action_spec()
    action_shape = spec.shape
    action_min = spec.minimum
    action_max = spec.maximum

    def get_next_actions(db_dir, eps_greedy):
        data = np.load(osp.join(save_dir, offline_db, 'data.npz'), allow_pickle=True)
        next_actions = []
        n_data = data['actions'].shape[0]
        idx_mapping = data['idx_mapping'].item()

        for i in tqdm(range(n_data)):
            next_obs_idxs = [idx_mapping[i] + j for j in range(1, frame_stack+1)] # get the next_obs frames
            next_obs = []
            for noi in next_obs_idxs:
                im = np.asarray(Image.open(osp.join(save_dir, offline_db, 'frames', f'{noi}.png')))
                next_obs.append(np.moveaxis(im, -1, 0)) # move channel to the first dim (pytorch convention)
            next_obs = np.concatenate(next_obs, axis=0)

            if eps_greedy > np.random.rand():
                action = np.random.uniform(low=action_min, high=action_max, size=action_shape)
            else:
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(next_obs, i, eval_mode=True)

            next_actions.append(action)

        with open(osp.join(save_dir, db_dir, f'next_actions_ranking.npy'), 'wb') as f:
            np.save(f, np.stack(next_actions))

    # get next actions with appropriate policy 
    for db, ep in zip(db_dirs, eps):
        print(f'Offline DB: {offline_db} | Target DB: {db} | Eps: {ep}')
        get_next_actions(db, ep)


if __name__ == '__main__':
    #db_dirs = ['d_target', 'd_offline_1', 'd_offline_2', \
    #    'd_offline_3', 'd_offline_4', 'd_offline_5', 'd_offline_6']
    db_dirs = ['d_target', 'd_offline_2', 'd_offline_4', 'd_offline_6']
    offline_db = 'd_offline_4'
    eps = [0.0, 0.2, 0.6, 1.0]
    #eps = [0.0, 0.015, 0.05, 0.2]
    #eps = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    root_dir = '/home/jdc396/offline_representation/drq_data'
    task_dirs = {
        'cartpole_swingup': 'cartpole-swingup',
        'cup_catch': 'cup-catch',
        'finger_turn_hard': 'finger-turn-hard',
        'humanoid_run': 'humanoid-run',
        'humanoid_stand': 'humanoid-stand',
        'quadruped_walk': 'quadruped-walk',
        'walker_walk': 'walker-walk',
        'cheetah_run': 'cheetah-run'
    }
    #tasks = ['cartpole_swingup', 'cup_catch', 'finger_turn_hard', 'humanoid_run', 'humanoid_stand', 'quadruped_walk', 'walker_walk']
    #tasks = ['humanoid_stand']
    tasks = ['quadruped_walk']

    checkpoints = {
        'cartpole_swingup': './exp_local/2021.12.19/015554_task=cartpole_swingup/snapshot.pt',
        'cup_catch': './exp_local/2021.12.19/051258_task=cup_catch/snapshot.pt',
        'finger_turn_hard': './exp_local/2021.12.18/133831_task=finger_turn_hard/snapshot.pt',
        'humanoid_run': './exp_local/2021.12.18/133530_task=humanoid_run/snapshot.pt',
        'humanoid_stand': './exp_local/2021.12.18/133539_task=humanoid_stand/snapshot.pt',
        'quadruped_walk': './exp_local/2021.12.18/133623_task=quadruped_walk/snapshot.pt',
        'walker_walk': './exp_local/2021.12.19/025259_task=walker_walk/snapshot.pt',
        'cheetah_run': 'exp_local/2021.12.04/111419_task=cheetah_run/snapshot.pt'
    }

    for task in tasks:
        if task != 'cartpole_swingup':
            print(f'Getting Comparison Next Actions for {task}')
            #main(task, checkpoints[task], osp.join(root_dir, task_dirs[task]), db_dirs, eps)
            next_action_for_offline(offline_db, task, checkpoints[task], osp.join(root_dir, task_dirs[task]), db_dirs, eps)




