import datetime
import io
import random
import traceback
from collections import defaultdict

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from dm_env import specs
import dmc
import utils

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())

def store_episode(episode, eps_idx, save_dir):
    eps_len = episode_len(episode)
    ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
    save_episode(episode, save_dir / eps_fn)

def collect(env, agent, save_dir, num_episodes=200, num_ep_start=None):
    #BoundedArray(shape=(9, 84, 84), dtype=dtype('uint8'), name='observation', minimum=0, maximum=255)
    data_specs = (specs.Array((9, 64, 64), np.uint8, 'raw'),
                  env.action_spec(),
                  specs.Array((1,), np.float32, 'reward'),
                  specs.Array((1,), np.float32, 'discount'))

    def add(time_step, episode):
        for spec in data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            if spec.name == 'raw':
                episode['observation'].append(value)
            else:
                episode[spec.name].append(value)

    episode = defaultdict(list)
    time_step = env.reset()
    add(time_step, episode)
    num_ep = 0 if num_ep_start is None else num_ep_start
    global_step = 0
    ep_score = 0
    scores = []
    while num_ep < num_episodes:
        if time_step.last():
            ep = {}
            for spec in data_specs:
                if spec.name == 'raw':
                    value = episode['observation']
                    ep['observation'] = np.array(value, spec.dtype)
                else:
                    value = episode[spec.name]
                    ep[spec.name] = np.array(value, spec.dtype)
            ep['episode_score'] = ep_score
            #save
            store_episode(ep, num_ep, save_dir)
            episode = defaultdict(list)
            time_step = env.reset()
            add(time_step, episode)
            num_ep += 1
            scores.append(ep_score)
            ep_score = 0
            if num_ep % 10 == 0:
                print(f'[{num_ep}/{num_episodes}]')

        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(time_step.observation,
                               1600000,
                               eval_mode=False)
        time_step = env.step(action)
        ep_score += time_step.reward
        add(time_step, episode)
        global_step += 1
    mean, std, rmin, rmax = np.mean(scores), np.std(scores), \
        np.min(scores), np.max(scores)
    print(f'{mean} {std} {rmin} {rmax}')

def get_buffer(buffer_dir, save_dir, threshold, medium_dir, db_type='replay', num_episodes=200):
    def get_time(path):
        fn = path.parts[-1]
        time = fn.split('_')[0].split('T')[-1]
        return int(time)
    eps, num_ep = [], 0
    for p in buffer_dir.glob('*.npz'):
        p_time = get_time(p)
        if db_type == 'replay':
            if p_time <= threshold:
                eps.append(p)
        elif db_type == 'expert':
            if p_time >= threshold:
                eps.append(p)

    if len(eps) > num_episodes:
        #eps = np.random.choice(eps, size=num_episodes)
        eps = eps[-num_episodes:]

    for ep in eps:
        data = np.load(ep)
        ep_score = data['reward'].sum()
        data['episode_score'] = ep_score
        store_episode(data, num_ep, save_dir)
        num_ep += 1

    # If not enough trajs, get more from medium rollouts
    if num_ep < num_episodes:
        n_diff = num_episodes - num_ep
        medium_eps = [x for x in list(medium_dir.glob('*.npz'))[:n_diff]]
        for ep in medium_eps:
            data = np.load(ep)
            store_episode(data, num_ep, save_dir)
            num_ep += 1

def test_agent(env, agent, global_step, num_episodes=200, num_ep_start=None):

    episode = defaultdict(list)
    time_step = env.reset()
    num_ep = 0 if num_ep_start is None else num_ep_start
    ep_score = 0
    scores = []
    while num_ep < num_episodes:
        if time_step.last():
            #save
            time_step = env.reset()
            num_ep += 1
            scores.append(ep_score)
            ep_score = 0
            print(num_ep)

        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(time_step.observation,
                               global_step,
                               eval_mode=False)
        time_step = env.step(action)
        ep_score += time_step.reward
        global_step += 1
    mean, std, rmin, rmax = np.mean(scores), np.std(scores), \
        np.min(scores), np.max(scores)
    print(f'{mean} {std} {rmin} {rmax}')

def main(task, checkpoint, db_type, save_root, n_eps=200):
    save_dir = save_root / db_type
    env = dmc.make(task, 3, 2, 0)
    agent = torch.load(checkpoint)['agent']
    collect(env, agent, save_dir, num_episodes=n_eps)

def main_test(task, checkpoint, db_type, n_eps=10):
    env = dmc.make(task, 3, 2, 0)
    agent = torch.load(checkpoint)['agent']
    global_step = torch.load(checkpoint)['_global_step']
    test_agent(env, agent, global_step, num_episodes=n_eps)

if __name__ == '__main__':
    task = 'cheetah_run'
    root = Path('./exp_local/2022.02.14/235820_task=cheetah_run')
    save_root = Path(f'../datasets/{task}')
    random_agent = root / '20220214T235840_0_checkpoint.pt'
    medium_agent = root / '20220215T003938_504_checkpoint.pt'
    expert_agent = root / 'snapshot.pt'
    checkpoints = [random_agent, medium_agent, expert_agent]
    db_types = ['random', 'medium', 'expert']
    main(task, random_agent, 'random', save_root)
    #main(task, medium_agent, 'medium', save_root)
    #main(task, expert_agent, 'expert', save_root)
    #main_test(task, medium_agent, 'medium')
    #get_buffer()

