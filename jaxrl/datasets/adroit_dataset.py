import gym
import numpy as np
import os

from jaxrl.datasets.dataset import Batch, Dataset
AWAC_DATA_DIR='../demonstrations/offpolicy_hand_data'

def process_expert_dataset(expert_datset):
    """This is a mess, but works
    """
    all_observations = []
    all_next_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []

    for x in expert_datset:
        all_observations.append(
            np.vstack([xx['state_observation'] for xx in x['observations']]))
        all_next_observations.append(
            np.vstack(
                [xx['state_observation'] for xx in x['next_observations']]))
        all_actions.append(np.vstack([xx for xx in x['actions']]))
        # for some reason rewards has an extra entry, so in rlkit they just remove the last entry: https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/demos/source/dict_to_mdp_path_loader.py#L84
        all_rewards.append(x['rewards'][:-1])
        all_terminals.append(x['terminals'])

    return {
        'observations':
        np.concatenate(all_observations, dtype=np.float32),
        'next_observations':
        np.concatenate(all_next_observations, dtype=np.float32),
        'actions':
        np.concatenate(all_actions, dtype=np.float32),
        'rewards':
        np.concatenate(all_rewards, dtype=np.float32),
        'terminals':
        np.concatenate(all_terminals, dtype=np.float32)
    }
def process_bc_dataset(bc_dataset):
    final_bc_dataset = {k: [] for k in bc_dataset[0] if 'info' not in k}

    for x in bc_dataset:
        for k in final_bc_dataset:
            final_bc_dataset[k].append(x[k])

    return {
        k: np.concatenate(v, dtype=np.float32).squeeze()
        for k, v in final_bc_dataset.items()
    }
class AdroitBinaryDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 remove_terminals=True,
                 include_bc_data=True):
        env_prefix = env.spec.name.split('-')[0]
        expert_dataset = np.load(os.path.join(AWAC_DATA_DIR,f'{env_prefix}2_sparse.npy'),allow_pickle=True)
        dataset_dict = process_expert_dataset(expert_dataset)
        if include_bc_data:
            bc_dataset = np.load(os.path.join(AWAC_DATA_DIR, f'{env_prefix}_bc_sparse4.npy'), allow_pickle=True)
            bc_dataset = process_bc_dataset(bc_dataset)

            dataset_dict = {
                k: np.concatenate([dataset_dict[k], bc_dataset[k]])
                for k in dataset_dict
            }
        if clip_to_eps:
            lim = 1 - eps
            dataset_dict['actions'] = np.clip(dataset_dict['actions'], -lim, lim)
        dones = np.full_like(dataset_dict['rewards'], False, dtype=bool)
        for i in range(len(dones) - 1):
            if np.linalg.norm(dataset_dict['observations'][i + 1] -
                              dataset_dict['next_observations'][i]
                              ) > 1e-6 or dataset_dict['terminals'][i] == 1.0:
                dones[i] = True
        # import pdb; pdb.set_trace()
        if remove_terminals:
            dataset_dict['terminals'] = np.zeros_like(dataset_dict['terminals'])

        dones[-1] = True
        dataset_dict['masks'] = 1.0 - dataset_dict['terminals']
        del dataset_dict['terminals']

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict['dones'] = dones

        super().__init__(dataset_dict['observations'].astype(np.float32),
                         actions=dataset_dict['actions'].astype(np.float32),
                         rewards=dataset_dict['rewards'].astype(np.float32),
                         masks=dataset_dict['masks'].astype(np.float32),
                         dones_float=dones.astype(np.float32),
                         next_observations=dataset_dict['next_observations'].astype(np.float32),
                         size=len(dataset_dict['observations']))

class AdroitBinaryTruncDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 include_bc_data=True):
                 
        dataset_dict = get_hand_dataset_truncate_pos(
            env_prefix=env.spec.name.split('-')[0],
            add_bc_demos=include_bc_data,
            clip_actions=clip_to_eps, 
            eps=eps)

        super().__init__(dataset_dict['observations'].astype(np.float32),
                         actions=dataset_dict['actions'].astype(np.float32),
                         rewards=dataset_dict['rewards'].astype(np.float32),
                         masks=dataset_dict['masks'].astype(np.float32),
                         dones_float=dataset_dict['dones'].astype(np.float32),
                         next_observations=dataset_dict['next_observations'].astype(np.float32),
                         size=len(dataset_dict['observations']))

def get_hand_dataset_truncate_pos(env_prefix, add_bc_demos=True, reward_scale=1.0, reward_bias=0.0, pos_ind=-1, keep_neg=False, clip_actions=True, eps=1e-5):
    expert_demo_paths = os.path.join(AWAC_DATA_DIR,f'{env_prefix}2_sparse.npy')
    bc_demo_paths = os.path.join(AWAC_DATA_DIR, f'{env_prefix}_bc_sparse4.npy')

    dataset_list=[]
    dataset_bc_list=[]
    action_lim=1-eps
    print("loading expert demos from:", expert_demo_paths)
    dataset = np.load(expert_demo_paths, allow_pickle=True)

    for i in range(len(dataset)):
        N = len(dataset[i]["observations"])
        for j in range(len(dataset[i]["observations"])):
            dataset[i]["observations"][j] = dataset[i]["observations"][j]['state_observation']
            dataset[i]["next_observations"][j] = dataset[i]["next_observations"][j]['state_observation']

        if np.array(dataset[i]["rewards"]).shape != np.array(dataset[i]["terminals"]).shape:
            print(np.array(dataset[i]["rewards"]).shape, np.array(dataset[i]["terminals"]).shape)
            dataset[i]["rewards"] = dataset[i]["rewards"][:N]

        if clip_actions:
            dataset[i]["actions"] = np.clip(dataset[i]["actions"], -action_lim, action_lim)

        assert np.array(dataset[i]["rewards"]).shape == np.array(dataset[i]["terminals"]).shape
        dataset[i].pop('terminals', None)

        if not (0 in dataset[i]["rewards"]):
            if keep_neg:
                d_neg = truncate_with_specific_ind(dataset, i, reward_scale, reward_bias, start_index=None, end_index=None)
                dataset_list.append(d_neg)
            continue

        trunc_ind = np.where(dataset[i]["rewards"] == 0)[0][pos_ind] + 1
        d_pos = truncate_with_specific_ind(dataset, i, reward_scale, reward_bias, start_index=None, end_index=trunc_ind)
        dataset_list.append(d_pos)
        if keep_neg:
            d_neg = truncate_with_specific_ind(dataset, i, reward_scale, reward_bias, start_index=trunc_ind, end_index=None)
            dataset_list.append(d_neg)


    if add_bc_demos:
        print("loading BC demos from:", bc_demo_paths)
        dataset_bc = np.load(bc_demo_paths, allow_pickle=True)
        for i in range(len(dataset_bc)):
            dataset_bc[i]["rewards"] = dataset_bc[i]["rewards"].squeeze()
            dataset_bc[i]["dones"] = dataset_bc[i]["terminals"].squeeze()
            dataset_bc[i].pop('terminals', None)
            if clip_actions:
                dataset_bc[i]["actions"] = np.clip(dataset_bc[i]["actions"], -action_lim, action_lim)

            if not (0 in dataset_bc[i]["rewards"]):
                if keep_neg:
                    d_neg = truncate_with_specific_ind(dataset_bc, i, reward_scale, reward_bias, start_index=None, end_index=None)
                    dataset_bc_list.append(d_neg)
                continue

            trunc_ind = np.where(dataset_bc[i]["rewards"] == 0)[0][pos_ind] + 1
            d_pos = truncate_with_specific_ind(dataset_bc, i, reward_scale, reward_bias, start_index=None, end_index=trunc_ind)
            dataset_bc_list.append(d_pos)
            if keep_neg:
                d_neg = truncate_with_specific_ind(dataset_bc, i, reward_scale, reward_bias, start_index=trunc_ind, end_index=None)
                dataset_bc_list.append(d_neg)

    dataset = np.concatenate([dataset_list, dataset_bc_list])
    
    print("num offline trajs:", len(dataset))
    concatenated = {}
    for key in dataset[0].keys():
        if key in ['agent_infos', 'env_infos']:
            continue
        concatenated[key] = np.concatenate([batch[key] for batch in dataset], axis=0).astype(np.float32)
    return concatenated


def truncate_with_specific_ind(dataset, i, reward_scale, reward_bias, start_index=None, end_index=None):
    
    observations = np.array(dataset[i]["observations"])[start_index:end_index]
    next_observations = np.array(dataset[i]["next_observations"])[start_index:end_index]
    rewards = dataset[i]["rewards"][start_index:end_index]
    dones = (rewards == 0)
    rewards = rewards * reward_scale + reward_bias
    actions = np.array(dataset[i]["actions"])[start_index:end_index]
    masks = 1.0 - dones

    return dict(
                observations=observations,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
                masks=masks
            )