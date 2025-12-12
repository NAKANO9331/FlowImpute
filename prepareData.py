import pickle
import numpy as np
import random
import argparse
import configparser
import os
import torch
from common.metrics import mae, rmse, mape
import matplotlib.pyplot as plt

def normalization(train, val, test):
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]
    mean = train.mean(axis=(0, 1, 3), keepdims=True)
    std = train.std(axis=(0, 1, 3), keepdims=True)

    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)

    def normalize(x):
        return np.nan_to_num((x - mean) / std)

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm

def phase_to_onehot(phase, num_class):
    assert num_class > phase.max()
    one_hot = np.zeros((1, num_class))
    one_hot[-1][phase.reshape(-1)] = 1.
    # one_hot = one_hot.reshape(*phase.shape, num_class)
    return one_hot


def onehot_to_phase(phase):
    '''reconstruct one hot phase to direction-level phase'''
    phase_dic = {
        0: ['WS', 'ES'],
        1: ['NS', 'SS'],
        2: ['WL', 'EL'],
        3: ['NL', 'SL'],
        4: ['WS', 'WL'],
        5: ['ES', 'EL'],
        6: ['NS', 'NL'],
        7: ['SS', 'SL']
    }
    # phase:[B,N,8,T]->[B,T,N,8]
    phase = np.transpose(phase, (0, 3, 1, 2))
    batch_size, num_of_timesteps, num_of_vertices, _ = phase.shape
    phase_more = np.full((batch_size, num_of_timesteps,num_of_vertices, 2, 2), ['XX', 'XX'])
    # idx must euqals to B*T*N
    idx = np.argwhere(phase == 1)
    assert len(idx) == batch_size*num_of_timesteps*num_of_vertices
    for x in idx:
        phase_more[x[0], x[1], x[2]] = np.array(phase_dic[x[3]])
    return phase_more


def phases_to_onehot(phase, num_class):
    assert num_class > phase.max()
    one_hot = np.zeros((phase.shape[0], num_class))
    one_hot[range(0, phase.shape[0]), phase.squeeze()] = 1
    # one_hot = one_hot.reshape(*phase.shape, num_class)
    return one_hot


def get_road_adj(filename):
    with open(filename, 'rb') as fo:
        result = pickle.load(fo)
    road_info = result['road_links']
    road_dict_road2id = result['road_dict_road2id']
    num_roads = len(result['road_dict_id2road'])
    adjacency_matrix = np.zeros(
        (int(num_roads), int(num_roads)), dtype=np.float32)

    adj_phase = np.full(
        (int(num_roads), int(num_roads)), 'XX')

    for inter_dic in road_info:
        for link_dic in road_info[inter_dic]:
            source = link_dic[0]
            target = link_dic[1]
            type_p = link_dic[2]
            direction = link_dic[3]

            if type_p == 'go_straight':
                if direction == 0:
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'WS'
                elif direction == 1:
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'SS'
                elif direction == 2:
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'ES'
                else:
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'NS'

            elif type_p == 'turn_left':
                if direction == 0:
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'WL'
                elif direction == 1:
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'SL'
                elif direction == 2:
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'EL'
                else:
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'NL'
            else:
                if direction == 0:
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'WR'
                elif direction == 1:
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'SR'
                elif direction == 2:
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'ER'
                else:
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'NR'

            adjacency_matrix[road_dict_road2id[source]
                             ][road_dict_road2id[target]] = 1

    return adjacency_matrix, adj_phase


def build_road_state(relation_file, state_file, neighbor_node, mask_num, save_dir):

    with open(relation_file, 'rb') as f_re:
        relation = pickle.load(f_re)
    inter_dict_id2inter = relation['inter_dict_id2inter']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']
    num_roads = len(road_dict_road2id)
    net_shape = relation['net_shape']
    neighbor_num = relation['neighbor_num']
    mask_inter = random.sample(neighbor_num[int(neighbor_node)], int(mask_num)) # idx of mask intersections
    # mask_inter = [6,9]
    mask_or = {} # ids of mask intersections
    for i in mask_inter:
        mask_or[i] = inter_dict_id2inter[i]

    adj_road, _ = get_road_adj(relation_file)
    # road_update:0:roads related to virtual inter,1:unmasked,2:masked
    road_update = np.zeros(int(num_roads), dtype=np.int32)

    all_road_feature = []

    for state_dic in state_file:
        with open(state_dic, "rb") as f_ob:
            state = pickle.load(f_ob)
        road_feature = np.zeros((len(state), int(num_roads), 11), dtype=np.float32)

        for id_time, step_dict in enumerate(state):
            for id_node, node_dict in enumerate(step_dict):
                obs = node_dict[0][0]
                phase = phase_to_onehot(node_dict[1], 8)[0]
                direction = []
                if obs.shape[-1] == 12:

                    direction.append(np.concatenate([obs[0:3], phase]))
                    direction.append(np.concatenate([obs[3:6], phase]))
                    direction.append(np.concatenate([obs[6:9], phase]))
                    direction.append(np.concatenate([obs[9:], phase]))

                in_roads = inter_in_roads[inter_dict_id2inter[id_node]]
                for id_road, road in enumerate(in_roads):
                    road_id = road_dict_road2id[road]
                    road_feature[id_time][road_id] = direction[id_road]
                    if id_time == 0:
                        if id_node in mask_inter:
                            road_update[road_id] = 2
                        else:
                            road_update[road_id] = 1
        all_road_feature.append(road_feature)
    road_info = {'road_feature': all_road_feature, 'adj_road': adj_road,
                 'road_update': road_update, 'mask_or': mask_or}
    with open(save_dir, 'wb') as fw:
        pickle.dump(road_info, fw)
    print("save road_node_info done")


def search_data(sequence_length, num_of_depend, label_start_idx, len_input, num_for_predict, points_per_mhalf):

    if points_per_mhalf < 0:
        raise ValueError("points_per_mhalf should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_mhalf * i
        end_idx = start_idx + len_input
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_mhalf, label_start_idx, len_input, num_for_predict, points_per_mhalf):

    mhalf_sample = None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return mhalf_sample, None

    if num_of_mhalf > 0:
        mhalf_indices = search_data(data_sequence.shape[0], num_of_mhalf, label_start_idx, len_input, num_for_predict, points_per_mhalf)
        
        if not mhalf_indices:
            return None, None

        mhalf_sample = np.concatenate([data_sequence[i: j] for i, j in mhalf_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return mhalf_sample, target


def read_and_generate_dataset(graph_signal_matrix_filename, num_of_mhalf, len_input, num_for_predict, points_per_mhalf, save=False):

    with open(graph_signal_matrix_filename, "rb") as f_ob:
        all_data = pickle.load(f_ob)
    data_all = all_data['road_feature']
    node_update = all_data['road_update']
    mask_or = all_data['mask_or']
    adj_road = all_data['adj_road']
    all_samples = []
    for data_seq in data_all:
        for idx in range(data_seq.shape[0]):
            sample = get_sample_indices(data_seq, num_of_mhalf, idx, len_input, num_for_predict, points_per_mhalf)
            if sample[0] is None:
                continue

            mhalf_sample, target = sample
            sample = []
            if num_of_mhalf > 0:
                # mhalf_sample:(T,N,4)->(1,T,N,4)->(1,N,4,T)
                mhalf_sample = np.expand_dims(mhalf_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(mhalf_sample)

            target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            # target = target[:, :, [0, 1, 2]]
            sample.append(target)

            time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
            sample.append(time_sample)

            all_samples.append(sample)  
    random.shuffle(all_samples)
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]  # [(B,N,F,T),(B,N,F,T'),(B,1)]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]  # (B,N,F,T')
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]
    
    # generate data for model
    train_x = mask_op(train_x, node_update, adj_road)
    train_target = mask_op(train_target, node_update, adj_road)
    val_x = mask_op(val_x, node_update, adj_road)
    test_x = mask_op(test_x, node_update, adj_road)

    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        },
        'node_update': node_update,
        'mask_or': mask_or,
        'adj_road':adj_road
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data _mean :', stats['_mean'].shape, stats['_mean'])
    print('train data _std :', stats['_std'].shape, stats['_std'])
    print('node update matrix :', all_data['node_update'].shape)
    print('mask intersection id:', all_data['mask_or'])

    if save:
        filename = graph_signal_matrix_filename.split('.')[0] + '_dataSplit.pkl'
        print('save file:', filename)

        dataset_info = {'train_x': all_data['train']['x'],
                        'train_target': all_data['train']['target'], 
                        'train_timestamp': all_data['train']['timestamp'],

                        'val_x': all_data['val']['x'],
                        'val_target': all_data['val']['target'], 
                        'val_timestamp': all_data['val']['timestamp'],

                        'test_x': all_data['test']['x'],
                        'test_target': all_data['test']['target'], 
                        'test_timestamp': all_data['test']['timestamp'],

                        'mean': all_data['stats']['_mean'], 
                        'std': all_data['stats']['_std'],

                        'node_update': all_data['node_update'],
                        'mask_or': all_data['mask_or']}

        with open(filename, 'wb') as fw:
            pickle.dump(dataset_info, fw)


def generate_actphase(phase, adj_mx, adj_phase):
    '''generate phase_activate matrix according to direction phase'''
    batch_size, num_of_timesteps, num_of_vertices, phase_row, phase_col = phase.shape
    # self.phase_act:record adj matrix of every time(after activation)
    phase_act = np.zeros((batch_size, num_of_timesteps,
                         num_of_vertices, num_of_vertices))
    phase_act = phase_act.reshape(-1, num_of_vertices, num_of_vertices)
    for idx, adj_x in enumerate(adj_mx.flat):
        if adj_x == 1.:
            if idx >= num_of_vertices:
                source = int(idx/num_of_vertices)
                target = idx-source*num_of_vertices
            else:
                source = 0
                target = idx
            phase_node = phase[:, :, source].reshape(-1, phase_row, phase_col)

            for phase_idx, x in enumerate(phase_node):
                if adj_phase[source][target] in x or adj_phase[source][target][1] == 'R':
                    phase_act[phase_idx][source][target] = 1.
    phase_act = phase_act.reshape(batch_size, num_of_timesteps, num_of_vertices, num_of_vertices)
    return phase_act


def generate_actphase_torch(phase, adj_mx, adj_phase):
    batch_size, num_of_timesteps, num_of_vertices, phase_row, phase_col = phase.shape
    # self.phase_act:record adj matrix of every time(after activation)
    phase_act = torch.zeros((batch_size, num_of_timesteps,
                         num_of_vertices, num_of_vertices))
    phase_act = phase_act.reshape(-1, num_of_vertices, num_of_vertices)
    if type(adj_mx) == np.ndarray:
        for idx, adj_x in enumerate(adj_mx.flat):
            if adj_x == 1.:
                if idx >= num_of_vertices:
                    source = int(idx / num_of_vertices)
                    target = idx - source * num_of_vertices
                else:
                    source = 0
                    target = idx
                phase_node = phase[:, :, source].reshape(-1, phase_row, phase_col)

                for phase_idx, x in enumerate(phase_node):
                    if adj_phase[source][target] in x or adj_phase[source][target][1] == 'R':
                        phase_act[phase_idx][source][target] = 1.
    else:
        for idx, adj_x in enumerate(adj_mx.view(-1)):
            if adj_x == 1.:
                if idx >= num_of_vertices:
                    source = int(idx / num_of_vertices)
                    target = idx - source * num_of_vertices
                else:
                    source = 0
                    target = idx
                phase_node = phase[:, :, source].reshape(-1, phase_row, phase_col)

                for phase_idx, x in enumerate(phase_node):
                    if adj_phase[source][target] in x or adj_phase[source][target][1] == 'R':
                        phase_act[phase_idx][source][target] = 1.
    phase_act = phase_act.reshape(batch_size, num_of_timesteps, num_of_vertices, num_of_vertices)
    return phase_act


def revise_unknown(origin_data, predict_data, mask_matrix):

    revise_data = torch.zeros_like(origin_data,dtype=torch.float)
    for node_idx, node in enumerate(mask_matrix):
        if node != 1:
            revise_data[:, node_idx, :, 0] = origin_data[:, node_idx, :, 0]
            revise_data[:, node_idx, :, 1:] = predict_data[:, node_idx, :, :]
        else:
            revise_data[:, node_idx] = origin_data[:, node_idx]
    return revise_data


def mask_op(data_or, mask_matrix, adj_matrix, method='neighbor_mean', **kwargs):
    import random
    import numpy as np
    if method == 'neighbor_mean':
        # Original neighbor mean filling
        for mask_id, value in enumerate(mask_matrix):
            if value != 1:
                neighbors = [col_id for col_id, x in enumerate(adj_matrix[:, mask_id]) if x == 1.]
                neighbor_all = np.zeros_like(data_or[:, 0, :3])
                if len(neighbors) != 0:
                    for node in neighbors:
                        neighbor_all = data_or[:, node, :3] + neighbor_all
                    data_or[:, mask_id, :3] = neighbor_all / len(neighbors)
                else:
                    rand_id = random.randint(0, len(mask_matrix)-1)
                    while mask_matrix[rand_id] != 1:
                        rand_id = random.randint(0, len(mask_matrix)-1)
                    data_or[:,mask_id, :3] = data_or[:,rand_id, :3]
                if value == 0:
                    rand_id = random.sample(neighbors, 1)[0] if neighbors else 0
                    data_or[:, mask_id, 3:] = data_or[:, rand_id, 3:]
        return data_or
    elif method in ['stgcn', 'gnn', 'chebyshev', 'graphsage', 'attention', 'residual', 'adaptive', 'multiscale']:
        from imputation_models import apply_model_imputation
        if data_or.ndim == 4:
            # (B, N, F, T) -> Impute for each batch separately
            filled_batches = []
            for b in range(data_or.shape[0]):
                arr = data_or[b]  # (N, F, T)
                data_for_fill = np.transpose(arr, (2, 0, 1))  # (T, N, F)
                print(f'[DEBUG] Shape passed to apply_model_imputation: {data_for_fill.shape}')
                filled = apply_model_imputation(data_for_fill, mask_matrix, adj_matrix, model_type=method, dataset_name=kwargs.get('dataset_name', 'hz_4x4'), train_data=data_for_fill)
                filled = np.transpose(filled, (1, 2, 0))  # (N, F, T)
                filled_batches.append(filled)
            data_or = np.stack(filled_batches, axis=0)
        elif data_or.ndim == 3:
            arr = data_or
            data_for_fill = np.transpose(arr, (2, 0, 1))
            print(f'[DEBUG] Shape passed to apply_model_imputation: {data_for_fill.shape}')
            filled = apply_model_imputation(data_for_fill, mask_matrix, adj_matrix, model_type=method, dataset_name=kwargs.get('dataset_name', 'hz_4x4'), train_data=data_for_fill)
            filled = np.transpose(filled, (1, 2, 0))
            data_or = filled
        else:
            raise ValueError('data_or shape does not support automatic conversion, please handle manually')
        return data_or
    elif method == 'zero_fill':
        # Zero fill for each masked node
        for mask_id, value in enumerate(mask_matrix):
            if value == 0:
                if data_or.ndim == 4:
                    data_or[:, mask_id, :, :] = 0
                elif data_or.ndim == 3:
                    data_or[mask_id, :, :] = 0
        return data_or
    else:
        raise ValueError(f'Unknown imputation method: {method}')

def plot_imputation_compare(original, masked, imputed, method, save_dir, node_indices=None, time_indices=None):
    """
    Plot imputation comparison.
    
    Args:
        original, masked, imputed: (B, N, F, T) or (N, F, T)
    """
    os.makedirs(save_dir, exist_ok=True)
    # Ensure shape is (N, F, T)
    if original.ndim == 4:
        original = original[0]
    if masked.ndim == 4:
        masked = masked[0]
    if imputed.ndim == 4:
        imputed = imputed[0]
    N, F, T = original.shape
    if node_indices is None:
        node_indices = random.sample(range(N), min(3, N))
    if time_indices is None:
        time_indices = random.sample(range(T), min(100, T))
        time_indices.sort()
    for node in node_indices:
        plt.figure(figsize=(12, 5))
        for f in range(min(1, F)):
            plt.plot(time_indices, original[node, f, time_indices], label='Original', color='black', linewidth=2)
            plt.plot(time_indices, masked[node, f, time_indices], label='Masked', color='red', linestyle='dashed')
            plt.plot(time_indices, imputed[node, f, time_indices], label='Imputed', color='blue', linestyle='dotted')
        plt.title(f'Node {node} - {method} Imputation Effect')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'compare_node{node}_{method}.png'))
        plt.close()


def batch_impute_and_visualize(data_dict, mask_matrix, adj_matrix, dataset_name, save_dir, vis_dir):

    from imputation_models import batch_imputation_with_all_models
    # Traditional methods
    methods = ['zero_fill', 'neighbor_mean']
    for method in methods:
        print(f"[Batch Imputation] {method}")
        masked = data_dict['masked'].copy()
        imputed = mask_op(masked.copy(), mask_matrix, adj_matrix, method=method, dataset_name=dataset_name)
        # Save
        save_path = os.path.join(save_dir, f'{dataset_name}_dataSplit_{method}.pkl')
        with open(save_path, 'wb') as fw:
            pickle.dump(imputed, fw)
        print(f"[Saved] {save_path}")
        # Visualization
        plot_imputation_compare(data_dict['original'], data_dict['masked'], imputed, method, os.path.join(vis_dir, method))
    # GNN-based methods
    gnn_results = batch_imputation_with_all_models(data_dict['masked'], mask_matrix, adj_matrix, dataset_name=dataset_name, train_data=data_dict['original'], save_dir=save_dir)
    for method, imputed in gnn_results.items():
        plot_imputation_compare(data_dict['original'], data_dict['masked'], imputed, method, os.path.join(vis_dir, method))


def batch_impute_and_visualize_all(data_dict, mask_matrix, adj_matrix, dataset_name, save_dir, vis_dir):

    from imputation_models import batch_imputation_with_all_models, apply_model_imputation
    methods = ['zero_fill', 'neighbor_mean']
    gnn_methods = ['stgcn', 'gnn', 'chebyshev', 'graphsage', 'attention', 'residual', 'adaptive', 'multiscale']
    for method in methods:
        print(f"[Batch Imputation] {method}")
        imputed_train_x = mask_op(data_dict['train_x'].copy(), mask_matrix, adj_matrix, method=method, dataset_name=dataset_name)
        imputed_val_x = mask_op(data_dict['val_x'].copy(), mask_matrix, adj_matrix, method=method, dataset_name=dataset_name)
        imputed_test_x = mask_op(data_dict['test_x'].copy(), mask_matrix, adj_matrix, method=method, dataset_name=dataset_name)
        dataset_info = {
            'train_x': imputed_train_x,
            'train_target': data_dict['train_target'],
            'train_timestamp': data_dict['train_timestamp'],
            'val_x': imputed_val_x,
            'val_target': data_dict['val_target'],
            'val_timestamp': data_dict['val_timestamp'],
            'test_x': imputed_test_x,
            'test_target': data_dict['test_target'],
            'test_timestamp': data_dict['test_timestamp'],
            'mean': data_dict['mean'],
            'std': data_dict['std'],
            'node_update': data_dict['node_update'],
            'mask_or': data_dict['mask_or']
        }
        save_path = os.path.join(save_dir, f'{dataset_name}_dataSplit_{method}.pkl')
        with open(save_path, 'wb') as fw:
            pickle.dump(dataset_info, fw)
        print(f"[Saved] {save_path}")
        # Visualize train_x
        plot_imputation_compare(data_dict['train_x'], data_dict['train_x'], imputed_train_x, method, os.path.join(vis_dir, method))
    # Process GNN-based methods batch by batch
    for method in gnn_methods:
        print(f"[Batch Imputation] Imputing with {method}...")
        # train_x
        filled_batches = []
        for b in range(data_dict['train_x'].shape[0]):
            print(f'[DEBUG] {method} train_x batch {b+1}/{data_dict["train_x"].shape[0]}')
            arr = data_dict['train_x'][b]  # (N, F, T)
            data_for_fill = np.transpose(arr, (2, 0, 1))  # (T, N, F)
            print(f'[DEBUG] Shape passed to apply_model_imputation: {data_for_fill.shape}')
            filled = apply_model_imputation(data_for_fill, mask_matrix, adj_matrix, model_type=method, dataset_name=dataset_name, train_data=data_for_fill)
            filled = np.transpose(filled, (1, 2, 0))  # (N, F, T)
            filled_batches.append(filled)
        imputed_train_x = np.stack(filled_batches, axis=0)
        # val_x
        filled_batches = []
        for b in range(data_dict['val_x'].shape[0]):
            print(f'[DEBUG] {method} val_x batch {b+1}/{data_dict["val_x"].shape[0]}')
            arr = data_dict['val_x'][b]
            data_for_fill = np.transpose(arr, (2, 0, 1))
            filled = apply_model_imputation(data_for_fill, mask_matrix, adj_matrix, model_type=method, dataset_name=dataset_name, train_data=data_for_fill)
            filled = np.transpose(filled, (1, 2, 0))
            filled_batches.append(filled)
        imputed_val_x = np.stack(filled_batches, axis=0)
        # test_x
        filled_batches = []
        for b in range(data_dict['test_x'].shape[0]):
            print(f'[DEBUG] {method} test_x batch {b+1}/{data_dict["test_x"].shape[0]}')
            arr = data_dict['test_x'][b]
            data_for_fill = np.transpose(arr, (2, 0, 1))
            filled = apply_model_imputation(data_for_fill, mask_matrix, adj_matrix, model_type=method, dataset_name=dataset_name, train_data=data_for_fill)
            filled = np.transpose(filled, (1, 2, 0))
            filled_batches.append(filled)
        imputed_test_x = np.stack(filled_batches, axis=0)
        dataset_info = {
            'train_x': imputed_train_x,
            'train_target': data_dict['train_target'],
            'train_timestamp': data_dict['train_timestamp'],
            'val_x': imputed_val_x,
            'val_target': data_dict['val_target'],
            'val_timestamp': data_dict['val_timestamp'],
            'test_x': imputed_test_x,
            'test_target': data_dict['test_target'],
            'test_timestamp': data_dict['test_timestamp'],
            'mean': data_dict['mean'],
            'std': data_dict['std'],
            'node_update': data_dict['node_update'],
            'mask_or': data_dict['mask_or']
        }
        save_path = os.path.join(save_dir, f'{dataset_name}_dataSplit_{method}.pkl')
        with open(save_path, 'wb') as fw:
            pickle.dump(dataset_info, fw)
        print(f"[Saved] {save_path}")
        plot_imputation_compare(data_dict['train_x'], data_dict['train_x'], imputed_train_x, method, os.path.join(vis_dir, method))


def get_key(d, *candidates):
    for k in candidates:
        if k in d:
            return d[k]
    print('Available keys:', list(d.keys()))
    raise KeyError(f"None of {candidates} found in dict.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/set.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config))
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']

    num_of_vertices = int(data_config['num_of_vertices'])
    len_input = int(data_config['len_input'])
    points_per_mhalf = int(data_config['len_input'])
    num_for_predict = int(data_config['num_for_predict'])
    
    dataset_name = data_config['dataset_name']
    base_model = data_config['base_model']

    num_of_mhalf = int(training_config['num_of_mhalf'])
    neighbor_node = data_config['neighbor_node']
    mask_num = data_config['mask_num']

    data_basedir = os.path.join('data',str(dataset_name))
    state_basedir = os.path.join(data_basedir,'state_data')
    relation_filename = os.path.join(data_basedir,'roadnet_relation.pkl')
    graph_signal_matrix_filename = 's'+str(points_per_mhalf)+'_p'+str(num_for_predict)+'_n'+str(neighbor_node)+'_m'+str(mask_num)+'.pkl'
    graph_signal_matrix_filename = os.path.join(state_basedir,graph_signal_matrix_filename)

    if not os.path.exists(data_basedir):
        os.makedirs(data_basedir)
    if not os.path.exists(state_basedir):
        os.makedirs(state_basedir)

    # read state of intersections,convert it into state which road graph needed,save.

    if dataset_name == 'hz_4x4':
        # rawstate.pkl: generated using origin taffic flows
        # rawstate_d.pkl: generated using double taffic flows
        state_file = ['rawstate.pkl','rawstate_d.pkl']
    else:
        state_file = ['rawstate.pkl']

    state_file_list = [os.path.join(data_basedir, s_dic) for s_dic in state_file]
    
    build_road_state(relation_filename, state_file_list, neighbor_node, mask_num, save_dir=graph_signal_matrix_filename)

    # according to file of task above, generate train set,val set and test set.
    read_and_generate_dataset(graph_signal_matrix_filename, num_of_mhalf, len_input, num_for_predict, points_per_mhalf=points_per_mhalf, save=True)

    # Read the split standard format data, ensure reading *_dataSplit.pkl file
    split_file = graph_signal_matrix_filename.replace('.pkl', '_dataSplit.pkl')
    with open(split_file, 'rb') as f:
        raw_data = pickle.load(f)
    # Read adj_road
    with open(graph_signal_matrix_filename, 'rb') as f:
        adj_data = pickle.load(f)
    adj_matrix = adj_data['adj_road']
    data_dict = {
        'train_x': get_key(raw_data, 'train_x', 'train x'),
        'train_target': get_key(raw_data, 'train_target', 'train target'),
        'train_timestamp': get_key(raw_data, 'train_timestamp', 'train timestamp'),
        'val_x': get_key(raw_data, 'val_x', 'val x'),
        'val_target': get_key(raw_data, 'val_target', 'val target'),
        'val_timestamp': get_key(raw_data, 'val_timestamp', 'val timestamp'),
        'test_x': get_key(raw_data, 'test_x', 'test x'),
        'test_target': get_key(raw_data, 'test_target', 'test target'),
        'test_timestamp': get_key(raw_data, 'test_timestamp', 'test timestamp'),
        'mean': get_key(raw_data, 'mean', 'stats._mean'),
        'std': get_key(raw_data, 'std', 'stats._std'),
        'node_update': get_key(raw_data, 'node_update', 'node update matrix'),
        'mask_or': get_key(raw_data, 'mask_or', 'mask intersection id')
    }
    mask_matrix = data_dict['node_update']
    save_dir = os.path.dirname(split_file)
    vis_dir = os.path.join('results', 'visualization')
    batch_impute_and_visualize_all(data_dict, mask_matrix, adj_matrix, dataset_name, save_dir, vis_dir)
    print('All imputation methods have been completed in batch, data and images have been saved.')
