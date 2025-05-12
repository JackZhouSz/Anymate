import torch
from torch.utils.data import Dataset
import os
import numpy as np
from Anymate.utils.dataset_utils import create_mask, index_to_sparse, index_to_sparse_con

def my_collate(batch):
    # print(len(batch))
    data = {}
    for key in batch[0]:
        if key=='vox' or key=='name' or key=='joints_num' or key=='skins_index' or key=='skins_weight' or key=='parent_index' or key=='conns' or key=='joints' or key=='bones' or key=='mesh_skins_index' or key=='mesh_skins_weight' or key=='mesh_pc' or key=='mesh_face':
            data[key] = [sample[key] for sample in batch]
        elif key=='pc':
            data['points_cloud'] = torch.stack([sample['pc'] for sample in batch])
        elif key=='skins':
            continue
        elif key=='bones_num':
            data[key] = torch.tensor([sample['bones_num'] for sample in batch])
        else:
            data[key] = torch.stack([sample[key] for sample in batch])
     
    if 'skins_index' in batch[0]:
        max_joints = max(data['joints_num'])
        max_bones = max(data['bones_num'])
        # max_joints = 64
        skin_list = [index_to_sparse(data['skins_index'][i].unsqueeze(0), data['skins_weight'][i].unsqueeze(0), [1, 8192, max_bones])[0] for i in range(len(data['skins_index']))]
        data['skins'] = torch.stack(skin_list,dim=0)
        data['joints_mask'] = torch.stack([create_mask(sample['joints_num'],max_len=max_joints) for sample in batch])
        data['bones_mask'] = torch.stack([create_mask(sample['bones_num'],max_len=max_bones) for sample in batch])
        
    if 'conns' in batch[0]:
        max_joints = max(data['joints_num'])
        conn_matrix = torch.zeros(len(data['conns']), 96, max_joints)
        for i in range(len(data['conns'])):
            for j in range(data['joints_num'][i]):
                conn_matrix[i, j, data['conns'][i][j].long()] = 1
        data['conns'] = conn_matrix
    if 'joints' in batch[0]:
        padded_joints_matrix = torch.ones(len(data['name']), 96, 3) * (-3)
        for i in range(len(data['name'])):
            padded_joints_matrix[i, :data['joints_num'][i], :] = data['joints'][i]
        data['joints'] = padded_joints_matrix
    if 'bones' in batch[0]:
        padded_bones_matrix = torch.ones(len(data['name']), 64, 6) * (-3)
        for i in range(len(data['name'])):
            padded_bones_matrix[i, :data['bones_num'][i], :] = data['bones'][i]
        data['bones'] = padded_bones_matrix
    return data

class AnymateDataset(Dataset):
    def __init__(self, name='Anymate_test', root='Anymate/data'):   

        if os.path.exists(os.path.join(root, name) + '.pt'):
            self.data_list = torch.load(os.path.join(root, name) + '.pt')
        else:
            raise ValueError('Dataset not found at path: {}'.format(os.path.join(root, name) + '.pt'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]