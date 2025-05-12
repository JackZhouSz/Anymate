from tqdm import tqdm
from torch.utils.data import DataLoader
from Anymate.utils.utils import load_checkpoint, get_joint, get_connectivity, get_skinning
from Anymate.utils.eval_utils import evaluate_joint, evaluate_connectivity, evaluate_skinning, evaluate_skeleton
from Anymate.args import ui_args, anymate_args
from Anymate.dataset import AnymateDataset, index_to_sparse
import torch
import os

checkpoint_joint = ui_args.checkpoint_joint
checkpoint_conn = ui_args.checkpoint_conn
checkpoint_skin = ui_args.checkpoint_skin

# save_dir = 'Anymate/results'

test_dataset = AnymateDataset(name='Anymate_test', root='Anymate/data')
if checkpoint_joint is not None:
    model_joint = load_checkpoint(checkpoint_joint, anymate_args.device, anymate_args.num_joints)
    joints = []
    joints_gt = []
    for data in tqdm(test_dataset):
        joints_pred = get_joint(data['pc'], model_joint, device=anymate_args.device, vox=data['vox'])
        if joints_pred.shape[0]<2:
            continue
        joints.append(joints_pred)
        joints_gt.append(data['joints'][:data['joints_num'], :3])
    evaluate_joint(joints, joints_gt)

if checkpoint_conn is not None:
    model_conn = load_checkpoint(checkpoint_conn, anymate_args.device, anymate_args.num_joints)
    conns = []
    conns_gt = []
    joints_gt = []
    vox_list = []
    for data in tqdm(test_dataset):
        conns.append(get_connectivity(data['pc'], data['joints'][:data['joints_num']], model_conn, device=anymate_args.device, return_prob=True))
        joints_gt.append(data['joints'][:data['joints_num'], :3])
        vox_list.append(data['vox'])
        conn_matrix = torch.zeros(data['joints_num'], data['joints_num'])
        for i in range(data['joints_num']):
            conn_matrix[i, int(data['conns'][i])] = 1
        conns_gt.append(conn_matrix)
    evaluate_connectivity(conns, conns_gt, joints_gt, vox_list)

if checkpoint_skin is not None:
    model_skin = load_checkpoint(checkpoint_skin, anymate_args.device, anymate_args.num_joints)
    skins = []
    skins_gt = []
    for data in tqdm(test_dataset):
        skins.append(get_skinning(data['pc'], None, None, model_skin, bones=data['bones'][:data['bones_num']], device=anymate_args.device))
        skins_gt.append(index_to_sparse(data['skins_index'].unsqueeze(0), data['skins_weight'].unsqueeze(0), [1, 8192, data['bones_num']])[0])
    evaluate_skinning(skins, skins_gt)

if checkpoint_joint is not None and checkpoint_conn is not None:
    if model_conn is None:
        model_conn = load_checkpoint(checkpoint_conn, anymate_args.device, anymate_args.num_joints)
    if model_joint is None:
        model_joint = load_checkpoint(checkpoint_joint, anymate_args.device, anymate_args.num_joints) 
    conns = []
    conns_gt = []
    joints_gt = []
    joints = []
    vox_list = []
    for data in tqdm(test_dataset):
        joints.append(get_joint(data['pc'], model_joint, device=anymate_args.device))
        joints_gt.append(data['joints'][:data['joints_num'], :3])
        conns.append(get_connectivity(data['pc'], joints[-1], model_conn, device=anymate_args.device, return_prob=True))
        conn_matrix = torch.zeros(data['joints_num'], data['joints_num'])
        for i in range(data['joints_num']):
            conn_matrix[i, int(data['conns'][i])] = 1
        conns_gt.append(conn_matrix)
        vox_list.append(data['vox'])
    evaluate_skeleton(joints,joints_gt,conns,conns_gt,vox_list,fs_threshold=0.1)




