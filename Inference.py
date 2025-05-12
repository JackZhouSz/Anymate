from tqdm import tqdm
from torch.utils.data import DataLoader
from Anymate.utils.utils import load_checkpoint, get_joint, get_connectivity, get_skinning
from Anymate.utils.eval_utils import evaluate_joint, evaluate_connectivity, evaluate_skinning, evaluate_skeleton
from Anymate.args import ui_args, anymate_args
from Anymate.dataset import AnymateDataset, index_to_sparse
import torch
import os
import trimesh

checkpoint_joint = ui_args.checkpoint_joint
checkpoint_conn = ui_args.checkpoint_conn
checkpoint_skin = ui_args.checkpoint_skin

save_dir = 'Anymate/results'

test_dataset = AnymateDataset()

model_joint = load_checkpoint(checkpoint_joint, anymate_args.device, anymate_args.num_joints)
model_conn = load_checkpoint(checkpoint_conn, anymate_args.device, anymate_args.num_joints)
model_skin = load_checkpoint(checkpoint_skin, anymate_args.device, anymate_args.num_joints)

for data in tqdm(test_dataset):
    joints = get_joint(data['pc'], model_joint, device=anymate_args.device, vox=data['vox'])
    conns = get_connectivity(data['pc'], joints, model_conn, device=anymate_args.device)
    skins = get_skinning(data['pc'], joints, conns, model_skin, vertices=data['mesh_pc'], device=anymate_args.device)
    print(joints.shape, conns.shape, skins.shape)
    save_path = os.path.join(save_dir, data['name'])
    os.makedirs(save_path, exist_ok=True)
    torch.save(joints, os.path.join(save_path, 'joints.pt'))
    torch.save(conns, os.path.join(save_path, 'conns.pt'))
    torch.save(skins, os.path.join(save_path, 'skins.pt'))
    vertices = data['mesh_pc']
    faces = data['mesh_face']
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.export(os.path.join(save_path, 'object.obj'))
    break
