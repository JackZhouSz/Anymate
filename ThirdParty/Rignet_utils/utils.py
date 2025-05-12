#-------------------------------------------------------------------------------
# Name:        utils.py
# Purpose:     utilize functions for skeleton generation
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import numpy as np
from .rig_parser import Info, TreeNode
from .mst_utils import increase_cost_for_outside_bone, loadSkel_recur,primMST_normal, increase_cost_for_outside_bone_tensor
import trimesh
import torch

def get_skel(pred_joints, prob_matrix,vox):
    "use predict connection which indicte the connection prob between joints to find the root joints,whihc is the joint with the highest connection prob with itself"
    root_id = np.argmax(np.diag(prob_matrix))
    # set the digonal to 0 and normalize the prob_matrix
    np.fill_diagonal(prob_matrix, 0)
    prob_matrix = prob_matrix / (np.sum(prob_matrix, axis=1, keepdims=True)+1e-6)
            
    cost_matrix = -np.log(prob_matrix + 1e-10)
    if torch.is_tensor(vox):
        cost_matrix = increase_cost_for_outside_bone_tensor(cost_matrix, pred_joints, vox)
    else:
        cost_matrix = increase_cost_for_outside_bone(cost_matrix, pred_joints, vox)
    
    pred_joints = np.array(pred_joints)

    # Create a matrix of shape (n, n, 3) where each element is the difference pred_joints[j] - pred_joints[i]
    diff_matrix = pred_joints[:, np.newaxis, :] - pred_joints[np.newaxis, :, :]
    norms = np.linalg.norm(diff_matrix, axis=2, keepdims=True)
    norms[norms == 0] = 1
    normal_matrix = diff_matrix / norms
    np.fill_diagonal(normal_matrix[:, :, 0], 0)
    np.fill_diagonal(normal_matrix[:, :, 1], 0)
    np.fill_diagonal(normal_matrix[:, :, 2], 0)
    
    pred_skel = Info()
    
    parent, key, root_id = primMST_normal(cost_matrix, root_id, normal_matrix)
    
    for i in range(len(parent)):
        if parent[i] == -1:
            pred_skel.root = TreeNode('root', tuple(pred_joints[i]))
            break
    loadSkel_recur(pred_skel.root, i, None, pred_joints, parent)
    pred_skel.joint_pos = pred_skel.get_joint_dict()
    #create mtrx n*n*3 matrix for normal vector between two joints

    return pred_skel, parent




