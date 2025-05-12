#-------------------------------------------------------------------------------
# Name:        mst_utils.py
# Purpose:     utilize functions for skeleton generation
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import sys
import numpy as np
from .rig_parser import TreeNode
from .rig_parser import Skel
import torch

def inside_check(pts, vox):
    """
    Check where points are inside or outside the mesh based on its voxelization.
    :param pts: points to be checked
    :param vox: voxelized mesh
    :return: internal points, and index of them in the input array.
    """
    vc = (pts - vox.translate) / vox.scale * vox.dims[0]
    vc = np.round(vc).astype(int)
    ind1 = np.logical_and(np.all(vc >= 0, axis=1), np.all(vc < 88, axis=1))
    vc = np.clip(vc, 0, 87)
    ind2 = vox.data[vc[:, 0], vc[:, 1], vc[:, 2]]
    ind = np.logical_and(ind1, ind2)
    pts = pts[ind]
    return pts, np.argwhere(ind).squeeze()


def sample_on_bone(p_pos, ch_pos):
    """
    sample points on a bone
    :param p_pos: parent joint position
    :param ch_pos: child joint position
    :return: a array of samples on this bone.
    """
    ray = ch_pos - p_pos
    bone_length = np.sqrt(np.sum((p_pos - ch_pos) ** 2))
    num_step = np.round(bone_length / 0.01)
    i_step = np.arange(1, num_step + 1)
    unit_step = ray / (num_step + 1e-30)
    unit_step = np.repeat(unit_step[np.newaxis, :], num_step, axis=0)
    res = p_pos + unit_step * i_step[:, np.newaxis]
    return res


def minKey(key, mstSet, nV):
    # Initilaize min value
    min = sys.maxsize
    for v in range(nV):
        if key[v] < min and mstSet[v] == False:
            min = key[v]
            min_index = v
    return min_index 
    
def primMST_normal(graph, init_id, normal_matrix):
    """
    Modified Prim's algorithm to generate a minimum spanning tree (MST).
    :param graph: pairwise cost matrix
    :param init_id: init node ID as root
    :return: parent array, key array, init_id
    """
    nV = graph.shape[0]
    key = [sys.maxsize] * nV
    parent = [None] * nV
    mstSet = [False] * nV
    key[init_id] = 0
    parent[init_id] = -1
    previous_normal = np.zeros((nV, 3))

    while not all(mstSet):
        u = minKey(key, mstSet, nV)
        mstSet[u] = True
        if parent[u] >= 0:
            previous_normal[u] = normal_matrix[u, parent[u]]
            updated_normal = np.dot(previous_normal[u], normal_matrix[u, :].T) #1*n
            updated_normal[updated_normal<0]=0
        # print('updated_normal',updated_normal.shape)
            graph[u, :] = graph[u, :] +(1e8*updated_normal**2+1)
            graph[:, u] = graph[:, u] +(1e8*updated_normal**2+1)

        for v in range(nV):
            
            if graph[u, v] > 0 and mstSet[v] is False and key[v] > graph[u, v]:
                key[v] = graph[u, v]
                parent[v] = u


    return parent, key, init_id   


def loadSkel_recur(p_node, parent_id, joint_name, joint_pos, parent):
    """
    Converst prim algorithm result to our skel/info format recursively
    :param p_node: Root node
    :param parent_id: parent name of current step of recursion.
    :param joint_name: list of joint names
    :param joint_pos: joint positions
    :param parent: parent index returned by prim alg.
    :return: p_node (root) will be expanded to linked with all joints
    """
    for i in range(len(parent)):
        if parent[i] == parent_id:
            if joint_name is not None:
                ch_node = TreeNode(joint_name[i], tuple(joint_pos[i]))
            else:
                ch_node = TreeNode('joint_{}'.format(i), tuple(joint_pos[i]))
            p_node.children.append(ch_node)
            ch_node.parent = p_node
            loadSkel_recur(ch_node, i, joint_name, joint_pos, parent)


def unique_rows(a):
    """
    remove repeat rows from a numpy array
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def increase_cost_for_outside_bone(cost_matrix, joint_pos, vox):
    """
    increase connectivity cost for bones outside the meshs
    """
    for i in range(len(joint_pos)):
        for j in range(i+1, len(joint_pos)):
            bone_samples = sample_on_bone(joint_pos[i], joint_pos[j])
            bone_samples_vox = (bone_samples - vox.translate) / vox.scale * vox.dims[0]
            bone_samples_vox = np.round(bone_samples_vox).astype(int)

            ind1 = np.logical_and(np.all(bone_samples_vox >= 0, axis=1), np.all(bone_samples_vox < vox.dims[0], axis=1))
            bone_samples_vox = np.clip(bone_samples_vox, 0, vox.dims[0]-1)
            ind2 = vox.data[bone_samples_vox[:, 0], bone_samples_vox[:, 1], bone_samples_vox[:, 2]]
            in_flags = np.logical_and(ind1, ind2)
            outside_bone_sample = np.sum(in_flags == False)

            if outside_bone_sample > 1:
                cost_matrix[i, j] = 2 * outside_bone_sample
                cost_matrix[j, i] = 2 * outside_bone_sample
            if np.abs(joint_pos[i, 0]) < 2e-2 and np.abs(joint_pos[j, 0]) < 2e-2:
                cost_matrix[i, j] *= 0.5
                cost_matrix[j, i] *= 0.5
    return cost_matrix

def increase_cost_for_outside_bone_tensor(cost_matrix, joint_pos, vox,resolution=64):
    """
    increase connectivity cost for bones outside the meshs
    vox is a tensor with size(N,3), N is the number of voxels that inside the mesh, range (0,64)
    """
    
    vox = torch.clamp(vox, 0, resolution-1).long()
    for i in range(len(joint_pos)):
        for j in range(i+1, len(joint_pos)):
            bone_samples = sample_on_bone(joint_pos[i], joint_pos[j]) # return coordinates of points on the bone
            bone_samples_vox = bone_samples * (resolution/2) + (resolution/2)
            bone_samples_vox = np.round(bone_samples_vox).astype(int)
            bone_samples_vox = np.clip(bone_samples_vox, 0, resolution-1)
            
            vox_remap = torch.zeros((resolution,resolution,resolution))
            vox_remap[vox[:,0],vox[:,1],vox[:,2]] = 1
            vox_remap = vox_remap.numpy()
            inside_index = vox_remap[bone_samples_vox[:,0],bone_samples_vox[:,1],bone_samples_vox[:,2]]
            outside_bone_sample = np.sum(inside_index == 0)

            
            # check the intersection of the bone with the mesh

            if outside_bone_sample > 1:
                cost_matrix[i, j] = 2 * outside_bone_sample
                cost_matrix[j, i] = 2 * outside_bone_sample
            if np.abs(joint_pos[i, 0]) < 2e-2 and np.abs(joint_pos[j, 0]) < 2e-2:
                cost_matrix[i, j] *= 0.5
                cost_matrix[j, i] *= 0.5
    return cost_matrix


