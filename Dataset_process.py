import json
import multiprocessing
import os
import platform
import random
import subprocess
import tempfile
import time
import zipfile
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import fire
import fsspec
import GPUtil
import pandas as pd
from loguru import logger

import objaverse.xl as oxl
from objaverse.utils import get_uid_from_str
from tqdm import tqdm
import csv
import sys

import trimesh
import numpy as np
import torch
from safetensors.torch import save_file
from Anymate.utils.dataset_utils import reduce, align, get_skin_direction, obj2mesh, sparse_to_index
from ThirdParty.Rignet_utils import binvox_rw

def process_to_tensor(obj_path, stamp=None):
    data = {}
    if stamp == None:
        data['name'] = '/'.join(obj_path.split('/')[-2:])
    else:
        data['name'] = '/'.join(obj_path.split('/')[-3:])

    mesh = obj2mesh(os.path.join(obj_path, 'object.obj'))
    points, face_idx = mesh.sample(8192, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points, normals], axis=-1)
    y_max = np.max(pc_normal[:,1])
    pc_normal = pc_normal * np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    pc_normal[:,1] = pc_normal[:,1] - y_max
    data['pc'] = torch.tensor(pc_normal, dtype=torch.float32)

    # try:
    binvox_path = os.path.join(obj_path, 'object.binvox')
    with open(binvox_path, 'rb') as fvox:
        vox = binvox_rw.read_as_3d_array(fvox)
    vox = reduce(vox)
    vox_data = align(vox, y_max)
    vox_data = torch.from_numpy(vox_data).to(torch.int8)
    data['vox'] = vox_data

    # except Exception as e:
    #     print(e)
    #     print('Error in processing {:s}'.format(data['name']))



    folder = obj_path
    # try:
    rigging_file_path = os.path.join(folder, 'rigging.json')
    with open(rigging_file_path, "r") as f:
        rig_info = json.load(f)

    joints = []
    parent_index = np.zeros((64,1),np.int8)-1
    index_mapping = {}
    parent_mapping = {}
    joint_count = 0
    for armature_name, armature_data in rig_info.items():
        for bone_name, bone_data in armature_data.items():
            if bone_name.endswith("world_matrix"):
                continue 
            if bone_data['parent'] == None:
                parent_mapping[bone_name] = bone_name
            else:
                parent_mapping[bone_name] = bone_data['parent']
            index_mapping[bone_name] = joint_count
            joint_count += 1

            head_world = bone_data["head_world"]
            joints.append([head_world[0], head_world[2], -head_world[1]])

    joints = np.array(joints)
    joints = torch.from_numpy(joints).float()
    joints = joints * 2.0
    joints[:,1] = joints[:,1] - y_max

    data['bones_num'] = joints.shape[0]
    bones_num = data['bones_num']

    bones = torch.ones((64, 6)) * (-3)
    bones[:bones_num, :3] = joints
    
    # Check for NaN values in head positions
    if torch.isnan(bones[:bones_num, :3]).any():
        raise ValueError(f"NaN values found in head positions for {data['name']}")
    



    ################### skin ###################
    ################### skin ###################

    skin_path = folder + '/skining.json'
    with open(skin_path, 'r') as f:
        skin = json.load(f)
        assert len(skin) == 1
        skin_data = skin[list(skin.keys())[0]]['weight']
        assert len(skin_data) == len(mesh.vertices)

    barycentric = trimesh.triangles.points_to_barycentric(mesh.triangles[face_idx], points)

    skin_index = []
    skin_weight = []
    for i in range(len(points)):
        skin_matrix = np.zeros((data['bones_num'], 3))
        for j, vertex in enumerate(mesh.faces[face_idx[i]]):
            for joint, weight in skin_data[vertex].items():
                skin_matrix[index_mapping[joint], j] = weight

        combined_skin_matrix = np.dot(skin_matrix, barycentric[i])

        index, weight = sparse_to_index(combined_skin_matrix)
        skin_index.append(torch.tensor(index, dtype=torch.int8))
        skin_weight.append(torch.tensor(weight, dtype=torch.float16))

    

    pad_token = -1  # int8 is [-1, 254]
    max_len = max([len(i) for i in skin_index])
    skin_index = [torch.cat([i, torch.tensor([pad_token] * (max_len - len(i)), dtype=torch.int8)]) for i in skin_index]
    skin_weight = [torch.cat([i, torch.tensor([0.0] * (max_len - len(i)), dtype=torch.float16)]) for i in skin_weight]


    data['skins_index'] = torch.stack(skin_index)
    data['skins_weight'] = torch.stack(skin_weight)



    ###########################conn&joint##########################
    #########################################################

    
    for bone_name, index in index_mapping.items():
        parent_index[index] = index_mapping[parent_mapping[bone_name]]
    parent_indices = torch.tensor(parent_index, dtype=torch.int8)[:bones_num]
    
    # Count number of children for each parent
    children_count = torch.zeros(bones_num, dtype=torch.int)
    for joint_idx in range(bones_num):
        parent_idx = parent_indices[joint_idx].item()
        if parent_idx != joint_idx:
            children_count[parent_idx] += 1

    reset_list = []
    
    # if repeated assign, children should have same position
    for joint_idx in range(bones_num):
        parent_idx = parent_indices[joint_idx].item()
        if parent_idx != joint_idx:
            if torch.equal(bones[parent_idx, 3:], torch.tensor([-3, -3, -3]).float()):
                if not torch.equal(bones[parent_idx, :3], bones[joint_idx, :3]):
                    # Set parent's tail position (last 3 dimensions) to current joint's head position
                    bones[parent_idx, 3:] = bones[joint_idx, :3]
            else:
                assert children_count[parent_idx] > 1, f"Error: children_count[{parent_idx}] <= 1 in {data['name']}"
                if (bones[parent_idx, :3] != bones[joint_idx, :3]).any():
                    # reset to -3
                    if parent_idx not in reset_list:
                        reset_list.append(parent_idx)
    
    # Set tail positions for bones that don't have them yet
    for joint_idx in range(bones_num):
        if torch.equal(bones[joint_idx, 3:], torch.tensor([-3, -3, -3]).float()):
            # assert children_count[joint_idx] == 0, f"Error: children_count[{joint_idx}] != 0 in {data['name']}"
            skin_direction = get_skin_direction(joint_idx, data, parent_indices, joints)
            bones[joint_idx, 3:] = bones[joint_idx, :3] + skin_direction

    # assert all the tail positions are unique
    unique_count = torch.unique(bones[:bones_num, 3:], dim=0).size(0)
    total_count = bones[:bones_num, 3:].size(0)
    if unique_count != total_count:
        assert False

    for joint_idx in reset_list:
        assert children_count[joint_idx] > 1, f"Error: children_count[{joint_idx}] <= 1 in {data['name']}"
        skin_direction = get_skin_direction(joint_idx, data, parent_indices, joints)
        bones[joint_idx, 3:] = bones[joint_idx, :3] + skin_direction

    
    # Final check for NaN values
    if torch.isnan(bones[:bones_num]).any():
        raise ValueError(f"NaN values found in final bones for {data['name']}")
    
    data['bones'] = bones[:bones_num]

    ################### conn
    head_positions = bones[:bones_num, :3]
    tail_positions = bones[:bones_num, 3:]

    # Collect all positions (both head and tail)
    all_positions = torch.cat([head_positions, tail_positions], dim=0)

    # Find unique positions with tolerance
    unique_positions = []
    position_to_idx = {}
    
    for i in range(all_positions.size(0)):
        pos = all_positions[i]
        found_match = False
        for j, unique_pos in enumerate(unique_positions):
            if torch.equal(pos, unique_pos):
                position_to_idx[i] = j
                found_match = True
                break
        if not found_match:
            position_to_idx[i] = len(unique_positions)
            unique_positions.append(pos)

    # Convert unique positions to tensor
    unique_positions = torch.stack(unique_positions)
    data['joints'] = unique_positions
    data['joints_num'] = len(unique_positions)
    
    # create conn
    conn = torch.ones(unique_positions.size(0)) * -1
    for i in range(bones_num):
        head_idx = position_to_idx[i]
        tail_idx = position_to_idx[i+bones_num]
        if conn[tail_idx] != -1:
            assert conn[tail_idx] == head_idx, f"Error: conn[{tail_idx}] != head_idx ({i}) in {data['name']}"
        assert head_idx != tail_idx, f"Error: head_idx equals tail_idx ({i}) in {data['name']}"
        conn[tail_idx] = head_idx

    # Count number of children for each parent
    children_count = torch.zeros(bones_num, dtype=torch.int)
    for joint_idx in range(bones_num):
        parent_idx = parent_indices[joint_idx].item()
        if parent_idx != joint_idx:
            children_count[parent_idx] += 1
    for i in range(bones_num):
        parent_idx = parent_indices[i].item()
        if parent_idx == i:
            assert conn[position_to_idx[i+bones_num]] == position_to_idx[i], f"Error: conn[tail_idx] != position_to_idx[head_idx] in {data['name']}"
            continue
        assert children_count[parent_idx] != 0, f"Error: children_count[{parent_idx}] == 0 in {data['name']}"

        if parent_idx in reset_list:
            if conn[position_to_idx[i]] != -1:
                # assert conn[position_to_idx[i]] == position_to_idx[parent_idx+bones_num], f"Error: conn[{position_to_idx[i]}] != position_to_idx[{parent_idx+bones_num}] in {data['name']}"
                ## it is a special case that's fine, it's just the child has same head position with its parent
                assert conn[position_to_idx[parent_idx+bones_num]] == position_to_idx[parent_idx], f"Error: position_to_idx[{parent_idx+bones_num}] != -1 in {data['name']}"
            else:
                conn[position_to_idx[i]] = position_to_idx[parent_idx+bones_num]
    for i in range(unique_positions.size(0)):
        if conn[i] == -1:
            conn[i] = i
    data['conns'] = conn.to(torch.int8)

    safetensor_path = folder + '/object.pt'
    torch.save(data, safetensor_path)
    # save_file(data, safetensor_path)
    return data

    # except Exception as e:
    #     print(e)
    #     print('Error in processing {:s}'.format(data['name']))


def handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[str, Any],
    save_dir: str,
    anymate_metadata: dict,
) -> bool:
    """Called when an object is successfully found and downloaded.

    Here, the object has the same sha256 as the one that was downloaded with
    Objaverse-XL. If None, the object will be downloaded, but nothing will be done with
    it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): File identifier of the 3D object.
        sha256 (str): SHA256 of the contents of the 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        num_renders (int): Number of renders to save of the object.
        render_dir (str): Directory where the objects will be rendered.
        only_northern_hemisphere (bool): Only render the northern hemisphere of the
            object.
        gpu_devices (Union[int, List[int]]): GPU device(s) to use for rendering. If
            an int, the GPU device will be randomly selected from 0 to gpu_devices - 1.
            If a list, the GPU device will be randomly selected from the list.
            If 0, the CPU will be used for rendering.
        render_timeout (int): Number of seconds to wait for the rendering job to
            complete.
        successful_log_file (str): Name of the log file to save successful renders to.
        failed_log_file (str): Name of the log file to save failed renders to.

    Returns: True if the object was rendered successfully, False otherwise.
    """

    fs, path = fsspec.core.url_to_fs(save_dir)

    try:
        output_dir = os.path.join(save_dir, anymate_metadata[sha256]['name'])
        os.makedirs(output_dir, exist_ok=True)

        fs.put(
            local_path,
            os.path.join(output_dir, f'original.{local_path.split(".")[-1]}'),
        )

        args = f"--object_path '{local_path}'"
        args += f" --output_dir '{output_dir}'"
        command = f"ThirdParty/blender-4.0.0-linux-x64/blender --background --python Anymate/blender_script.py -- {args}"

        subprocess.run(
            ["bash", "-c", command],
            timeout=120,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        binvox_path = os.path.join(output_dir, 'object.binvox')
        # Check if binvox file exists and delete it
        if os.path.exists(binvox_path): 
            os.remove(binvox_path)

        # os.system(f"ThirdParty/binvox -d 64 -dc {os.path.join(output_dir, 'object.obj')}")
        #### this requires screen by running the following command
        #### Xvfb :99 -screen 0 640x480x24 &
        #### export DISPLAY=:99

        os.system(f"ThirdParty/binvox -d 64 -e {os.path.join(output_dir, 'object.obj')}")
        #### this does not require screen

        process_to_tensor(output_dir)

        if 'extra_frame' in anymate_metadata[sha256].keys():
            for frame in anymate_metadata[sha256]['extra_frame']:
                output_dir = os.path.join(save_dir, anymate_metadata[sha256]['name'], f'{frame}')
                os.makedirs(output_dir, exist_ok=True)

                fs.put(
                    local_path,
                    os.path.join(output_dir, f'original.{local_path.split(".")[-1]}'),
                )

                args = f"--object_path '{local_path}'"
                args += f" --output_dir '{output_dir}'"
                args += f" --stamp {frame}"
                command = f"ThirdParty/blender-4.0.0-linux-x64/blender --background --python Anymate/blender_script.py -- {args}"

                subprocess.run(
                    ["bash", "-c", command],
                    timeout=120,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                binvox_path = os.path.join(output_dir, 'object.binvox')
                # Check if binvox file exists and delete it
                if os.path.exists(binvox_path): 
                    os.remove(binvox_path)
                
                # os.system(f"ThirdParty/binvox -d 64 -dc {os.path.join(output_dir, 'object.obj')}")
                #### this requires screen by running the following command
                #### Xvfb :99 -screen 0 640x480x24 &
                #### export DISPLAY=:99

                os.system(f"ThirdParty/binvox -d 64 -e {os.path.join(output_dir, 'object.obj')}")
                #### this does not require screen

                process_to_tensor(output_dir, frame)

        return True

    except subprocess.TimeoutExpired:
        logger.error(f"Timed out for object {sha256}: {e}")
        return False
    except subprocess.CalledProcessError:
        logger.error(f"Failed to process object {sha256}: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to save object {sha256}: {e}")
        return False

def get_example_objects() -> pd.DataFrame:
    """Returns a DataFrame of example objects to use for debugging."""
    return pd.read_json("example-objects.json", orient="records")

def get_objects_oxl(id_path = 'Anymate/data/Anymate_id.json'):

    with open(id_path, 'r') as f:
        ids = json.load(f)[:NUM_OBJECTS]

    sha256s = [id['id'] for id in ids]
    metadata = {}
    for id in ids:
        metadata[id['id']] = id
    
    annotations = oxl.get_annotations()
    annotations = annotations[annotations["sha256"].isin(sha256s)]

    return annotations, metadata

def download_oxl(
    download_dir: Optional[str] = None,
    save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]] = None,
) -> None:
    """Renders objects in the Objaverse-XL dataset with Blender

    Args:
        render_dir (str, optional): Directory where the objects will be rendered.
        download_dir (Optional[str], optional): Directory where the objects will be
            downloaded. If None, the objects will not be downloaded. Defaults to None.
        num_renders (int, optional): Number of renders to save of the object. Defaults
            to 12.
        processes (Optional[int], optional): Number of processes to use for downloading
            the objects. If None, defaults to multiprocessing.cpu_count() * 3. Defaults
            to None.
        save_repo_format (Optional[Literal["zip", "tar", "tar.gz", "files"]], optional):
            If not None, the GitHub repo will be deleted after rendering each object
            from it.
        only_northern_hemisphere (bool, optional): Only render the northern hemisphere
            of the object. Useful for rendering objects that are obtained from
            photogrammetry, since the southern hemisphere is often has holes. Defaults
            to False.
        render_timeout (int, optional): Number of seconds to wait for the rendering job
            to complete. Defaults to 300.
        gpu_devices (Optional[Union[int, List[int]]], optional): GPU device(s) to use
            for rendering. If an int, the GPU device will be randomly selected from 0 to
            gpu_devices - 1. If a list, the GPU device will be randomly selected from
            the list. If 0, the CPU will be used for rendering. If None, all available
            GPUs will be used. Defaults to None.

    Returns:
        None
    """

    objects, metadata = get_objects_oxl()
    if len(objects) == 0:
        logger.info("No objects to download.")
        return
    objects.iloc[0]["fileIdentifier"]
    objects = objects.copy()
    logger.info(f"Provided {len(objects)} objects to download.")

    objects = objects.reset_index(drop=True)
    logger.info(f"Downloading {len(objects)} new objects.")

    oxl.download_objects(
        objects=objects,
        processes=NUM_PROCESSES,
        save_repo_format=save_repo_format,
        download_dir=download_dir,
        handle_found_object=partial(
            handle_found_object,
            save_dir=SAVE_DIR,
            anymate_metadata=metadata,
        ),
    )


if __name__ == "__main__":
    NUM_OBJECTS = 5  # only process 5 objects for testing
    NUM_PROCESSES = 72
    SAVE_DIR = 'Anymate/data'
    os.makedirs(SAVE_DIR, exist_ok=True)
    fire.Fire(download_oxl)

