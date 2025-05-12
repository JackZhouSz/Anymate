import bpy
import mathutils
from mathutils import Vector, Matrix

import os
import sys
import random
import numpy as np
import json
import argparse


IMPORT_FUNCTIONS = {
    "obj": bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)

####################### save json ################################
def save_json(output_path, mesh_obj, armature_obj, extra=None, arm_name=False):
    # makedirs output_path
    os.makedirs(output_path, exist_ok=True)
    
    # start retrieve the information of mesh, skining and rigging
    
    #1. retrieve the information of rigging, save the world matrix of the amature object
    total_armature_info = {}
    for obj in armature_obj:
        # depsgraph = bpy.context.evaluated_depsgraph_get()
        # obj = obj.evaluated_get(depsgraph)
        armature_info = {}
        armature_info["world_matrix"] = [list(row) for row in obj.matrix_world.copy()]
        translation = obj.matrix_world.translation
        for bone in obj.pose.bones:
            bone_info = {}
            bone_info["head_local"] = list(bone.head.copy())
            bone_info["head_world"] = list((obj.matrix_world.to_3x3() @ bone.head+translation).copy())
            # bone_info["matrix_local"] = [list(row) for row in bone.matrix_local.copy()]
            bone_info["tail_local"] = list(bone.tail.copy())
            bone_info["tail_world"] = list((obj.matrix_world.to_3x3() @ bone.tail+translation).copy())

            if bone.parent:
                bone_info["parent"] = bone.parent.name.replace(" ", "_")
                if arm_name:
                    bone_info["parent"] = obj.name + "--" + bone_info["parent"]
            else:
                bone_info["parent"] = None
            bone_info["children"] = []
            if bone.children:
                for child in bone.children:
                    if arm_name:
                        bone_info["children"].append(obj.name + "--" + child.name.replace(" ", "_"))
                    else:
                        bone_info["children"].append(child.name.replace(" ", "_"))
            bone_name = bone.name.replace(" ", "_")
            if arm_name:
                bone_name = obj.name + "--" + bone_name
            armature_info[bone_name] = bone_info
        obj_name = obj.name.replace(" ", "_")
        total_armature_info[obj.name] = armature_info
        
        
    #2. retrieve the informatioon of skining
    total_skinning_info = {}
    for obj in mesh_obj:
        vertex_groups = obj.vertex_groups
        # if not vertex_groups:
        #     continue
        # for group in vertex_groups:
        skinning_info = {}
        skinning_info["world_matrix"] = [list(row) for row in obj.matrix_world.copy()]
        weight_info = []
        for vertex in obj.data.vertices:
            vertex_info = {}
            for group in vertex.groups:
                name = vertex_groups[group.group].name
                name = name.replace(" ", "_")
                if arm_name:
                    arm_modifier = [modifier for modifier in obj.modifiers if modifier.type == 'ARMATURE']
                    assert(len(arm_modifier) == 1)
                    name = arm_modifier[0].object.name + "--" + name
                weight = group.weight
                vertex_info[name] = weight
            weight_info.append(vertex_info)
        skinning_info["weight"] = weight_info
        obj_name = obj.name.replace(" ", "_")
        total_skinning_info[obj_name]=skinning_info


    rigging_file_path = os.path.join(output_path, "rigging.json")
    if extra:
        rigging_file_path = rigging_file_path.replace("rigging.json", f'rigging_{extra}.json')
    with open(rigging_file_path, "w") as f:
        json.dump(total_armature_info, f, indent = 2)
        
    skining_file_path = os.path.join(output_path, "skining.json")
    if extra:
        skining_file_path = skining_file_path.replace("skining.json", f'skining_{extra}.json')
    with open(skining_file_path, "w") as f:
        json.dump(total_skinning_info, f , indent = 2)
        
    
    return rigging_file_path


def apply_skinning_weights(json_file):
    
    with open(json_file, "r") as f:
        skinning_data = json.load(f)

    armature_obj = bpy.data.objects.get("Armature")
    if not armature_obj:
        print("Error: Armature object 'Armature' not found.")
        return
    
    # 将所有网格对象放置在骨骼对象的子集中
    count = 0
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.parent = armature_obj
            count += 1
            
    print("total mesh count:", count)

    for obj in bpy.context.scene.objects:
        vertex_index = 0
        if obj.type == 'MESH':
            mesh_name = obj.name
            if mesh_name in skinning_data:
                skinning_info = skinning_data[mesh_name]
                if "weight" in skinning_info:
                    print("Applying skinning data for mesh:", mesh_name)
                    vertex_index = 0
                    for vertex_weight in skinning_info["weight"]:
                        for bone_name, weight_value in vertex_weight.items():
                            vertex_group = obj.vertex_groups.get(bone_name)
                            if vertex_group is None:
                                vertex_group = obj.vertex_groups.new(name=bone_name)
                                print("Vertex group created:", bone_name)
                            vertex_group.add([vertex_index], weight_value, 'REPLACE')
                        vertex_index += 1
            else:
                print("No skinning data found for mesh:", mesh_name)
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            modifier = obj.modifiers.new(name="Armature", type='ARMATURE')
            modifier.object = armature_obj
            modifier.use_vertex_groups = True
            print("Armature modifier added to mesh:", obj.name)            

def reload_rigging(rigging_file_path):
    with open(rigging_file_path, "r") as f:
        total_armature_info = json.load(f)
        
    bpy.ops.object.armature_add()
    armature_obj = bpy.context.object
    armature_obj.name = "Armature"

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')

    world_matrix = mathutils.Matrix([[1, 0, 0, 0],  
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
    armature_obj.matrix_world = world_matrix
    
    for armature_name, armature_info in total_armature_info.items():
        for bone_name, bone_info in armature_info.items():
            if bone_name == "world_matrix":
                continue
            bone = armature_obj.data.edit_bones.new(bone_name)
            bone.head = bone_info["head_world"]
            bone.tail = bone_info["tail_world"]

        for bone_name, bone_info in armature_info.items():
            if bone_name == "world_matrix":
                continue
            bone = armature_obj.data.edit_bones[bone_name]
            parent_name = bone_info["parent"]
            if parent_name:
                parent_bone = armature_obj.data.edit_bones[parent_name]
                bone.parent = parent_bone
    edit_len = len(armature_obj.data.edit_bones.keys())
    bpy.ops.object.mode_set(mode='OBJECT')  
    bone_len = len(armature_obj.data.bones.keys())
    assert(edit_len == bone_len, "bone number not match!" + str(edit_len) + " " + str(bone_len))
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    print("Rigging information has been reloaded!")

############################# reload json ################################
def reload_json(folder_path, version=0, export = None):
    bpy.ops.wm.read_homefile(use_empty=True)
    if version == 0:
        obj_path = os.path.join(folder_path, "object.obj")
        skinning_file_path = os.path.join(folder_path, "skining.json")
        rigging_file_path = os.path.join(folder_path, "rigging.json")
    elif version == 1:
        obj_path = os.path.join(folder_path, "join.obj")
        skinning_file_path = os.path.join(folder_path, "skining_norig.json")
        rigging_file_path = os.path.join(folder_path, "rigging_norig.json")
    elif version == 2:
        obj_path = os.path.join(folder_path, "join.obj")
        skinning_file_path = os.path.join(folder_path, "skining_norig2.json")
        rigging_file_path = os.path.join(folder_path, "rigging_norig2.json")
    # import_obj(obj_path)
    load_object(obj_path)
    reload_rigging(rigging_file_path)
    apply_skinning_weights(skinning_file_path)
    if export:
        bpy.ops.wm.save_as_mainfile(filepath=export)
    print("Done!")


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def save_mesh(path, mtl=False, obj_path=None):
    if mtl:
        # save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=obj_path + '/object.blend')
        # reopen the blend file
        bpy.ops.wm.open_mainfile(filepath=obj_path + '/object.blend')
        # unpack all the materials and textures to obj_path
        bpy.ops.file.unpack_all(method='WRITE_LOCAL')
    # save to .obj without material
    bpy.ops.wm.obj_export(filepath=path, export_materials=mtl, export_uv=mtl, export_triangulated_mesh=True)


def get_root_obj(obj):
    if not obj.parent:
        return obj
    return get_root_obj(obj.parent)

def normalize(objs):
    # bpy.ops.object.select_all(action='DESELECT')
    # # select objs and join them
    # for obj in objs:
    #     obj.select_set(True)
    # bpy.context.view_layer.objects.active = objs[0]
    # name_join = objs[0].name
    # bpy.ops.object.join()
    # obj_join = bpy.context.active_object
    # print(obj_join.matrix_world)
    # print(name_join)
    # assert(name_join == obj_join.name)

    objs_eval = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objs:
        objs_eval.append(obj.evaluated_get(depsgraph))

    vertices = []
    for obj in objs_eval:
        for v in obj.data.vertices:
            vertices.append(obj.matrix_world @ Vector((v.co.x, v.co.y, v.co.z, 1)))
    
    vertices = np.array(vertices)
    min_x, min_y, min_z, _ = np.min(vertices, axis=0)
    max_x, max_y, max_z, _ = np.max(vertices, axis=0)

    # print(min_x, min_y, min_z)
    # print(max_x, max_y, max_z)

    scale_x = 1 / (max_x - min_x)
    scale_y = 1 / (max_y - min_y)
    scale_z = 1 / (max_z - min_z)
    scale_min = min(scale_x, scale_y, scale_z)

    assert scale_min < 1e6

    translate_x = - (max_x + min_x) / 2 * scale_min
    translate_y = - (max_y + min_y) / 2 * scale_min
    translate_z = - min_z * scale_min

    # form transformation matrix
    trans = Matrix.Translation((translate_x, translate_y, translate_z))
    
    scale = Matrix.Scale(scale_min, 4, (1, 0, 0)) @ Matrix.Scale(scale_min, 4, (0, 1, 0)) @ Matrix.Scale(scale_min, 4, (0, 0, 1))

    # print(trans, scale)
    

    root = get_root_obj(objs[0])
    # print(root.name)
    # print(root.scale)
    # print(root.location)
    # print(root.matrix_world)
    # root.location = mathutils.Vector(root.location) + mathutils.Vector((translate_x, translate_y, translate_z))
    # root.scale = mathutils.Vector(root.scale) * mathutils.Vector((scale_x, scale_y, scale_z))
    
    # add the extra transformation to the root object's world matrix
    root.matrix_world = trans @ scale @ root.matrix_world
    # print(root.name)
    # print(root.scale)
    # print(root.location)
    # print(root.matrix_world)

    # refresh
    bpy.context.view_layer.update()

    ######### check if its successful
    # objs_eval = []
    # depsgraph = bpy.context.evaluated_depsgraph_get()
    # for obj in objs:
    #     objs_eval.append(obj.evaluated_get(depsgraph))

    # vertices = []
    # for obj in objs_eval:
    #     for v in obj.data.vertices:
    #         vertices.append(obj.matrix_world @ Vector((v.co.x, v.co.y, v.co.z, 1)))

    # vertices = np.array(vertices)
    # min_x, min_y, min_z, _ = np.min(vertices, axis=0)
    # max_x, max_y, max_z, _ = np.max(vertices, axis=0)

    # print(min_x, min_y, min_z)
    # print(max_x, max_y, max_z)

def remesh(objs, target=5000):
    num_v = {}
    for obj in objs:
        num_v[obj] = len(obj.data.vertices)

    # sort the num_v dict and make it a dict again
    num_v_sort = sorted(num_v.items(), key=lambda x: x[1], reverse=True)

    # print(num_v_sort)
    total_v = sum([num_v[obj] for obj in num_v])

    iters = 0
    while total_v > target and iters<20:
        reduce = []
        for obj, v in num_v_sort:
            reduce.append(obj)
            if sum([num_v[oo] for oo in reduce]) > 0.5 * total_v:
                break
        for obj in reduce:
            # check if have shape key
            if obj.data.shape_keys is not None:
                # remove obj from num_v
                num_v.pop(obj)
                continue

            ratio = 0.5
            # apply decimate modifier
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_add(type='DECIMATE')
            bpy.context.object.modifiers["Decimate"].ratio = ratio
            bpy.ops.object.modifier_apply(modifier="Decimate")
            # update num_v
            num_v[obj] = len(obj.data.vertices)
        total_v = sum([num_v[obj] for obj in num_v])
        num_v_sort = sorted(num_v.items(), key=lambda x: x[1], reverse=True)
        # print(num_v_sort)
        iters+=1


def get_parents(obj):
    if not obj.parent:
        return [obj.name]
    parents = get_parents(obj.parent)
    parents.append(obj.name)
    return parents

def check(objs, arm):
    # assert('Sketchfab_model' in bpy.data.objects)

    # root_arm = get_root_obj(arm)
    # for obj in objs:
    #     if root_arm != get_root_obj(obj):
    #         print('not same root')
    #         return -1
    # return 1

    # action_num = 0
    # actions = bpy.data.actions
    # for act in actions:
    #     action_num += 1
    #     fcurves = act.fcurves
    #     data_paths = []
    #     not_pose = False
    #     for fcurve in fcurves:
    #         data_paths.append(fcurve.data_path)
    #         if not fcurve.data_path.startswith('pose.bones'):
    #             # print(fcurve.data_path)
    #             not_pose = True
    #             # return -1
    #     if not_pose:
    #         print('zyhsb')
    #         print(data_paths)
    #         return -1
    # return action_num

    for obj in objs:
        vertex_groups = obj.vertex_groups
        # if not vertex_groups:
        #     continue
        # for group in vertex_groups:
        for vertex in obj.data.vertices:
            vertex_info = {}
            for group in vertex.groups:
                name = vertex_groups[group.group].name
                name = name.replace(" ", "_")
                if True:
                    arm_modifier = [modifier for modifier in obj.modifiers if modifier.type == 'ARMATURE']
                    if len(arm_modifier) != 1:
                        print('zyhsb', len(arm_modifier))
                        return -2
                    # name = arm_modifier[0].object.name + "--" + name
    return 1

    # for obj in objs:
    #     if obj.data.shape_keys is not None:
    #         return 1
    #         # only 942!!!
    # return 0


def delete(objs):
    # check if the mesh object has skinning weight
    for obj in objs:
        vertex_groups = obj.vertex_groups
        if not vertex_groups:
            # delete the object
            bpy.data.objects.remove(obj)
            # print('delete!!!')
    meshes = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            meshes.append(obj)
    
    return meshes


def merge_mesh(folder_path, export = None, save_join = True):
    # output_path = os.path.join(folder_path, "rigging_norig.json")
    # if os.path.exists(output_path):
    #     print("Already processed folder:", folder_path)
    #     return
    bpy.ops.wm.read_homefile(use_empty=True)
    try:
        reload_json(folder_path)
    except:
        print("Error in reloading json file")
        # remove the folder
        os.system(f"rm -r {folder_path}")
        return None, None
    
    bpy.ops.object.select_all(action='DESELECT')
    if export:
        bpy.ops.wm.save_as_mainfile(filepath='reload_' + export)

    meshes = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            meshes.append(obj)
    print("meshes length", len(meshes))

    bpy.ops.object.join()
    if export:
        bpy.ops.wm.save_as_mainfile(filepath='join_' + export)

    meshes = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            meshes.append(obj)
    if len(meshes) != 1:
        bpy.ops.wm.save_as_mainfile(filepath='join_f.blend')
    assert len(meshes) == 1
    # remesh(meshes[0])


    if save_join:
        obj_path = os.path.join(folder_path, "object.obj")
        bpy.ops.wm.obj_export(filepath=obj_path, export_materials=False, export_uv=False, export_triangulated_mesh=True)
    # mesh = trimesh.load(glb_file_path)
    # mesh.export(obj_path, file_type='obj')


    # save to json file
    total_armature_count = 0
    armature_obj = []
    mesh_obj = []
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            total_armature_count += 1
            armature_obj.append(obj)
        if obj.type == "MESH":
            mesh_obj.append(obj)
    if total_armature_count == 0:
        print("No rigging information for the file:", folder_path+"\n")
        return None, None
    

    ######### delete bones that are not in the vertex group
    vertex_group_name = [group.name for group in mesh_obj[0].vertex_groups]
    bpy.context.view_layer.objects.active = armature_obj[0]
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature_obj[0].data.edit_bones
    bone_delete = set([bone.name for bone in edit_bones]) - set(vertex_group_name)
    print(f"Deleting {len(bone_delete)} bones")
    for bone in bone_delete:
        # if the bone is root, then do not delete it
        if edit_bones[bone].parent == None:
            # return len([1 for child in edit_bones[bone].children if child.name in bone_delete])
            num_children = len(edit_bones[bone].children)
            if num_children <= 1:
                edit_bones.remove(edit_bones[bone])
                continue
            if num_children > 1:
                center = mathutils.Vector((0, 0, 0))
                for child in edit_bones[bone].children:
                    center += child.head
                center /= num_children
                min_dist = 1e9
                for child in edit_bones[bone].children:
                    dist = (child.head - center).length
                    if dist < min_dist:
                        min_dist = dist
                        min_child = child
                for child in edit_bones[bone].children:
                    if child != min_child:
                        child.parent = min_child
                edit_bones.remove(edit_bones[bone])
                continue
            continue
        # assign bone's children to bone's parent
        bone_obj = edit_bones[bone]
        for child in bone_obj.children:
            child.parent = bone_obj.parent

        edit_bones.remove(edit_bones[bone])
    bpy.ops.object.mode_set(mode='OBJECT')

    if export:
        bpy.ops.wm.save_as_mainfile(filepath='delete_' + export)
    
    mesh_obj = []
    armature_obj = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            mesh_obj.append(obj)
        if obj.type == "ARMATURE":
            armature_obj.append(obj)
    assert len(mesh_obj) == 1
    assert len(armature_obj) == 1

    return mesh_obj, armature_obj


def process(file_path, obj_path=None, stamp=None, tex=False):
    # check if obj_path exists
    # if os.path.exists(obj_path + '/object.obj'):
    #     print('object.obj exists')
    #     return True
    reset_scene()
    load_object(file_path)
    # bpy.ops.import_scene.gltf(filepath=glb_file_path)

    # delete hierarchy collections['glTF_not_exported']
    if 'glTF_not_exported' in bpy.data.collections:
        print('DELETE glTF_not_exported')
        bpy.data.collections.remove(bpy.data.collections['glTF_not_exported'])

    if stamp is not None:
        # Set the current frame to the stamp value
        bpy.context.scene.frame_set(stamp)
        print(f'Set the current frame to {stamp}')
        
        # Ensure all objects are updated to this frame
        bpy.context.view_layer.update()

    mesh_obj = []
    armature_obj = []
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            # if len(armature_obj) > 0:
            #     print(file_path, 'has more than 1 armature')
            #     return -2
            armature_obj.append(obj)
            # obj.show_in_front = True
            armature_obj[-1].data.pose_position = 'POSE'
        if obj.type == "MESH":
            mesh_obj.append(obj)
            # if obj.data.shape_keys is not None:
            #     return False

    # mesh_obj = delete(mesh_obj)
    # if len(mesh_obj) == 0:
    #     # print('zyhsb -1', file_path, obj_path)
    #     return -1
    # return check(mesh_obj, armature_obj)


    # total_vertices = np.array([len(obj.data.vertices) for obj in mesh_obj]).sum()
    # if total_vertices < 1000: return
    # if total_vertices > 10000: remesh(mesh_obj)


    # bpy.ops.object.select_all(action='DESELECT')
    # armature_obj.select_set(True)
    # execute(bpy.context)


    # normalize(mesh_obj)


    mesh_obj = delete(mesh_obj)
    if len(mesh_obj) == 0:
        # print('zyhsb -1', file_path, obj_path)
        return -1


    save_json(obj_path, mesh_obj, armature_obj, arm_name=True)


    if not tex:
        save_mesh(obj_path + '/object.obj')
    else:
        save_mesh(obj_path + '/object.obj', mtl=True, obj_path=obj_path)


    mesh_obj, armature_obj = merge_mesh(obj_path)
    if mesh_obj is None or armature_obj is None:
        # print('zyhsb -2', file_path, obj_path)
        return -2

    
    try:
        normalize(mesh_obj)
    except:
        os.system(f"rm -r {obj_path}")
        # print('zyhsb -3', file_path, obj_path)
        return -3


    save_json(obj_path, mesh_obj, armature_obj)

    if not tex:
        save_mesh(obj_path + '/object.obj')
    else:
        save_mesh(obj_path + '/object.obj', mtl=True, obj_path=obj_path)


    return 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--stamp",
        type=int,
        required=False,
        help="Stamp to be used for the rendering.",
    )
    parser.add_argument(
        "--tex",
        type=bool,
        required=False,
        help="Save the texture.",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = args.stamp if args.stamp else None
    print(f'Stamp: {stamp}')
    result = process(args.object_path, obj_path=args.output_dir, stamp=stamp, tex=args.tex)
    # import numpy as np
    # os.makedirs(args.output_dir, exist_ok=True)  # the directory may be removed
    # np.save(args.output_dir + '/result.npy', np.array(result))