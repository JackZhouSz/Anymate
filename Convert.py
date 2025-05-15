import argparse
import bpy
import mathutils
from Anymate.blender_script import load_object, save_mesh
from Anymate.utils.render_utils import empty


def parse_args():
    parser = argparse.ArgumentParser(description='Anymate rendering script')
    parser.add_argument('--path', type=str, required=True, help='Path to the model file')
    return parser.parse_args()

args = parse_args()

print(f"Starting converting {args.path} to obj format...")

# empty the scene
empty()

# load the glb file
load_object(args.path)

# save the mesh
save_mesh(args.path.replace('.glb', '.obj'))