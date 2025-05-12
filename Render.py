import argparse
import bpy
import mathutils
from Anymate.utils.render_utils import empty, setup_armature


def parse_args():
    parser = argparse.ArgumentParser(description='Anymate rendering script')
    parser.add_argument('--path', type=str, required=True, help='Path to the model file')
    return parser.parse_args()

args = parse_args()

print(f"Starting converting {args.path} to blender format...")

empty()
setup_armature(args.path)