import argparse
import glob
import math
import os
import random

import bpy

ROTATION_MODE = 'XYZ'

def main():
    parser = argparse.ArgumentParser(
        description='''Convert .fbx/.blend meshes to normalized .blend meshes''')
    parser.add_argument('--input-directory', type=str,
                        default='input', help='Input directory that contains the models (.fbx, .blend)')
    parser.add_argument('--output-directory', type=str,
                        default='output', help='Output directory')
    parser.add_argument('--pose-preprocess-params-json', type=str, default=None, help='Path to pose preprocess params JSON file')
    parser.add_argument('--keypoint-names-file', type=str, default=None, help='Path to keypoint names file')
    parser.add_argument('--skip-randomize-pose', action='store_true', help='Skip randomizing pose')
    parser.add_argument('--rotation-range-generator-path', type=str, default=None, help='Path to rotation range generator Python script')
    parser.add_argument('--chunk-size', type=int, default=None, help='Chunk size')
    parser.add_argument('--start-index', type=int, default=None, help='Start index')
    args = parser.parse_args()
    process_meshes(args.input_directory, args.output_directory, args)


def process_meshes(input_directory, output_directory, args):
    fbx_files = glob.glob(os.path.join(input_directory, '*.fbx'))
    blend_files = glob.glob(os.path.join(input_directory, '*.blend'))
    input_mesh_paths = fbx_files + blend_files
    
    if args.chunk_size != None and args.start_index != None:
        input_mesh_paths.sort()
        input_mesh_paths = input_mesh_paths[args.start_index:args.start_index+args.chunk_size]

    for input_mesh_path in input_mesh_paths:
        # Clear the file
        bpy.ops.wm.read_homefile()

        output_blend_path = os.path.join(output_directory, os.path.splitext(
            os.path.basename(input_mesh_path))[0] + '.blend')
        
        process_mesh(input_mesh_path, output_blend_path, args)


def process_mesh(input_mesh_path, output_blend_path=None, args=None):
    print(f"Processing {input_mesh_path}")

    # Clear the file
    bpy.ops.wm.read_homefile(use_empty=True)

    # Clear the scene
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Step 1) Convert FBX to .blend
    if input_mesh_path.endswith('.fbx'):
        bpy.ops.import_scene.fbx(filepath=input_mesh_path)
    elif input_mesh_path.endswith('.blend'):
        bpy.ops.wm.open_mainfile(filepath=input_mesh_path)

    # Select armature
    armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']

    if len(armatures) != 1:
        raise Exception(f"Expected exactly one armature, but found {len(armatures)}.")
    
    armature = armatures[0]
    
    fix_leaf_joints(armature, args)

    if not args.skip_randomize_pose:
        randomize_pose(args)

    # Set output path if necessary
    if output_blend_path is None:
        output_blend_path = os.path.splitext(input_mesh_path)[0] + '.blend'

    # Create output directory if necessary
    os.makedirs(os.path.dirname(output_blend_path), exist_ok=True)

    # Save the mesh
    bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)


def fix_leaf_joints(armature, args):
    import json
    if args is not None and args.pose_preprocess_params_json is not None:
        with open(args.pose_preprocess_params_json, 'r') as f:
            pose_preprocess_params = json.load(f)
    else:
        pose_preprocess_params = {
            'additional_leaf_bones': [],
        }
    scene = bpy.context.scene
    depsgraph = bpy.context.view_layer.depsgraph
    # Deselect all objects
    for ob in scene.objects:
        ob.select_set(False)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    leaf_bones = [bone.name for bone in armature.data.bones if not bone.children]

    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones

    for bone_name in pose_preprocess_params['additional_leaf_bones']:
        if bone_name not in leaf_bones:
            leaf_bones.append(bone_name)
        else:
            print(f"Warning: bone {bone_name} is already in leaf bones.")

    for bone_name in leaf_bones:
        bone = edit_bones[bone_name]
        head = armature.matrix_world @ bone.head
        tail = armature.matrix_world @ bone.tail
        previous_length = (tail - head).magnitude
        did_hit, hit_location, *_ = scene.ray_cast(depsgraph, head, tail - head)
        if did_hit:
            new_length = (hit_location - head).magnitude
            if new_length < previous_length:
                if new_length > previous_length / 3:
                    bone.tail = bone.head + (armature.matrix_world.inverted() @ hit_location - bone.head) * 0.95
                else:
                    did_hit, hit_location, *_ = scene.ray_cast(depsgraph, tail, head - tail)
                    if did_hit:
                        new_length = (hit_location - head).magnitude
                        if new_length < previous_length:
                            if new_length > previous_length / 3:
                                bone.tail = bone.head + (armature.matrix_world.inverted() @ hit_location - bone.head) * 0.95
                            else:
                                print(f"Warning: bone {bone_name} possibly too short after fixing leaf joint. Skipping...")
        else:
            print(f"Error occured on this model, could not hit raycast for bone {bone_name}.")

    bpy.ops.object.mode_set(mode='OBJECT')


def randomize_pose(args):
    import json
    if args is not None and args.pose_preprocess_params_json is not None:
        with open(args.pose_preprocess_params_json, 'r') as f:
            pose_preprocess_params = json.load(f)
    else:
        pose_preprocess_params = {
            'collision_check_ignore_bones': [],
            'removed_valid_bone_names': []
        }
            
    # Select the armature and enter pose mode
    armature = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    # Triangulate meshes
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']

    for mesh in meshes:
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        bpy.ops.object.mode_set(mode='OBJECT')

    # Select the armature and enter pose mode
    armature = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    def generate_rotation_ranges(rotation_range_generator_path=None):
        import importlib
        import importlib.util

        if rotation_range_generator_path is None:
            rotation_range_generator_path = os.path.join(os.path.dirname(__file__), 'rotation_range_generator.py')
        
        if not os.path.exists(rotation_range_generator_path):
            raise Exception(f"Could not find rotation range generator at {rotation_range_generator_path}.")

        spec = importlib.util.spec_from_file_location("rotation_range_generator", rotation_range_generator_path)
        rotation_range_generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rotation_range_generator)
        return rotation_range_generator.generate_rotation_ranges()

    # Generate rotation ranges
    if args is not None and args.rotation_range_generator_path is not None:
        rotation_ranges = generate_rotation_ranges(args.rotation_range_generator_path)
    else:
        rotation_ranges = generate_rotation_ranges()

    rotation_euler_by_bone = {}

    # Get rest pose rotations for individual bones
    for bone_name in rotation_ranges.keys():
        bone = armature.pose.bones[bone_name]   
        bone.rotation_mode = ROTATION_MODE      
        rotation_euler_by_bone[bone_name] = bone.rotation_euler.copy()
    
    rotation_ranges = list(rotation_ranges.items())
    random.shuffle(rotation_ranges)

    # For each bone, find all ray hits before tail in the rest pose.
    def find_all_ray_hits_before_tail(bone):
        scene = bpy.context.scene
        depsgraph = bpy.context.view_layer.depsgraph
        armature = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
        hits = []
        head = armature.matrix_world @ bone.head
        tail = armature.matrix_world @ bone.tail
        origin = head
        direction = tail - head
        did_hit, hit_location, *_ = scene.ray_cast(depsgraph, origin, direction)
        while did_hit and (hit_location - origin).magnitude < (tail - origin).magnitude:
            hits.append(hit_location)
            origin = hit_location
            did_hit, hit_location, *_ = scene.ray_cast(depsgraph, hit_location, direction)
            # Needed this part because sometimes the raycast would hit the same point where the ray started
            if (hit_location - origin).magnitude < 1e-5:
                break
        return hits
    
    ray_hits_before_tail_by_bone = {}

    # Get all ray hits before tail for individual bones

    for bone_name, _ in rotation_ranges:
        bone = armature.pose.bones[bone_name]
        ray_hits_before_tail_by_bone[bone_name] = find_all_ray_hits_before_tail(bone)

    # Iterate over the bones and set their rotation values
    for bone_name, ((min_rot_x, max_rot_x), (min_rot_y, max_rot_y), (min_rot_z, max_rot_z)) in rotation_ranges:
        armature = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
        bone = armature.pose.bones[bone_name]
        bone.rotation_mode = ROTATION_MODE

        # -- Collision checking --
        # Since we are rotating each bone individually, we can check for collisions after each rotation.
        # If a collision is detected, we can undo the rotation and try again.

        valid_rotation = False
        left_iter = 100
        while not valid_rotation and left_iter > 0:
            left_iter -= 1
            rot_x = random.uniform(min_rot_x, max_rot_x)
            rot_y = random.uniform(min_rot_y, max_rot_y)
            rot_z = random.uniform(min_rot_z, max_rot_z)
            bone.rotation_euler[0] = math.radians(rot_x)
            bone.rotation_euler[1] = math.radians(rot_y)
            bone.rotation_euler[2] = math.radians(rot_z)

            # Update bone
            bpy.context.view_layer.update()
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.mode_set(mode='POSE')

            # For each bone, raycast into own direction and see if length from bone head to hit point is less than bone length, which would mean a collision.
            armature = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
            for bone2 in armature.pose.bones:
                valid_bone_names = [bone_name for bone_name, _ in rotation_ranges]

                if bone2.name not in valid_bone_names:
                    continue

                # For some bones collisions with the mesh is a normal factor of their design or they are outside the mesh
                if bone2.name in pose_preprocess_params['collision_check_ignore_bones']:
                    continue

                num_ray_hits_before_tail_before_rotation = len(ray_hits_before_tail_by_bone[bone2.name])
                num_ray_hits_before_tail_after_rotation = len(find_all_ray_hits_before_tail(bone2))

                if num_ray_hits_before_tail_after_rotation != num_ray_hits_before_tail_before_rotation:
                    print(f'Collision detected for bone {bone2.name}. Was {num_ray_hits_before_tail_before_rotation}, now {num_ray_hits_before_tail_after_rotation}.')
                    valid_rotation = False
                    break
                else:
                    valid_rotation = True

        if left_iter == 0:
            print(f'Could not find valid rotation for bone {bone_name}. Skipping...')
            continue
            
    # Update the scene
    bpy.context.view_layer.update()

if __name__ == '__main__':
    main()