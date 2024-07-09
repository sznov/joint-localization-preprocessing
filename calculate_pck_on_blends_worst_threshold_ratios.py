import argparse
import os
import pathlib

import bpy
import numpy as np
import pandas
from mathutils import Matrix

# For percentages get from 0.0001 to 0.005 in steps of 0.0001
# NEIGHBOURHOOD_PERCENTAGES = [i / 10000 for i in range(1, 51)]

# Threshold multipliers from 0.01 to 1.0 in steps of 0.01
THRESHOLD_MULTIPLIERS = [i / 100 for i in range(1, 201)]

def create_joint_to_bones_map():
    # We have the armature and spheres for each keypoint of the form BONE_NAME or BONE_NAME_TAIL (for the tail of the leaf bones)
    # Get the armature
    armature = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]

    # Get the list of bones
    bones = [bone for bone in armature.pose.bones]

    # Get the list of keypoints
    keypoint_names = [keypoint.name for keypoint in bpy.data.objects if keypoint.type == 'MESH' and keypoint.name != 'MESH' and keypoint.parent != armature]

    # Create a map from each bone to the list of keypoints that are directly connected to it, later we will build an inverse of this map
    bone_to_joints_map = { bone.name: [] for bone in bones }

    # Create a map from each keypoint to the list of bones that are connected to it
    joint_to_bones_map = { keypoint_name: [] for keypoint_name in keypoint_names }

    # Now traverse the hierarchy of bones and map them to names of joints that are directly connected to them (including _tail variants, which might get discarded later)
    for bone in bones:
        # Get the list of children bones
        children_bones = [child_bone for child_bone in bone.children]

        # Add this bone and _tail variant to the map
        bone_to_joints_map[bone.name].append(bone.name)
        bone_to_joints_map[bone.name].append(f'{bone.name}_tail')

        # Get the list of children bones that are also keypoint names
        children_bones = [child_bone for child_bone in children_bones if child_bone.name in keypoint_names]

        # Add the children bones to the map
        bone_to_joints_map[bone.name].extend([child_bone.name for child_bone in children_bones])

    # Now build the inverse map
    for bone_name, joint_names in bone_to_joints_map.items():
        for joint_name in joint_names:
            if joint_name in joint_to_bones_map.keys():
                joint_to_bones_map[joint_name].append(bone_name)
            
    return joint_to_bones_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing the .blend files')
    parser.add_argument('--output-csv', type=str, required=True, help='Output CSV')
    args = parser.parse_args()

    measurements = {}

    # Computed threshold ratios
    threshold_to_distance_ratios = []

    def process_current_blend(joint_to_bones_map, blend_file):
        # We have the armature and spheres for each keypoint of the form BONE_NAME or BONE_NAME_TAIL (for the tail of the leaf bones)
        # Get the armature
        #armature = bpy.data.objects['Armature']
        armature = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]

        model_name = os.path.basename(blend_file).split('.')[0]

        dims = []
        meshes = [o for o in bpy.data.objects if o.type == 'MESH']
        for mesh in meshes:
            dims.append(mesh.dimensions[0])
            dims.append(mesh.dimensions[1])
            dims.append(mesh.dimensions[2])
        #dims = [mesh.dimensions[0], mesh.dimensions[1], mesh.dimensions[2]]
        scale = 1.0 / max(dims)
        # assert argmax dims is 2
        #assert dims.index(max(dims)) == 2

        vertices = [v.co for v in mesh.data.vertices]
        vertices = np.array(vertices) * scale

        objs = [o for o in bpy.data.objects if o.type == 'MESH' and o.name != 'MESH' and o.parent != armature]
        num_objs = len(objs)

        for obj in objs:
            obj.hide_viewport = True # prevent spheres from being hit by raycast

        for obj in objs:
            for threshold_multiplier in THRESHOLD_MULTIPLIERS:
                if measurements.get((obj.name, threshold_multiplier)) is None:
                    measurements[(obj.name, threshold_multiplier)] = 0
            if obj.name.endswith('_tail'):
                # Leaf bone
                bone_name = obj.name[:-len('_tail')]
            else:
                bone_name = obj.name
            
            # Get the bone
            bone = armature.pose.bones[bone_name]

            # Get the bone head coordinates
            if obj.name.endswith('_tail'):
                # Leaf bone
                joint = bone.tail
            else:
                joint = bone.head

            # Apply the armature transformation
            joint = armature.matrix_world @ joint

            # Get the list of bones that are connected to this joint
            bones = joint_to_bones_map[obj.name]

            # Raycast from the joint in directions perpendicular to the bones and find the closest hit points
            distances = []

            hit_found = False

            iterations_left = 10

            while not hit_found and iterations_left > 0:
                # Fire rays from groundtruth joint in directions perpendicular to the bones
                for bone_name in bones:
                    # Get the bone
                    bone = armature.pose.bones[bone_name]

                    # Apply armature transformation
                    bone_head = armature.matrix_world @ bone.head
                    bone_tail = armature.matrix_world @ bone.tail

                    # Get the vector from the bone head to the bone tail
                    bone_head_to_tail = bone_tail - bone_head
                    bone_head_to_tail_norm = bone_head_to_tail.normalized()

                    # Get the length of the bone
                    bone_length = bone_head_to_tail.length

                    # Get perpendicular vector to the bone
                    bone_head_to_tail_perp = bone_head_to_tail.orthogonal()

                    # Get the normalized perpendicular vector
                    bone_head_to_tail_perp_norm = bone_head_to_tail_perp.normalized()

                    # Now rotate the perpendicular vector in steps around the bone head to tail vector and fire rays

                    hit_missed = False
                    for angle in range(0, 360, 1):
                        rot_matrix = Matrix.Rotation(np.deg2rad(angle), 4, bone_head_to_tail_norm)
                        ray_origin = joint
                        ray_direction = bone_head_to_tail_perp_norm @ rot_matrix

                        # Fire the ray
                        # Can't do mesh.ray_cast if theres multiple meshes
                        # hit, location, normal, face_index = mesh.ray_cast(ray_origin, ray_direction)

                        hit, location, normal, face_index, object, matrix = bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, ray_origin, ray_direction)
    
                        if hit:
                            assert object.parent == armature
                            hit_found = True

                        if not hit and not hit_missed:
                            print(f'No hit for {obj.name} {bone_name} {angle}')
                            hit_missed = True
                            continue
                        
                        # Get the distance from the joint to the hit point
                        distance = (location - joint).length
                        distances.append(distance)
            
                if not hit_found:
                    assert len(distances) == 0
                    iterations_left -= 1
                    # Is this a tail joint
                    if obj.name.endswith('_tail'):
                        # Move the joint in the direction of the bone head to tail vector, back to the bone head
                        joint = joint - bone_head_to_tail_norm * 0.1
                    else:
                        # Move the joint in the direction of the bone head to tail vector, back to the bone tail
                        joint = joint + bone_head_to_tail_norm * 0.1
                        # NOTE This shouldn't happen.
                        # raise Exception(f'No hit found for {obj.name}')

            if not hit_found:
                raise Exception(f'No hit found for {obj.name}')

            # print(f'{obj.name} max {max(distances):.5f} min {min(distances):.5f} avg {sum(distances) / len(distances):.5f}')

            # Sort distances
            distances.sort()

            # Halve the amount of distances, so we only get the distances to the half of the closest hit points
            distances = distances[:len(distances) // 2]
            distances = np.array(distances) * scale

            # Compute the average distance
            average_distance = distances.mean()

            # Get the ground truth joint
            joint = joint * scale

            # Get the predicted joint
            predicted_joint = obj.location
            predicted_joint = predicted_joint * scale

            distance = (joint - predicted_joint).length

            threshold_to_distance_ratios.append((model_name, obj.name, distance / (average_distance)))
            
            for threshold_multiplier in THRESHOLD_MULTIPLIERS:
                threshold = average_distance * threshold_multiplier
                if distance <= threshold:
                    measurements[(obj.name, threshold_multiplier)] += 1
                    

    blend_files = list(pathlib.Path(args.input_dir).glob('*.blend'))

    for blend_file in blend_files:
        print(f'Processing {blend_file}')
        bpy.ops.wm.open_mainfile(filepath=str(blend_file))
        process_current_blend(joint_to_bones_map=create_joint_to_bones_map(), blend_file=blend_file)

    n = len(blend_files)

    for (joint_name, threshold_multiplier), hits in measurements.items():
        hits_percentage = hits / n
        measurements[(joint_name, threshold_multiplier)] = hits_percentage

    # Sort threshold-to-distance ratios in reverse order
    threshold_to_distance_ratios.sort(key=lambda x: x[2], reverse=True)

    # Write out the worst threshold-to-distance ratios
    worst_threshold_to_distance_ratios_path = args.output_csv.replace('.csv', '_worst_threshold_to_distance_ratios.txt')
    with open(worst_threshold_to_distance_ratios_path, 'w') as f:
        for model_name, joint_name, ratio in threshold_to_distance_ratios:
            f.write(f'{model_name} {joint_name} {ratio:.5f}\n')

    # Create new measurements array so that rows are joint names and columns are IoUS where the column names are the threshold multipliers
    measurements_new = {}
    for (joint_name, threshold_multiplier), hits in measurements.items():
        if measurements_new.get(joint_name) is None:
            measurements_new[joint_name] = {}
        measurements_new[joint_name][threshold_multiplier] = hits

    # df = pandas.DataFrame.from_dict(measurements_new, orient='index')
    # df = df.sort_index()
    # df.index.name = 'Joint'
    # df.to_csv(args.output_csv)

    # df = pandas.DataFrame.from_dict(measurements, orient='index', columns=['IoU'])
    # df = df.sort_index()
    # df.index.name = 'Joint'
    # df.to_csv(args.output_csv)


if __name__ == '__main__':
    main()