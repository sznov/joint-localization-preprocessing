import argparse
import glob
import json
import os

import bpy
import numpy as np
import open3d as o3d
import pymeshlab
from mathutils import Vector

SEED = 0
DEFAULT_TARGET_NUM_FACES = 20480

def main():
    parser = argparse.ArgumentParser(
        description='Convert FBX meshes to normalized OBJ meshes and Rig Info text files for RigNet (with normalized joint positions)')
    parser.add_argument('--input-directory', type=str,
                        default='input', help='Input directory')
    parser.add_argument('--output-directory', type=str,
                        default='output', help='Output directory')
    parser.add_argument('--keypoint-names-file', type=str, help='Path to keypoint names file')
    parser.add_argument('--seed', type=int, default=SEED, help='Seed for random number generator')
    parser.add_argument('--target-num-faces', default=DEFAULT_TARGET_NUM_FACES, type=int, help='Target number of faces for remeshing')
    parser.add_argument('--chunk-size', type=int, default=None, help='Number of meshes to process at a time')
    parser.add_argument('--start-index', type=int, default=None, help='Index of first mesh to process')
    parser.add_argument('--keypoint-extraction-params-json', type=str, default=None, help='Path to JSON file containing model preprocess and keypoint extraction parameters')
    args = parser.parse_args()
    process_meshes(args.input_directory, args.output_directory, args.keypoint_names_file, args.chunk_size, args.start_index, seed=args.seed, target_num_faces=args.target_num_faces, args=args)


def process_meshes(input_directory, output_directory, keypoint_names_file=None, chunk_size=None, start_index=None, args=None, **kwargs):
    # Create output directories if necessary
    output_obj_dir = os.path.join(output_directory, "obj")
    output_obj_remesh_dir = os.path.join(output_directory, "obj_remesh")
    output_normalization_transform_dir = os.path.join(output_directory, "normalization_transform")
    output_text_dir = os.path.join(output_directory, "rig_info")
    output_rig_info_visualization_dir = os.path.join(output_directory, "rig_info_visualization")
    for dir_path in [output_obj_dir, output_obj_remesh_dir, output_text_dir]:
        os.makedirs(dir_path, exist_ok=True)

    if chunk_size == None or start_index == None:
        blend_files = glob.glob(os.path.join(input_directory, '*.blend'))
    elif chunk_size != None and start_index != None:
        blend_files = glob.glob(        os.path.join(input_directory, '*.blend'))
        blend_files.sort()
        blend_files = blend_files[start_index:start_index+chunk_size]
    
    for input_mesh_path in blend_files:
        # Clear the file
        bpy.ops.wm.read_homefile(use_empty=True)

        output_obj_path = os.path.join(output_obj_dir, 
                                       os.path.splitext(os.path.basename(input_mesh_path))[0] + '.obj')
        output_obj_remesh_path = os.path.join(output_obj_remesh_dir, 
                                              os.path.splitext(os.path.basename(input_mesh_path))[0] + '_remesh.obj')
        output_text_path = os.path.join(output_text_dir,
                                        os.path.splitext(os.path.basename(input_mesh_path))[0] + '.txt')
        
        process_mesh(input_mesh_path, 
                     output_obj_path=output_obj_path, 
                     output_obj_remesh_path=output_obj_remesh_path,
                     output_normalization_transform_dir=output_normalization_transform_dir,
                     output_text_path=output_text_path,
                     output_rig_info_visualization_dir=output_rig_info_visualization_dir,
                     keypoint_names_file=keypoint_names_file,
                     args=args,
                     **kwargs)


def process_mesh(input_mesh_path, 
                 output_obj_path=None, 
                 output_obj_remesh_path=None, 
                 output_normalization_transform_dir=None,
                 output_text_path=None,
                 output_rig_info_visualization_dir=None,
                 keypoint_names_file=None,
                 args=None,
                 **kwargs):
    print(f"Processing {input_mesh_path}")
    translation_normalize, scale_normalize = save_obj(input_mesh_path, output_obj_path, output_obj_remesh_path, output_normalization_transform_dir, **kwargs)
    save_riginfo(input_mesh_path, 
                 output_text_path, 
                 translation_normalize=translation_normalize, 
                 scale_normalize=scale_normalize,
                 remeshed_obj_path=output_obj_remesh_path,
                 keypoint_names_file=keypoint_names_file,
                 rig_info_visualization_dir=output_rig_info_visualization_dir,
                 args=args)
    

def save_obj(input_mesh_path, output_obj_path=None, output_obj_remesh_path=None, output_normalization_transform_dir=None, **kwargs):
    # Set output path if necessary
    if output_obj_path is None:
        output_obj_path = os.path.splitext(input_mesh_path)[0] + '.obj'

    if output_obj_remesh_path is None:
        output_obj_remesh_path = os.path.splitext(input_mesh_path)[0] + '_remesh.obj'

    # Create output directory if necessary
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)

    # -- Export the mesh from Blender --

    # Clear the scene
    bpy.ops.wm.read_homefile(use_empty=True)
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Load the mesh
    if input_mesh_path.endswith('.blend'):  
        bpy.ops.wm.open_mainfile(filepath=input_mesh_path)
    elif input_mesh_path.endswith('.fbx'):
        bpy.ops.import_scene.fbx(filepath=input_mesh_path)

    # Create output directory if necessary
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)

    # Save the mesh
    bpy.ops.wm.obj_export(filepath=output_obj_path, path_mode='COPY', export_triangulated_mesh=True)

    # --Remesh the mesh with PyMeshLab quadric edge collapse --

    np.random.seed(kwargs.get('seed') or SEED)

    ms = pymeshlab.MeshSet()

    # Loads either a mesh or a scene
    ms.load_new_mesh(output_obj_path)
    
    # Extract all meshes from the loaded scene if necessary
    ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                    targetfacenum=kwargs.get('target_num_faces') or DEFAULT_TARGET_NUM_FACES)

    # Create output directory if necessary
    os.makedirs(os.path.dirname(output_obj_remesh_path), exist_ok=True)

    # Save the simplified mesh
    ms.save_current_mesh(output_obj_remesh_path, save_face_color=False, save_wedge_texcoord=False)

    # -- Normalize the meshes --
    
    # Normalize the mesh
    mesh = o3d.io.read_triangle_mesh(output_obj_path)
    mesh.compute_vertex_normals()
    mesh_v = np.asarray(mesh.vertices)
    mesh_f = np.asarray(mesh.triangles)
    mesh_v, translation_normalize, scale_normalize = normalize_obj(mesh_v)
    mesh_normalized = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh_v), 
                                                triangles=o3d.utility.Vector3iVector(mesh_f))
    o3d.io.write_triangle_mesh(output_obj_path, mesh_normalized)

    # Normalize the remeshed mesh
    mesh = o3d.io.read_triangle_mesh(output_obj_remesh_path)
    mesh.compute_vertex_normals()
    mesh_v = np.asarray(mesh.vertices)
    mesh_f = np.asarray(mesh.triangles)
    mesh_v, translation_normalize, scale_normalize = normalize_obj(mesh_v)
    mesh_normalized = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh_v), 
                                                triangles=o3d.utility.Vector3iVector(mesh_f))
    o3d.io.write_triangle_mesh(output_obj_remesh_path, mesh_normalized)

    # -- Resave both with Blender --
    
    # Clear the file
    bpy.ops.wm.read_homefile()
    # Clear the scene
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.wm.obj_import(filepath=output_obj_path)
    bpy.ops.wm.obj_export(filepath=output_obj_path, path_mode='COPY', export_triangulated_mesh=True)

    # Clear the file
    bpy.ops.wm.read_homefile()
    # Clear the scene
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.wm.obj_import(filepath=output_obj_remesh_path)
    bpy.ops.wm.obj_export(filepath=output_obj_remesh_path, path_mode='COPY', export_triangulated_mesh=True)

    # Clear the file
    bpy.ops.wm.read_homefile()    
    # Clear the scene
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # -- Resave meshes with MeshLab --

    # This is necessary because Blender saves extra information in the OBJ files 
    # that later in the pipeline gets interpreted as vertices.
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(output_obj_path)
    ms.save_current_mesh(output_obj_path, save_face_color=False, save_wedge_texcoord=False)
    ms.load_new_mesh(output_obj_remesh_path)
    ms.save_current_mesh(output_obj_remesh_path, save_face_color=False, save_wedge_texcoord=False)

    # save translation_normalize and scale_normalize to json file
    transform = {'translation_normalize': list(translation_normalize), 'scale_normalize': scale_normalize}
    transform_path = os.path.join(output_normalization_transform_dir, 
                                  os.path.splitext(os.path.basename(input_mesh_path))[0] + '_transform.json')
    
    os.makedirs(output_normalization_transform_dir, exist_ok=True)
    with open(transform_path, 'w') as f:
        json.dump(transform, f)

    return translation_normalize, scale_normalize


def traverse_bones(root_bone, keypoint_names, armature, armature_scale=1, translation_normalize=None, scale_normalize=None):
    keypoints = []
    hier = []

    def should_skip_keypoint(keypoint_name):
        return keypoint_names is not None and keypoint_name not in keypoint_names

    if not should_skip_keypoint(root_bone.name):
        bone_head = armature.matrix_world @ root_bone.head
        bone_head = np.array(tuple(bone_head))
        if translation_normalize is not None:
            bone_head = bone_head - translation_normalize
        if scale_normalize is not None: 
            bone_head = bone_head * scale_normalize
        keypoints.append((f'{root_bone.name}', tuple(bone_head)))

    any_childbone_has_nonzero_head = any(not np.allclose(tuple(child_bone.head), 
                                                         tuple(Vector((0, 0, 0))), 
                                                         atol=1e-5) 
                                         for child_bone in root_bone.children)

    if len(root_bone.children) == 0 or any_childbone_has_nonzero_head:
        if not should_skip_keypoint(f'{root_bone.name}_tail'):
            bone_tail = armature.matrix_world @ root_bone.tail
            bone_tail = np.array(tuple(bone_tail))
            if translation_normalize is not None:
                bone_tail = bone_tail - translation_normalize
            if scale_normalize is not None:
                bone_tail = bone_tail * scale_normalize
            keypoints.append((f'{root_bone.name}_tail', tuple(bone_tail)))
            hier.append((f'{root_bone.name}', f'{root_bone.name}_tail'))

    for child_bone in [bone for bone in root_bone.children if not should_skip_keypoint(bone.name)]:
        hier.append((f'{root_bone.name}', f'{child_bone.name}'))
        child_keypoints, child_hier = traverse_bones(child_bone, keypoint_names, armature, armature_scale=armature_scale, translation_normalize=translation_normalize, scale_normalize=scale_normalize)
        keypoints.extend(child_keypoints)
        hier.extend(child_hier)
    
    return keypoints, hier


def save_riginfo(input_mesh_path, 
                 output_text_path=None, 
                 translation_normalize=None, 
                 scale_normalize=None,
                 remeshed_obj_path=None,
                 keypoint_names_file=None,
                 skip_keypoints=None,
                 rig_info_visualization_dir=None,
                 args=None):
    
    if args is not None and args.keypoint_extraction_params_json is not None:
        import json
        with open(args.keypoint_extraction_params_json, 'r') as f:
            keypoint_extraction_params = json.load(f)
    else:
        keypoint_extraction_params = {
            'skip_keypoints': None,
            'root_override': None,
        }
    
    keypoint_names = None
    if keypoint_names_file is not None:
        with open(keypoint_names_file, 'r') as f:
            keypoint_names = f.read().splitlines()
            keypoint_names = set(keypoint_names)

    # Set output path if necessary
    if output_text_path is None:
        output_text_path = os.path.splitext(input_mesh_path)[0] + '.txt'

    # Create output directory if necessary
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)

    # Clear the scene
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Load the mesh from .blend
    bpy.ops.wm.open_mainfile(filepath=input_mesh_path)

    # Create output directory if necessary
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
    
    # Select armature
    armature = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE'][0]

    # Get root bone as the bone with no parent
    root_bone = [bone for bone in armature.pose.bones if bone.parent is None][0]
    root = keypoint_extraction_params['root_override'] or f'{root_bone.name}'
    
    # Traverse bones beginning at root bone and compute keypoints and hierarchy
    # HACK: This is a hack to adjust for the coordinate system.
    translation_normalize_perm = np.empty(3)
    translation_normalize_perm[0] = translation_normalize[0]
    translation_normalize_perm[1] = -translation_normalize[2]
    translation_normalize_perm[2] = translation_normalize[1]
    keypoints, hier = traverse_bones(root_bone, keypoint_names, armature, armature_scale=armature.scale, translation_normalize=translation_normalize_perm, scale_normalize=scale_normalize)

    output_lines = []

    skip_keypoints = keypoint_extraction_params['skip_keypoints'] or []

    for keypoint in keypoints:
        keypoint_name = keypoint[0]
        # HACK: This is a hack to adjust for the coordinate system.
        keypoint_position = np.empty_like(keypoint[1])
        keypoint_position[0] = keypoint[1][0]
        keypoint_position[1] = keypoint[1][2]
        keypoint_position[2] = -keypoint[1][1]
        if keypoint_name not in skip_keypoints:
            output_lines.append(f'joints {keypoint_name} {keypoint_position[0]} {keypoint_position[1]} {keypoint_position[2]}')
    
    output_lines.append(f'root {root}')
    for h in hier:
        if h[0] not in skip_keypoints and h[1] not in skip_keypoints:
            output_lines.append(f'hier {h[0]} {h[1]}')

    # Create output directory if necessary
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)

    with open(output_text_path, 'w') as f:
        f.write('\n'.join(output_lines))

    if remeshed_obj_path is None:
        return

    # Clear the file
    bpy.ops.wm.read_homefile()    
    # Clear the scene
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Import remeshed mesh
    bpy.ops.wm.obj_import(filepath=remeshed_obj_path)

    # For each keypoint, add a sphere at the keypoint location and name it
    for keypoint in keypoints:
        if keypoint[0] not in skip_keypoints:
            bpy.ops.mesh.primitive_uv_sphere_add(location=keypoint[1], radius=0.01)
            bpy.context.object.name = "keypoint {}".format(keypoint[0])

    model_name = os.path.splitext(os.path.basename(input_mesh_path))[0]

    os.makedirs(rig_info_visualization_dir, exist_ok=True)

    bpy.ops.wm.save_as_mainfile(
        filepath=os.path.abspath(os.path.join(rig_info_visualization_dir, model_name + '.blend')))


# Sourced from quick_start.py in official RigNet implementation repo. Modified to put center on origin.
def normalize_obj(mesh_v):
    dims = [max(mesh_v[:, 0]) - min(mesh_v[:, 0]),
            max(mesh_v[:, 1]) - min(mesh_v[:, 1]),
            max(mesh_v[:, 2]) - min(mesh_v[:, 2])]
    scale = 1.0 / max(dims)
    pivot = np.array([(min(mesh_v[:, 0]) + max(mesh_v[:, 0])) / 2, 
                      (min(mesh_v[:, 1]) + max(mesh_v[:, 1])) / 2,
                      (min(mesh_v[:, 2]) + max(mesh_v[:, 2])) / 2])
    mesh_v[:, 0] -= pivot[0]
    mesh_v[:, 1] -= pivot[1]
    mesh_v[:, 2] -= pivot[2]
    mesh_v *= scale
    return mesh_v, pivot, scale


if __name__ == "__main__":
    main()