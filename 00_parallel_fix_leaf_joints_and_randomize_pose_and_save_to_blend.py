import argparse
import math
import os
import subprocess


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input-directory', type=str, help='Directory containing .fbx files')
    arg_parser.add_argument('--output-directory', type=str, help='Directory to save .blend files')
    arg_parser.add_argument('--pose-preprocess-params-json', type=str, help='Path to JSON file containing model preprocess and pose parameters')
    arg_parser.add_argument('--keypoint-names-file', type=str, help='File containing keypoint names')
    arg_parser.add_argument('--rotation-range-generator-path', type=str, help='Path to python file containing function to generate random rotation ranges')
    args = arg_parser.parse_args()

    print('Running 00_parallel_fix_leaf_joints_and_randomize_pose_and_save_to_blend.py')
    cpu_count = os.cpu_count()
    num_files = len([e for e in os.listdir(os.path.join(args.input_directory)) if e.lower().endswith('.fbx') or e.lower().endswith('.blend')])
    ps = []
    for i in range(cpu_count):
        chunk_size = math.ceil(num_files / cpu_count)
        start_index = i * chunk_size
        p = subprocess.Popen(['python', '00_fix_leaf_joints_and_randomize_pose_and_save_to_blend.py',
                        '--input-directory', args.input_directory,
                        '--output-directory', args.output_directory,
                        '--pose-preprocess-params-json', args.pose_preprocess_params_json,
                        '--keypoint-names-file', args.keypoint_names_file,
                        '--rotation-range-generator-path', args.rotation_range_generator_path,
                        '--chunk-size', str(chunk_size),
                        '--start-index', str(start_index)], shell=True)
        ps.append(p)

    for p in ps:
        p.wait()