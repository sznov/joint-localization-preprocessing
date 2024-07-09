import argparse
import math
import os
import subprocess

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input-directory', type=str, help='Directory containing .blend files')
    arg_parser.add_argument('--output-directory', type=str, help='Directory to output obj and obj_remesh directories')
    arg_parser.add_argument('--keypoint-names-file', type=str, help='File containing keypoint names')
    arg_parser.add_argument('--keypoint-extraction-params-json', type=str, default=None, help='Path to JSON file containing model preprocess and keypoint extraction parameters')
    args = arg_parser.parse_args()

    print('Running blend_2_obj_and_riginfo_normalized.py')
    cpu_count = os.cpu_count()
    num_files = len([e for e in os.listdir(os.path.join(args.input_directory)) if e.endswith('.blend')])
    ps = []
    for i in range(cpu_count):
        chunk_size = math.ceil(num_files / cpu_count)
        start_index = i * chunk_size
        p = subprocess.Popen(['python', '01_blend_2_obj_and_riginfo_normalized.py',
                        '--input-directory', args.input_directory,
                        '--output-directory', args.output_directory,
                        '--keypoint-names-file', args.keypoint_names_file,
                        '--keypoint-extraction-params-json', args.keypoint_extraction_params_json,
                        '--chunk-size', str(chunk_size),
                        '--start-index', str(start_index)], shell=True)
        ps.append(p)

    for p in ps:
        p.wait()