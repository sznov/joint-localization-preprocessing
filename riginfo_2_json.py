import argparse
import glob
import json
import os
import numpy as np
import bpy # TODO DEBUG

def process_riginfo(input_directory, output_directory):


    for input_riginfo_path in glob.glob(os.path.join(input_directory, '*.txt')):
        model_name = os.path.splitext(os.path.basename(input_riginfo_path))[0]
        output_json_path = os.path.join(output_directory, model_name + '.json')

        # load riginfo as lines
        with open(input_riginfo_path, 'r') as f:
            lines = f.readlines()
        
        # remove newline characters
        lines = [line.strip() for line in lines]

        # remove empty lines
        lines = [line for line in lines if line]

        # remove lines that don't start with joint
        lines = [line for line in lines if line.startswith('joint')]

        # remove joint prefix
        lines = [line.replace('joints', '') for line in lines]

        output_dict = {}

        output_dict['name'] = model_name
        output_dict['keypoints'] = []

        for line in lines:
            line = line.strip()
            line = line.split(' ')
            # line is of the form JointName X Y Z
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])
            output_dict['keypoints'].append([x, y, z])
            # DEBUG create sphere (bpy)
            # bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=(x, y, z))

        
        # write output dict to json
        os.makedirs(output_directory, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(output_dict, f)

        # DEBUG save blend file
        # output_blend_path = os.path.join(output_directory, model_name + '.blend')
        # bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert riginfo to json')
    parser.add_argument('--input-directory', type=str, default='input', help='Input directory containing riginfo files')
    parser.add_argument('--output-directory', type=str, default='output', help='Output directory')
    args = parser.parse_args()
    process_riginfo(args.input_directory, args.output_directory)