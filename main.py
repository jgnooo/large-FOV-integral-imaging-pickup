import os
import argparse
import numpy as np
import multiprocessing

from PIL import Image
from functools import partial

import monodepth.depth_estimator as estimator

import utils


parser = argparse.ArgumentParser(description='Large field-of-view integral imaging pickup system.')
parser.add_argument('--color_path', type=str,
                    default='./inputs/183516667.jpg', help='Path of input image.')
parser.add_argument('--output_path', type=str,
                    default='./results/', help='Output root directory.')

parser.add_argument('--model_path', type=str,
                    default='./monodepth/model.h5', help='Model file for predicting a depth.')

parser.add_argument('--is_gpu', action='store_true',
                    help='Select GPU or Not.')

parser.add_argument('--num_of_lenses', type=int,
                    default=200, help='Number of elemental lenses.')
parser.add_argument('--P_D', type=float,
                    default=1, help='Pixel pitch of LCD.')
parser.add_argument('--P_L', type=int,
                    default=15, help='Size of elemental lens.')
parser.add_argument('--f', type=float,
                    default=10, help='Focal length of elemental lens.')
parser.add_argument('--g', type=float,
                    default=11, help='Gap between lens and display.')

args = parser.parse_args()


def divide_image(image):
    width_list = [i * 200 for i in range(4)]
    height_list = [0, 100]

    image_list = []
    for h in height_list:
        for w in width_list:
            image_list.append(image[h:h + 600, w:w + 600])
    return image_list


def get_depth_map(color, model_path):
    """Predict a depth map from a single RGB image.

    Args:
        color : Input color image.
    Returns:
        depth : Predicted a depth image corresponding a input RGB image.
    """
    height, width, _ = color.shape

    # if color height != 480 and color width != 640, Resize the input image.
    if height != 480 and width != 640:
        color = estimator.resize_image(color)
    
    net_input = estimator.preprocess_image(color)
    depth = estimator.estimate_depth(net_input, height, width, model_path)
    return depth


def get_lens_params():
    """Lens Parameters

        Information of Lens-array
            - P_L           : Size of elemental lens.
            - num_of_lenses : Number of elemental lens.
            - f             : Focal length of elemental lens.

        Information of Display
            - P_D           : Pixel pitch of LCD.
            - g             : Gap between lens and display.
    """    
    inputs = {}
    inputs['num_of_lenses'] = args.num_of_lenses
    inputs['P_D'] = args.P_D
    inputs['P_L'] = args.P_L
    inputs['f'] = args.f
    inputs['g'] = args.g
    return inputs


def multiprocess_ini():

    pass


def main():
    # Set experiment name.
    experiment_name = args.color_path.split('/')[-1].split('.')[0]

    # Make directory for saving results.
    output_dir = os.path.join(
        args.output_path,
        experiment_name
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Setup the micro lens parameters.
    inputs = get_lens_params()

    # Load input RGB image and predict a depth image.
    inputs['color'] = utils.load_image(args.color_path)
    inputs['depth'] = get_depth_map(inputs['color'], args.model_path)

    # Divide image / depth for hierarchical integral imaging pickup system.
    color_list = divide_image(inputs['color'])
    depth_list = divide_image(inputs['depth'])
    data_list = zip(color_list, depth_list)
    
    # Hierarchical integral imaging pickup system using multi-processing
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=8)
    func = partial(multiprocess_ini, 'RGB')
    pool.map(func, data_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()