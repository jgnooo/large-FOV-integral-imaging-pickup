import os
import argparse
import numpy as np
import tensorflow as tf
import multiprocessing

from PIL import Image
from functools import partial

import monodepth.depth_estimator as estimator
import InIsystem.convert as cvt
import InIsystem.pickup as pickup
import InIsystem.subaperture as sub

import utils


parser = argparse.ArgumentParser(description='Large field-of-view integral imaging pickup system.')
parser.add_argument('--color_path', type=str,
                    default='./inputs/test.jpg', help='Path of input image.')
parser.add_argument('--output_path', type=str,
                    default='./results/', help='Output root directory.')

parser.add_argument('--model_path', type=str,
                    default='./monodepth/model.h5', help='Model file for predicting a depth.')

parser.add_argument('--is_gpu', action='store_true',
                    help='Select GPU or Not.')

args = parser.parse_args()


def divide_image(image):
    """

    Args:
        
    Returns:
        
    """
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
    inputs['num_of_lenses'] = 400
    inputs['P_D'] = 1
    inputs['P_L'] = 15
    inputs['f'] = 10
    inputs['g'] = 11
    return inputs


def merge_sub_apertures(outputs, num_of_lenses):
    """

    Args:
        
    Returns:
        
    """
    print('Generate large FOV sub-aperture image array...')
    LFOV_sub_apertures = np.zeros((467 * 7, 805 * 7, 3))
    for i in range(7):
        for j in range(7):
            i_start = i * num_of_lenses
            i_end = i * num_of_lenses + num_of_lenses
            j_start = j * num_of_lenses
            j_end = j * num_of_lenses + num_of_lenses

            rows = np.hstack([
                outputs[0][i_start:i_end, j_start:j_end],
                outputs[1][i_start:i_end, j_start + 265:j_end],
                outputs[2][i_start:i_end, j_start + 265:j_end],
                outputs[3][i_start:i_end, j_start + 265:j_end]
            ])

            cols = np.hstack([
                outputs[4][i_start:i_end, j_start:j_end],
                outputs[5][i_start:i_end, j_start + 265:j_end],
                outputs[6][i_start:i_end, j_start + 265:j_end],
                outputs[7][i_start:i_end, j_start + 265:j_end]
            ])

            lfov = np.vstack([
                rows,
                cols[333:, :]
            ])
            LFOV_sub_apertures[i * 467:i * 467 + 467, j * 805:j * 805 + 805] = lfov
    return LFOV_sub_apertures


def multiprocess_ini(inputs, data):
    """

    Args:
        
    Returns:
        
    """
    image = data[0]
    depth = data[1]
    
    d, P_I, delta_d, L = cvt.convert_depth(depth, inputs['f'], inputs['g'],
                                        inputs['P_D'], inputs['P_L'])

    EIA = pickup.generate_elemental_imgs_GPU(image, L, inputs['P_L'],
                                             P_I, inputs['g'], inputs['num_of_lenses'])

    sub_apertures = sub.generate_sub_apertures(EIA, inputs['P_L'], inputs['num_of_lenses'])
    sub_apertures = sub_apertures.astype(np.uint8)
    sub_apertures = sub_apertures[7 * 400:, 7 * 400:]
    return sub_apertures


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
    image = utils.load_image(args.color_path)
    depth = get_depth_map(image, args.model_path)

    # Divide image / depth for hierarchical integral imaging pickup system.
    image_list = divide_image(image)
    depth_list = divide_image(depth)
    data_list = list(zip(image_list, depth_list))
    
    # Hierarchical integral imaging pickup system using multi-processing
    print('Integral imaging pickup system...')
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=4)
    func = partial(multiprocess_ini, inputs)
    outputs = pool.map(func, data_list)
    pool.close()
    pool.join()

    # Merge large FOV sub-aperture images
    large_sub_apertures = merge_sub_apertures(outputs, inputs['num_of_lenses'])
    Image.fromarray(large_sub_apertures.astype(np.uint8)).save(output_dir + '/large_FOV_sub_apertures.jpg')


if __name__ == "__main__":
    ##
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
        except RuntimeError as e:
            print(e)

    main()