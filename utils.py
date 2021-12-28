import numpy as np

from PIL import Image


def load_image(path):
    """Load image data.

    Args:
        path : Image path.
    Returns:
        img  : Numpy array. (image)
    """
    # In this system, the input image need to have resolution (480, 854).
    # if color height != 480 and color width != 854, Resize the input image.
    img = Image.open(path).resize((854, 480))
    img = np.asarray(img)
    return img


def save_image(img, path):
    """Save image.

    Args:
        img  : Numpy array. (image)
        path : Path for saving image.
    """
    img = Image.fromarray(img.astype(np.uint8))
    img.save(path)


def visualize_depth(depth):
    """Visualize raw depth image.

    Args:
        depth        : depth map.
    Returns:
        visual_depth : Visualized depth map. (0 ~ 255)
    """
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    visual_depth = np.uint8(depth * 255.0)
    return visual_depth


def generate_coords(height, width, depth=None, is_depth=False):
    """Generate coordinates of color image.

    Args:
        height   : range of y coordinate.
        width    : range of x coordinate.
        depth    : depth map. (z coordinate)
        is_depth : flag whether to not 3D coordinates.
    Returns
        coords   : 
            is_depth is Ture  : x, y, depth 3D coordinates.
            is_depth is False : x, y 2D coordinates.
    """
    x = np.linspace(-int(width / 2), int(width / 2) - 1, width).astype(np.float32)
    y = np.linspace(-int(height / 2), int(height / 2) - 1, height).astype(np.float32)
    x, y = np.meshgrid(x, y)
    
    if is_depth:
        coords = np.stack([x, y, depth], axis=0)
    else:
        coords = np.stack([x, y], axis=0)
    return coords