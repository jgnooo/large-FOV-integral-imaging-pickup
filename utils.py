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
    # if color height != 700 and color width != 1200, Resize the input image.
    img = Image.open(path).resize((1200, 700))
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


def divide_image(image):
    """Divide image for hierarchiclar integral imaging pickup system.

    Args:
        image : Numpy array. (image)
    Returns:
        image_list : Partitioned image list.
    """
    width_list = [i * 200 for i in range(4)]
    height_list = [0, 100]

    image_list = []
    for h in height_list:
        for w in width_list:
            image_list.append(image[h:h + 600, w:w + 600])
    return image_list


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