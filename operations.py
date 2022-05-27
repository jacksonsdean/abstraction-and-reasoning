from functools import partial
import itertools
import torch
from typing import List


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

# from: https://www.kaggle.com/code/zenol42/dsl-and-genetic-algorithm-applied-to-arc/notebook

def torch_bitwise_or_reduce(x):
    """ Compute the pixelwise or of all images in the list. """
    r = torch.tensor(0).to(device).type(torch.uint8)
    x = x.type(torch.uint8)
    for i in range(len(x)):
        r = torch.bitwise_or(r, x[i])
    return r.type(torch.bool)


def identity(x):
    return x

def group_by_color_unlifted(pixmap):
    """ Split an image into a collection of images with unique color """
    # Count the number of colors
    nb_colors = int(pixmap.max()) + 1
    # Create a pixmap for each color
    split_by_color = [(pixmap == i) * i for i in range(1, nb_colors)]
    # Filter out empty images
    return [x for x in split_by_color if torch.any(x)]

def crop_to_content_unlifted(pixmap):
    """ Crop an image to fit exactly the non 0 pixels """
    # Op argwhere will give us the coordinates of every non-zero point
    true_points = torch.argwhere(pixmap)
    if len(true_points) == 0:
        return []
    # Take the smallest points and use them as the top left of our crop
    top_left = true_points.min(axis=0)[0]
    # Take the largest points and use them as the bottom right of our crop
    bottom_right = true_points.max(axis=0)[0]
    # Crop inside the defined rectangle
    pixmap = pixmap[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    return [pixmap]

def split_h_unlifted(pixmap):
    """ Split horizontally an image """
    h = pixmap.shape[0]
    if h % 2 == 1:
        h = h // 2
        return [pixmap[:h,:], pixmap[h+1:,:]]
    else:
        h = h // 2
        return [pixmap[:h,:], pixmap[h:,:]]

def negative_unlifted(pixmap):
    """ Compute the negative of an image (and conserve the color) """
    if pixmap.numel() == 0:
        return []
    negative = torch.logical_not(pixmap)
    color = max(pixmap.max(), 1)
    return [negative * color] 



# Added by Jackson -------------------------------------------------------------
def color_shift_unlifted(pixmap, shift=1, shift_background=False):
    """ Shift the color of an image by a random amount. """
    max_color = 9
    if not shift_background:
        pixmap[pixmap==0] = -1
    # Shift the color of the image
    pixmap = [(pixmap + shift) % max_color]
    return pixmap

def swap_color_unlifted(pixmap, color_a, color_b):
    """_summary_
    Swap the color of an image.
    """
    # Replace the color of the image
    indices_a = (pixmap == color_a)
    pixmap[indices_a] = color_b
    return [pixmap]

def rot_90_unlifted(pixmap):
    """ Rotate an image 90 degrees clockwise. """
    return [torch.rot90(pixmap, 1, [0, 1])]

def flip_h_unlifted(pixmap):
    """ Flip an image horizontally. """
    return [torch.flip(pixmap, dims=[1])]

def flip_v_unlifted(pixmap):
    """ Flip an image vertically. """
    return [torch.flip(pixmap, dims=[0])]

def tile_2_unlifted(pixmap):
    """ Tile an image twice. """
    return [torch.tile(pixmap, (2, 2))]

def split_v_unlifted(pixmap):
    """ Split an image vertically """
    w = pixmap.shape[1]
    if w % 2 == 1:
        w = w // 2
        return [pixmap[:,:w], pixmap[:,w+1:]]
    else:
        w = w // 2
        return [pixmap[:,:w], pixmap[:,w:]]

def reciprocal_unlifted(x):
    """ Compute the reciprocal of an image. """
    return [torch.reciprocal(x)]

def argmax_unlifted(x):
    """ Compute the argmax of an image. """
    return [torch.argmax(x).unsqueeze(0)]

def unique_unlifted(x):
    """ Compute the unique pixels of all images in the list. """
    return [torch.unique(x)]

def tril_unlifted(x):
    """ Compute the lower triangular part of an image. """
    return [torch.tril(x)]

def triu_unlifted(x):
    """ Compute the upper triangular part of an image. """
    return [torch.triu(x)]

def pad_one_col_unlifted(x):
    """ Pad an image with a single pixel in the left column. """
    if x.numel() == 0:
        return []
    return[torch.nn.functional.pad(x, (1, 0, 0, 0))]

def pad_one_row_unlifted(x):
    """ Pad an image with a single pixel in the top row. """
    if x.numel() == 0:
        return []
    return[torch.nn.functional.pad(x, (0, 0, 1, 0))]

def delete_one_row_unlifted(x):
    """ Delete a single pixel in the top row of an image. """
    if x.numel() == 0:
        return []
    return [x[:, 1:]]
def delete_one_col_unlifted(x):
    """ Delete a single pixel in the left column of an image. """
    if x.numel() == 0:
        return []
    return [x[:, :-1]]

def transpose_unlifted(x):
    """ Transpose an image. """
    return [x.transpose(0, 1)]

def duplicate_unlifted(x):
    """ Duplicate an image. """
    return [x, x]

#------------------------------------------------------------------------------
#%%
# multiple image operations
def tail(x):
    if len(x) > 1:
        return x[1:]
    else:
        return x

def init(x):
    if len(x) > 1:
        return x[:1]
    else:
        return x

def union(x):
    """ Compute the pixel union of all images in the list. """
    if len(x) < 2:
        return x
    # Make all have the same shape
    first_shape = tuple(x[0].shape)
    for pixmap in x[1:]:
        if first_shape != tuple(pixmap.shape):
            return []

    x = torch.stack(x)            
    return [torch_bitwise_or_reduce(x)]
    
def intersect(x):
    """ Compute the pixel intersection of all images in the list. """
    if len(x) < 2:
        return x
    
    # Make sure all have the same shape
    first_shape = tuple(x[0].shape)
    for pixmap in x[1:]:
        if first_shape != tuple(pixmap.shape):
            return []

    x = torch.stack(x)            
    return [(torch.prod(x, axis=0))]
    
def sortByColor(xs):
    """ Sort pictures by increasing color id. """
    xs = [x for x in xs if len(x.reshape(-1)) > 0]
    return list(sorted(xs, key=lambda x: x.max()))

def sortByWeight(xs):
    """ Sort images by how many non zero pixels are contained. """
    xs = [x for x in xs if len(x.reshape(-1)) > 0]
    return list(sorted(xs, key=lambda x: (x>0).sum()))

def reverse(x):
    """ Reverse the order of a list of images. """
    return x[::-1]


# added by jackson -------------------------------------------------------------
def stack_col(x):
    """ Stack a list of images into one tensor by column. """
    # Make sure all have the same shape
    first_shape = tuple(x[0].shape)
    for pixmap in x[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    return [torch.column_stack(x)]

def stack_row(x):
    """ Stack a list of images into one tensor by row. """
    # Make sure all have the same shape
    first_shape = tuple(x[0].shape)
    for pixmap in x[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    return [torch.row_stack(x)]

def pad_all_to_max(xs):
    """ Pad all images in the list to the maximum size. """
    max_shape = tuple(max(x.shape for x in xs))
    return [torch.nn.functional.pad(x, (0, max_shape[1] - x.shape[1], 0, max_shape[0] - x.shape[0])) for x in xs]

def crop_all_to_min(xs):
    """ Crop all images in the list to the minimum size. """
    min_shape = tuple(min(x.shape for x in xs))
    return [x[:min_shape[0], :min_shape[1]] for x in xs]

def elementwise_sum(xs):
    """ Sum all images in the list."""
    # Make sure all have the same shape
    first_shape = tuple(xs[0].shape)
    for pixmap in xs[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    return [torch.add(torch.tensor(xs))]

def elementwise_sub(xs):
    """ Subtract all images in the list."""
    # Make sure all have the same shape
    first_shape = tuple(xs[0].shape)
    for pixmap in xs[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    return [torch.sub(torch.tensor(xs))]

def elementwise_max(xs):
    # Make sure all have the same shape
    first_shape = tuple(xs[0].shape)
    for pixmap in xs[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    return [torch.max(torch.tensor(xs))]

def elementwise_min(xs):
    # Make sure all have the same shape
    first_shape = tuple(xs[0].shape)
    for pixmap in xs[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    return [torch.min(torch.tensor(xs))]

def elementwise_mul(xs):
    # Make sure all have the same shape
    first_shape = tuple(xs[0].shape)
    for pixmap in xs[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    return [torch.mul(torch.tensor(xs))]

def elementwise_div(xs):
    # Make sure all have the same shape
    first_shape = tuple(xs[0].shape)
    for pixmap in xs[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    return [torch.div(torch.tensor(xs))]

def elementwise_gcd(xs):
    # Make sure all have the same shape
    first_shape = tuple(xs[0].shape)
    for pixmap in xs[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    return [torch.gcd(torch.tensor(xs))]

def elementwise_lcm(xs):
    # Make sure all have the same shape
    first_shape = tuple(xs[0].shape)
    for pixmap in xs[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    return [torch.lcm(torch.tensor(xs))]

def elementwise_lstsq(xs):
    # Make sure all have the same shape
    first_shape = tuple(xs[0].shape)
    for pixmap in xs[1:]:
        if first_shape != tuple(pixmap.shape):
            return []
    res = xs[0]
    index = 0
    while index < len(xs):
        res = torch.lstsq(res, torch.tensor(xs[index+1]))[0]
        index += 1
    return [res]

#%% 
def lift(fct):
    # Lift the function
    def lifted_function(xs, **kwargs):
        list_of_results = [fct(x, **kwargs) for x in xs]
        return list(itertools.chain(*list_of_results))
    # Give a nice name to the lifted function
    import re
    lifted_function.__name__ = re.sub('_unlifted$', '_lifted', fct.__name__)
    return lifted_function

crop_to_content = lift(crop_to_content_unlifted)
group_by_color = lift(group_by_color_unlifted)
split_h = lift(split_h_unlifted)
negative = lift(negative_unlifted)
color_shift = lift(color_shift_unlifted)
swap_color = lift(swap_color_unlifted)
rot_90 = lift(rot_90_unlifted)
flip_h = lift(flip_h_unlifted)
flip_v = lift(flip_v_unlifted)
tile_2 = lift(tile_2_unlifted)
split_v = lift(split_v_unlifted)
reciprocal = lift(reciprocal_unlifted)
argmax = lift(argmax_unlifted)
unique = lift(unique_unlifted)
tril = lift(tril_unlifted)
triu = lift(triu_unlifted)
pad_one_col = lift(pad_one_col_unlifted)
pad_one_row = lift(pad_one_row_unlifted)
delete_col = lift(delete_one_col_unlifted)
delete_row = lift(delete_one_row_unlifted)
transpose = lift(transpose_unlifted)
duplicate = lift(duplicate_unlifted)

all_operations = [
    identity,
    crop_to_content,
    group_by_color,
    split_h,
    negative,
    tail, init, union, intersect,
    sortByColor, sortByWeight, reverse,
    
    # Jackson:
    color_shift,
    rot_90,
    flip_h,
    flip_v,
    split_v,
    reciprocal,
    pad_one_col,
    pad_one_row,
    delete_col,
    delete_row,
    transpose,
    duplicate,
    # tile_2,
    # argmax,
    # unique,
    # tril, triu,# BROKE
    pad_all_to_max, crop_all_to_min,
    # stack_h, 
    stack_col, stack_row
    # BROKE:
    # elementwise_sum,
    # elementwise_sub,
    # elementwise_max,
    # elementwise_min,
    # elementwise_mul,
    # elementwise_div,
    # elementwise_gcd,
    # elementwise_lcm,
    # elementwise_lstsq,
]
# operations with parameters:

# swap color:
for in_out in itertools.permutations(range(10), 2):
    func = partial(swap_color, color_a=in_out[0], color_b=in_out[1])
    func.__name__ = f"swap_color:{in_out[0]}->{in_out[1]}"
    all_operations.append(func)        