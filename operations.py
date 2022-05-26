from functools import partial
import itertools
import torch
from typing import List


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

# from: https://www.kaggle.com/code/zenol42/dsl-and-genetic-algorithm-applied-to-arc/notebook

def torch_bitwise_or_reduce(x):
    """ Compute the pixelwise or of all images in the list. """
    r = torch.tensor(0).to(device)
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

def swap_color_unlifted(pixmap, color_in, color_out):
    """_summary_
    Swap the color of an image.
    """
    # Replace the color of the image
    indices = (torch.round(pixmap) == color_in)
    pixmap[indices] = color_out
    return [pixmap]


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



all_operations = [
    identity,
    crop_to_content,
    group_by_color,
    split_h,
    negative,
    color_shift,
    tail, init, union, intersect,
    sortByColor, sortByWeight, reverse,
]

# operations with parameters:

# swap color:
for in_out in itertools.combinations(range(10), 2):
    func = partial(swap_color, color_in=in_out[0], color_out=in_out[1])
    func.__name__ = f"swap_color:{in_out[0]}->{in_out[1]}"
    all_operations.append(func)        