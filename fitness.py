import torch

from operations import pad_all_to_max

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

def tensor_intersect1d(a, b):
    # https://discuss.pytorch.org/t/intersection-between-to-vectors-tensors/50364/9
    a_cat_b, counts = torch.cat([a, b]).unique(return_counts=True)
    intersection = a_cat_b[torch.where(counts.gt(1))]
    return intersection

def intersect_over_union_inv(predicted, expected_output):
    """ Return the inverted IoU between the predicted and the expected output. """
    if len(predicted.shape)<2:
       return 100
    shape = (max(predicted.shape[0], expected_output.shape[0]), max(predicted.shape[1], expected_output.shape[1]))
    diff = torch.zeros(shape, dtype=torch.uint8, device=device)
    diff[0:predicted.shape[0], 0:predicted.shape[1]] = (predicted > 0).type(torch.uint8) 
    diff[0:expected_output.shape[0], 0:expected_output.shape[1]] -= (expected_output > 0).type(torch.uint8) 
    
    intersection = (diff != 0).sum()
    union = (diff != -1).sum()
    return 1.0-(intersection / union)

def width_fitness(predicted, expected_output):
    """ How close the predicted image is to have the right width. Less is better."""
    if len(predicted.shape)<2:
       return 100
    return abs(predicted.shape[0] - expected_output.shape[0])

def height_fitness(predicted, expected_output):
    """ How close the predicted image is to have the right height. Less is better."""
    if len(predicted.shape)<2:
       return 100
    return abs(predicted.shape[1] - expected_output.shape[1])

def pixel_distance_fitness(p, e):
    """ How close the predicted image to have the right pixels. Less is better."""
    if len(p.shape)<2:
       return 100

    all = pad_all_to_max([p,e])
    if len(all)<2:
       return 100
    
    p,e = all[0].type(torch.uint8), all[1].type(torch.uint8)
    diff = (p != e).type(torch.uint8)
    return diff.sum()

def mse_fitness(p, e):
    """ Return the MSE between the predicted and the expected output. """
    if len(p.shape)<2:
       return 100
    all = pad_all_to_max([p,e])
    p,e = all[0].type(torch.uint8), all[1].type(torch.uint8)
    return ((p - e)**2).sum()

def activated_pixels_fitness(p, e):
    """ How close the predicted image to have the right pixels. Less is better."""
    if len(p.shape)<2:
       return 100
    shape = (max(p.shape[0], e.shape[0]), max(p.shape[1], e.shape[1]))
    diff = torch.zeros(shape, dtype=torch.uint8, device=device)
    diff[0:p.shape[0], 0:p.shape[1]] = (p > 0).type(torch.uint8) 
    diff[0:e.shape[0], 0:e.shape[1]] -= (e > 0).type(torch.uint8) 
    return (diff != 0).sum()

def colors_fitness(p, e):
    p_colors = torch.unique(p)
    e_colors = torch.unique(e)
    
    nb_inter = len(tensor_intersect1d(p_colors, e_colors))

    return (len(p_colors) - nb_inter) + (len(e_colors) - nb_inter)

fitness_functions = [
    colors_fitness,
    activated_pixels_fitness,
    height_fitness,
    width_fitness,
    pixel_distance_fitness,
    # mse_fitness,
    # intersect_over_union_inv
    ]


def product_less(a, b):
    """ Return True if the two tuples a and b respect a<b for the partial order. """
    for i in range(len(a)):
        if a[i] > b[i]:
            return False
    return True
    # a = torch.tensor(a).to(device)
    # b = torch.tensor(b).to(device)
    # return (a < b).all()

