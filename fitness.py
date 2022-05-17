from typing import List
import torch
def evaluate(program: List[callable], input_image: torch.tensor):
    if not isinstance(input_image, torch.Tensor):
        input_image = torch.tensor(input_image)
    assert type(input_image) == torch.Tensor
    
    # Apply each function on the image
    image_list = [input_image]
    for fct in program:
        # Apply the function
        image_list = fct(image_list)
        # Filter out empty images
        image_list = [img for img in image_list if img.shape[0] > 0 and img.shape[1] > 0]
        # Break if there is no data
        if image_list == []:
            return []
    return image_list 
     

def tensor_intersect1d(a, b):
    # https://discuss.pytorch.org/t/intersection-between-to-vectors-tensors/50364/9
    a_cat_b, counts = torch.cat([a, b]).unique(return_counts=True)
    intersection = a_cat_b[torch.where(counts.gt(1))]
    return intersection

def width_fitness(predicted, expected_output):
    """ How close the predicted image is to have the right width. Less is better."""
    return abs(predicted.shape[0] - expected_output.shape[0])

def height_fitness(predicted, expected_output):
    """ How close the predicted image is to have the right height. Less is better."""
    return abs(predicted.shape[1] - expected_output.shape[1])

def activated_pixels_fitness(p, e):
    """ How close the predicted image to have the right pixels. Less is better."""
    shape = (max(p.shape[0], e.shape[0]), max(p.shape[1], e.shape[1]))
    diff = torch.zeros(shape, dtype=torch.uint8)
    diff[0:p.shape[0], 0:p.shape[1]] = (p > 0).type(torch.uint8) 
    diff[0:e.shape[0], 0:e.shape[1]] -= (e > 0).type(torch.uint8) 
    
    return (diff != 0).sum()

def colors_fitness(p, e):
    p_colors = torch.unique(p)
    e_colors = torch.unique(e)
    
    nb_inter = len(tensor_intersect1d(p_colors, e_colors))

    return (len(p_colors) - nb_inter) + (len(e_colors) - nb_inter)

fitness_functions = [colors_fitness, activated_pixels_fitness, height_fitness, width_fitness]


def product_less(a, b):
    """ Return True iff the two tuples a and b respect a<b for the partial order. """
    a = torch.tensor(a)
    b = torch.tensor(b)
    return (a < b).all()

def evaluate_fitness(program, task):
    """ Take a program and a task, and return its fitness score as a tuple. """
    score = torch.zeros((len(fitness_functions)))
    
    # For each sample
    for sample in task:
        i = torch.tensor(sample['input'])
        o = torch.tensor(sample['output'])
        
        # For each fitness function
        for index, fitness_function in enumerate(fitness_functions):
            images = evaluate(program, i)
            if images == []: # Penalize no prediction!
                score[index] += 500
            else: # Take only the score of the first output
                score[index] = fitness_function(images[0], o)
    return tuple(score)
