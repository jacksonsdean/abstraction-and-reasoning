import os
# %cd /content
# !rm -rf "abstraction-and-reasoning"
# !git clone --recurse-submodules https://github.com/jacksonsdean/abstraction-and-reasoning.git
%cd "/content/abstraction-and-reasoning"
# !git checkout evolution
# !pwd
# !ls
# using code from: https://www.kaggle.com/code/zenol42/dsl-and-genetic-algorithm-applied-to-arc/notebook
#%%
import json
import random
import torch
from tqdm import trange

from operations import *
from util import *
from fitness import *


#%%
# Data

TRAIN_PATH = './ARC/data/training'
EVAL_PATH = './ARC/data/evaluation'

if not os.path.exists(TRAIN_PATH): 
    !git clone https://github.com/fchollet/ARC.git

training_tasks = sorted(os.listdir(TRAIN_PATH))
eval_tasks = sorted(os.listdir(EVAL_PATH))

all_tasks = training_tasks + eval_tasks
#%% 
def build_candidates(allowed_nodes=[identity], best_candidates=[], length_limit=4, nb_candidates=200):
    """
    Create a pool of fresh candidates using the `allowed_nodes`.
    
    The pool contain a mix of new single instructions programs
    and mutations of the best candidates.
    """
    new_candidates = []
    
    def random_node():
        return random.choice(allowed_nodes)
    
    # Until we have enougth new candidates
    while(len(new_candidates) < nb_candidates):
        # Add 10 new programs
        for i in range(5):
            new_candidates += [[random_node()]]
        
        # Create new programs based on each best candidate
        for best_program in best_candidates:
            # Add one op on its right but limit the length of the program
            if len(best_program) < length_limit - 1:
                new_candidates += [[random_node()] + best_program]
            # Add one op on its left but limit the length of the program
            if len(best_program) < length_limit - 1:
                new_candidates += [best_program + [random_node()]]
            # Mutate one instruction of the existing program
            new_candidates += [list(best_program)]
            new_candidates[-1][random.randrange(0, len(best_program))] = random_node()
   
    # Truncate if we have too many candidates
    random.shuffle(new_candidates)
    return new_candidates[:nb_candidates]


#%%
def build_model(task, candidates_nodes, max_iterations=50, length_limit=4, verbose=False, task_id=0):
    
    show_progress = False
    if verbose:
        print("Candidates nodes are:", [program_desc([n]) for n in candidates_nodes])
        print()
    if show_progress:
        pbar = trange(max_iterations)
    else:
        pbar = range(max_iterations)
    best_candidates = {} # A dictionary of {score:candidate}
    for i in pbar:
        if verbose:
            print("Iteration ", i+1)
            print("-" * 10)
        
        # Create a list of candidates
        candidates = build_candidates(candidates_nodes, best_candidates.values(), length_limit=length_limit)
        
        # Keep candidates with best fitness.
        # They will be stored in the `best_candidates` dictionary
        # where the key of each program is its fitness score.
        for candidate in candidates:
            score = evaluate_fitness(candidate, task)
            is_incomparable = True # True if we cannot compare the two candidate's scores
            
            # Compare the new candidate to the existing best candidates
            best_candidates_items = list(best_candidates.items())
            for best_score, best_candidate in best_candidates_items:
                if product_less(score, best_score):
                    # Remove previous best candidate and add the new one
                    del best_candidates[best_score]
                    best_candidates[score] = candidate
                    is_incomparable = False # The candidates are comparable
                if product_less(best_score, score) or best_score == score:
                    is_incomparable = False # The candidates are comparable
            if is_incomparable: # The two candidates are incomparable
                best_candidates[score] = candidate

        # For each best candidate, we look if we have an answer
        for program in best_candidates.values():
            if is_solution(program, task):
                return program
            
        # Give some information by selecting a random candidate
        if verbose:
            print("Best candidates length:", len(best_candidates))
            random_candidate_score = random.choice(list(best_candidates.keys()))
            print("Random candidate score:", random_candidate_score, "average:", ((random_candidate_score[0] + random_candidate_score[1] + random_candidate_score[2]) / 3).item())
            print("Random candidate implementation:", program_desc(best_candidates[random_candidate_score]))
        
        top_score = torch.tensor([torch.tensor(c).mean() for c in list(best_candidates.keys())]).min().item()
        if show_progress:
            pbar.set_description_str(f"{task_id}: {top_score:.2f}")
    return None


#%%
# testing

per_task_iterations = 20
length_limit = 4 # Maximal length of a program

num_correct = 0
num_total = 0
candidates_nodes = all_operations

pbar = trange(len(all_tasks))

for task_id in pbar:
    num_total+=1
    task_file = str((TRAIN_PATH if task_id < len(training_tasks) else EVAL_PATH) + "/" + all_tasks[task_id])
    with open(task_file, 'r') as f:
        task = json.load(f)

    program = build_model(task['train'], candidates_nodes, max_iterations=per_task_iterations, length_limit=length_limit, verbose=False, task_id=task_id)
    pbar.set_description_str(f"{num_correct/num_total}")
    if program is None:
        # print("No program was found")
        continue # no answer in top 3
    else:
        num_correct+=1

        # print("Found program:", program_desc(program))
        # print("Fitness:", evaluate_fitness(program, task['train']))
        # print("Is solution:", is_solution(program, task['train']))
        # results = evaluate(program=program, input_image=task['test'][0]['input'])
        # show_image_list([task['test'][0]['input'], results[1]])
