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
%load_ext autoreload
%autoreload 2

import json
import random
import torch
from tqdm.notebook import trange

from operations import *
from util import *
from fitness import *
from candidate import *

#%%
# Data

TRAIN_PATH = './ARC/data/training'
EVAL_PATH = './ARC/data/evaluation'

if not os.path.exists(TRAIN_PATH): 
    !git clone https://github.com/fchollet/ARC.git

training_tasks = sorted(os.listdir(TRAIN_PATH))
eval_tasks = sorted(os.listdir(EVAL_PATH))

# all_tasks = training_tasks + eval_tasks
all_tasks = training_tasks



#%% 
def build_candidates(best_candidates=[], length_limit=4, nb_candidates=100):
    """
    Create a pool of fresh candidates using the `allowed_nodes`.
    
    The pool contain a mix of new single instructions programs
    and mutations of the best candidates.
    """
    new_candidates = []
    
    # Until we have enougth new candidates
    while(len(new_candidates) < nb_candidates):
        # Add new programs
        for i in range(5):
            new_candidates += [Candidate()]
        
        # Create new programs based on each best candidate
        # for best_program in best_candidates:
        #     new_candidates += [copy.deepcopy(best_program)]
        #     new_candidates[-1].mutate()
        
        # create 10 crossovers
        if len(best_candidates) > 0:
            for _ in range(10):
                # Select two parents
                parent_1 = random.choice(list(best_candidates))
                parent_2 = random.choice(list(best_candidates))
                # Create a child
                child = parent_1.crossover(parent_2)
                child.mutate()
                # Add the child to the pool
                new_candidates += [child]    
   
    # Truncate if we have too many candidates
    random.shuffle(new_candidates)
    return new_candidates[:nb_candidates]


#%%
def pareto_front_model(task, candidates_nodes, max_iterations=50, length_limit=4, verbose=False, task_id=0, nb_candidates=100,show_progress=False):
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
        candidates = build_candidates(best_candidates.values(), nb_candidates=nb_candidates)
        # They will be stored in the `best_candidates` dictionary
        # where the key of each program is its fitness score.
        for _, candidate in enumerate(candidates):
            score = candidate.evaluate_fitness(task)
            is_incomparable = True # True if we cannot compare this candidates score with any other
            
            # Compare the new candidate to the existing best candidates
            best_candidates_items = list(best_candidates.items())
            for best_score, best_candidate in best_candidates_items:
                if product_less(score, best_score):
                    # dominates this best candidate
                    # Remove previous best candidate and add the new one
                    del best_candidates[best_score]
                    best_candidates[score] = candidate
                    is_incomparable = False # The candidates are comparable
                if product_less(best_score, score) or (best_score == score):
                    is_incomparable = False # The candidates are comparable
            if is_incomparable: # The two candidates are incomparable
                best_candidates[score] = candidate

        # For each best candidate, we look if we have an answer
        for program in best_candidates.values():
            def get_key(val):
                for key, value in best_candidates.items():
                    if val == value:
                        return key
                return "key doesn't exist"
            score = get_key(program)
            score_is_0 =  score == (0,)*len(fitness_functions)
            if is_solution(program, task):
                return program
            elif score_is_0:
                print("Score was 0 but no solution")
                # visualize_network(program)
                i = torch.tensor(task[0]['input']).to(device)
                o = torch.tensor(task[0]['output']).to(device)
                i = i.type(torch.FloatTensor).clone()
                o = o.type(torch.FloatTensor).clone()
                output = program.evaluate(i)
                output = output[0]
                # show_image_list([i, o] + output, ["Input", "Output"] + [f"Prediction {i}" for i in range(len(output))])

        # Give some information by selecting a random candidate
        if verbose:
            print("Best candidates length:", len(best_candidates))
            random_candidate_score = random.choice(list(best_candidates.keys()))
            print("Random candidate score:", random_candidate_score, "average:", ((random_candidate_score[0] + random_candidate_score[1] + random_candidate_score[2]) / 3).item())
            print("Random candidate implementation:", program_desc(best_candidates[random_candidate_score]))
        
        scores = [np.max(c) for c in list(best_candidates.keys())]
        top_score = torch.tensor(scores).min().item()
        top_index = list(scores).index(top_score)

        # best_candidates sorted by score
        best_candidates_items = list(best_candidates.items())
        best_candidates_items.sort(key=lambda x: x[0])
        best_candidates_score = [c[0] for c in best_candidates_items][0]

        if show_progress:
            pbar.set_description_str(f"{task_id}: {top_score:.2f}")
            pbar.set_postfix_str(f"{[f'{s:.2f}|' for s in best_candidates_score]}")
            
    if False:
        best = None
        best_score = (np.inf,)*len(fitness_functions)
        for i, (score, candidate) in enumerate(best_candidates.items()):
            if product_less(score, best_score):
                best = candidate
                best_score = score
        inp = torch.tensor(task[-1]['input']).to(device)
        out = torch.tensor(task[-1]['output']).to(device)
        inp = inp.type(torch.FloatTensor)
        out = out.type(torch.FloatTensor)
        results = best.evaluate(inp)
        visualize_network(best)
        show_image_list([inp, out, results[0][0]])
        print(f"{task_id}: {top_score:.2f}")
    return None


#%%
# testing

print([n.__name__ for n in all_operations])
show_progress = True
do_shuffle = True

per_task_iterations = 50
length_limit = 4 # Maximal length of a program
pop_size = 200

num_correct = 0
num_total = 0
candidates_nodes = all_operations
pbar = trange(len(all_tasks))

random_indices = random.sample(range(len(all_tasks)), len(all_tasks))


used_nodes = {}

for task_id in pbar:
    try:
        indx = random_indices[task_id] if do_shuffle else task_id
        
        indx = 115

        num_total+=1
        task_file = str((TRAIN_PATH if indx < len(training_tasks) else EVAL_PATH) + "/" + all_tasks[indx])
        with open(task_file, 'r') as f:
            task = json.load(f)

        program = pareto_front_model(task['train'], candidates_nodes, max_iterations=per_task_iterations, length_limit=length_limit, verbose=False, task_id=indx,nb_candidates=pop_size, show_progress=show_progress)
        pbar.set_description_str(f"{num_correct/num_total}")
        
        if program is None:
            # print("No program was found")
            continue # no answer in top 3
        else:
            num_correct+=1
            nodes = [n.activation.__name__ for n in program.input_nodes+program.hidden_nodes+program.output_nodes]
            for node in nodes:
                if node not in used_nodes:
                    used_nodes[node] = 0
                used_nodes[node]+=1
            # visualize_network(program)
    except KeyboardInterrupt:
        break
    # except Exception as e:
    #     print(type(e), e)
    #     continue

plt.bar(list(used_nodes.keys()), list(used_nodes.values()))
plt.show()
#%%
plt.figure(figsize=(10,10))
plt.bar(list(used_nodes.keys()), list(used_nodes.values()))
plt.gca().set_xticklabels(list(used_nodes.keys()), rotation='vertical')
plt.show()
#%%
import time
with open("./results.txt",'w') as f:
    f.write(f"{time.time()}: {num_correct/num_total}")