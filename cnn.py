#%%
import json
import os
import torch
from torch import nn
from tqdm.notebook import trange
from torchvision.transforms import Resize
import random
TRAIN_PATH = './ARC/data/training'
EVAL_PATH = './ARC/data/evaluation'

MAX_GRID = 30
MAX_INPUT_EXAMPLES = 10
MAX_VALUE = 9

#%%
train_task_files = os.listdir(TRAIN_PATH)
train_tasks = []
for task_file in train_task_files:
    with open(os.path.join(TRAIN_PATH , task_file), 'r') as f:
        task = json.load(f)
        train_tasks.append(task)


#%%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.input = nn.Conv2d(MAX_INPUT_EXAMPLES, 16, kernel_size=3, stride=1, padding=1)
        self.hidden = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.output = nn.Conv2d(16, MAX_INPUT_EXAMPLES, kernel_size=3, stride=1, padding=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        x = self.Sigmoid(x)
        return x

#%%
batch_size = 32
num_epochs = 5

#%%
train_net = CNN()

pbar = trange(num_epochs)
losses = []

optim = torch.optim.Adam(train_net.parameters(), lr=0.001)
criterion = nn.MSELoss()

resize = Resize((MAX_GRID, MAX_GRID))
for epoch in pbar:
    random.shuffle(train_tasks)
    batch_index = 0
    while batch_index < len(train_tasks):
        train_x = torch.zeros(batch_size, MAX_INPUT_EXAMPLES, MAX_GRID, MAX_GRID)
        train_y = torch.zeros(batch_size, MAX_INPUT_EXAMPLES, MAX_GRID, MAX_GRID)
        for idx, task in enumerate(train_tasks[batch_index:batch_index+batch_size]):
            for i, example in enumerate(task["train"]):
                example_in = torch.tensor(example["input"])
                example_out = torch.tensor(example["output"])
                example_in = resize(example_in.unsqueeze(0)).squeeze(0)
                example_out = resize(example_out.unsqueeze(0)).squeeze(0)
                train_x[idx, i] = example_in/MAX_VALUE
                train_y[idx, i] = example_out/MAX_VALUE
       

        train_net.train()
        optim.zero_grad()
        y_hat = train_net(train_x)
        learn_loss = criterion(y_hat, train_y)
        learn_loss.backward()
        optim.step()
      
      
        train_net.eval()
        with torch.no_grad():
            test_x = torch.zeros(batch_size, MAX_INPUT_EXAMPLES, MAX_GRID, MAX_GRID)
            test_y = torch.zeros(batch_size, MAX_INPUT_EXAMPLES, MAX_GRID, MAX_GRID)
            for idx, task in enumerate(train_tasks[batch_index:batch_index+batch_size]):
                for i, question in enumerate(task["test"]):
                    question_in = torch.tensor(question["input"])
                    question_out = torch.tensor(question["output"])
                    question_in = resize(question_in.unsqueeze(0)).squeeze(0)
                    question_out = resize(question_out.unsqueeze(0)).squeeze(0)
                    test_x[idx, i] = question_in/MAX_VALUE
                    test_y[idx, i] = question_out/MAX_VALUE
        
            y_hat = train_net(train_x)
            test_loss = criterion(y_hat, train_y)
      

        batch_index += batch_size
        pbar.set_description(f'Epoch {epoch} Loss {test_loss.item():.4f}')
        losses.append(test_loss.item())
#%%
import matplotlib.pyplot as plt
plt.title("Train loss")
plt.plot(losses)
plt.show()

#%%
# evaluate
train_net.eval()
test_loss = 0.0
with torch.no_grad():
    eval_task_files = os.listdir(EVAL_PATH)
    eval_tasks = []
    for task_file in eval_task_files:
        with open(os.path.join(EVAL_PATH , task_file), 'r') as f:
            task = json.load(f)
            eval_tasks.append(task)
    
    correct = 0
    total = 0
    for task in eval_tasks:
        test_x = torch.zeros(1, MAX_INPUT_EXAMPLES, MAX_GRID, MAX_GRID)
        test_y = torch.zeros(1, MAX_INPUT_EXAMPLES, MAX_GRID, MAX_GRID)
        for idx, task in enumerate([task]):
            for i, question in enumerate(task["test"]):
                question_in = torch.tensor(question["input"])
                question_out = torch.tensor(question["output"])
                question_in = resize(question_in.unsqueeze(0)).squeeze(0)
                question_out = resize(question_out.unsqueeze(0)).squeeze(0)
                test_x[idx, i] = question_in/MAX_VALUE
                test_y[idx, i] = question_out/MAX_VALUE
        
        y_hat = train_net(test_x)
        test_loss += criterion(y_hat, test_y)
        if torch.all(torch.isclose(y_hat, test_y)):
            correct += 1
        total += 1

    print(f"Loss: {test_loss/total}")
    print(f"Accuracy: {correct/total}")

#%%
# show examples
import torchvision.utils
from torchvision.transforms import ToPILImage
train_net.eval()
with torch.no_grad():
    for task in eval_tasks:
        test_x = torch.zeros(1, MAX_INPUT_EXAMPLES, MAX_GRID, MAX_GRID)
        test_y = torch.zeros(1, MAX_INPUT_EXAMPLES, MAX_GRID, MAX_GRID)
        for idx, task in enumerate([task]):
            for i, question in enumerate(task["test"]):
                question_in = torch.tensor(question["input"])
                question_out = torch.tensor(question["output"])
                question_in = resize(question_in.unsqueeze(0)).squeeze(0)
                question_out = resize(question_out.unsqueeze(0)).squeeze(0)
                test_x[idx, i] = question_in/MAX_VALUE
                test_y[idx, i] = question_out/MAX_VALUE
        
        y_hat = train_net(test_x)
        for i in range(y_hat.shape[1]):
            examples = torch.cat((test_x[0, i], y_hat[0, i], test_y[0, i]), dim=0)
            grid = torchvision.utils.make_grid(examples, nrow=3)
            grid = ToPILImage()(grid)
            plt.imshow(grid)
            plt.show()
        break