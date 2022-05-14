#%%
import json
import os
import torch
from torch import nn
from tqdm.notebook import trange


TRAIN_PATH = './ARC/data/training'

MAX_GRID = 30
MAX_INPUT_EXAMPLES = 10

#%%
train_task_files = os.listdir(TRAIN_PATH)
train_tasks = []
for task_file in train_task_files:
    with open(os.path.join(TRAIN_PATH , task_file), 'r') as f:
        task = json.load(f)
        train_tasks.append(task)

#%%

class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnected, self).__init__()
        self.input = nn.Linear(input_size, 256)
        self.hidden = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.output = nn.Linear(256, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

#%%
batch_size = 1

train_input_size = (MAX_GRID * MAX_GRID * MAX_INPUT_EXAMPLES)
train_output_size = (MAX_GRID * MAX_GRID * MAX_INPUT_EXAMPLES)

test_input_size = 1*(MAX_GRID * MAX_GRID * 1)
test_output_size = 1*(MAX_GRID * MAX_GRID * 1)

num_epochs = 2

#%%
train_net = FullyConnected(input_size=train_input_size, output_size=train_output_size)

pbar = trange(num_epochs)
losses = []

optim = torch.optim.Adam(train_net.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in pbar:
    for task in train_tasks:
        optim.zero_grad()
        train_x = torch.zeros(train_input_size)
        train_y = torch.zeros(train_output_size)
        for idx, task in enumerate(task["train"]):
            this_in = torch.tensor(task["input"]).view(-1)
            train_x[idx:idx+len(this_in)] = this_in 

            this_out = torch.tensor(task["output"]).view(-1)
            train_y[idx:idx+len(this_out)] = this_out


        y_hat = train_net(train_x)
        learn_loss = criterion(y_hat, train_y)
        learn_loss.backward()
        optim.step()

    for task in train_tasks:
        optim.zero_grad()
        train_x = torch.zeros(train_input_size)
        train_y = torch.zeros(train_output_size)
        for idx, task in enumerate(task["test"]):
            this_in = torch.tensor(task["input"]).view(-1)
            train_x[idx:idx+len(this_in)] = this_in 

            this_out = torch.tensor(task["output"]).view(-1)
            train_y[idx:idx+len(this_out)] = this_out


        y_hat = train_net(train_x)
        loss = criterion(y_hat, train_y)
        loss.backward()
        optim.step()


        pbar.set_description(f'Epoch {epoch} Loss {loss.item()}')
        losses.append(loss.item())
#%%
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()