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

if not os.path.exists(TRAIN_PATH): 
    !git clone https://github.com/fchollet/ARC.git

MAX_GRID = 30
MAX_INPUT_EXAMPLES = 10
MAX_VALUE = 9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
train_task_files = os.listdir(TRAIN_PATH)
train_tasks = []
for task_file in train_task_files:
    with open(os.path.join(TRAIN_PATH , task_file), 'r') as f:
        task = json.load(f)
        train_tasks.append(task)
eval_task_files = os.listdir(EVAL_PATH)
eval_tasks = []
for task_file in eval_task_files:
    with open(os.path.join(EVAL_PATH , task_file), 'r') as f:
        task = json.load(f)
        eval_tasks.append(task)

#%%
def get_batches(tasks, group, shuffle=True, batch_size=16):
    if shuffle:
        random.shuffle(tasks)
    batch_index = 0
    XS = []
    YS = []
    while batch_index < len(tasks):
        batch_X = []
        batch_Y = []
        for idx, task in enumerate(tasks[batch_index:batch_index+batch_size]):
            if group == "both":
                tasks_iterable = enumerate(task["train"]+ task["test"])
            else:
                tasks_iterable = enumerate(task[group])
            for i, example in tasks_iterable:
                example_in = torch.tensor(example["input"], dtype=torch.float32, device=device)
                example_out = torch.tensor(example["output"], dtype=torch.float32, device=device)
                batch_X.append(example_in)
                batch_Y.append(example_out)
        XS.append(batch_X)
        YS.append(batch_Y)
        batch_index += batch_size
    return XS, YS 

#%%
def pad_output(y_hat, train_y):
    # cheating?
    # pad the output to match the input
    if y_hat.shape[-2] < train_y.shape[-2]:
        y_hat = torch.nn.functional.pad(y_hat, (0,0, train_y.shape[-2] - y_hat.shape[-2], 0), value=0)
    elif y_hat.shape[-2] > train_y.shape[-2]:
        train_y = torch.nn.functional.pad(train_y, (0,0, y_hat.shape[-2] - train_y.shape[-2], 0), value=0)
    if y_hat.shape[-1] < train_y.shape[-1]:
        y_hat = torch.nn.functional.pad(y_hat, (0, train_y.shape[-1] - y_hat.shape[-1], 0, 0), value=0)
    elif y_hat.shape[-1] > train_y.shape[-1]:
        train_y = torch.nn.functional.pad(train_y, (0, y_hat.shape[-1] - train_y.shape[-1], 0, 0), value=0)
    return y_hat, train_y

def evaluate(model):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        xs,ys = get_batches(train_tasks, "both", shuffle=False, batch_size=batch_size)
        correct = 0
        total = 0
        test_loss = 0
        for x, y  in zip(xs,ys):
            for test_x, test_y in zip(x,y):
                # NO parallelization!!
                test_x = test_x.unsqueeze(0).unsqueeze(0)
                test_y = test_y.unsqueeze(0).unsqueeze(0)

                y_hat = model(test_x)
                if y_hat.shape != test_y.shape:
                    y_hat, test_y = pad_output(y_hat, test_y)

                test_loss += criterion(y_hat/MAX_VALUE, test_y/MAX_VALUE)
                if torch.all(torch.isclose(torch.round(y_hat), test_y)):
                    correct += 1
                total += 1
        test_loss /= total

    return test_loss.item(), correct/total

#%%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.input = nn.Conv2d(1, 64, 3, padding=1)
        self.hidden = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Dropout(0.5),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.output = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        x = self.tanh(x) 
        x = x * MAX_VALUE
        # x = torch.round(x)
        return x

#%%
batch_size = 16
num_epochs = 5000

#%%
model = CNN()
model.to(device)

pbar = trange(num_epochs)
losses = []
accs = []
val_losses = []

optim = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in pbar:
    model.train()
    xs,ys = get_batches(train_tasks, "both", shuffle=True, batch_size=batch_size)
    for x, y  in zip(xs,ys):
        optim.zero_grad()
        learn_loss = 0
        for train_x, train_y in zip(x,y):
            # NO parallelization!!
            train_x = train_x.unsqueeze(0).unsqueeze(0)
            train_y = train_y.unsqueeze(0).unsqueeze(0)

            y_hat = model(train_x)
            if y_hat.shape != train_y.shape:
                # pad output to calculate loss
                y_hat, train_y = pad_output(y_hat, train_y)

            learn_loss += criterion(y_hat/MAX_VALUE, train_y/MAX_VALUE)
            
        learn_loss = learn_loss / len(x)
        learn_loss.backward()
        optim.step()

        pbar.set_description(f'Epoch {epoch} Loss {learn_loss.item():.4f}')
        losses.append(learn_loss.item())
    
    if epoch % 2 == 0:
        test_loss, test_acc = evaluate(model)
        val_losses.append(test_loss)
        accs.append(test_acc)
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
#%%
import matplotlib.pyplot as plt
plt.title("Train loss")
plt.plot(losses)
plt.show()

plt.title("Validation loss")
plt.plot(val_losses)
plt.show()

plt.title("Accuracy")
plt.plot(accs)
plt.show()


#%%
# evaluate
test_loss, test_acc = evaluate(model)
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
#%%
# show examples
import torchvision.utils
from torchvision.transforms import ToPILImage
model.eval()
with torch.no_grad():
    # show examples:
    count= 0 
    xs,ys = get_batches(train_tasks, "both", shuffle=True, batch_size=2)
    for x, y  in zip(xs,ys):
        for train_x, train_y in zip(x,y):
            train_x = train_x.unsqueeze(0).unsqueeze(0)
            train_y = train_y.unsqueeze(0).unsqueeze(0)

            y_hat = model(train_x)
            if y_hat.shape != train_y.shape:
                y_hat, train_y = pad_output(y_hat, train_y)

            y_hat = torch.round(y_hat).squeeze(0).squeeze(0)
            train_y = torch.round(train_y).squeeze(0).squeeze(0)
            train_x = train_x.squeeze(0).squeeze(0)

            fig, ax = plt.subplots(1,3)
            ax[0].imshow(train_x.cpu().numpy(), cmap="Paired_r", vmin=0, vmax=MAX_VALUE, interpolation="none")
            ax[1].imshow(train_y.cpu().numpy(), cmap="Paired_r", vmin=0, vmax=MAX_VALUE, interpolation="none")
            ax[2].imshow(y_hat.cpu().numpy(), cmap="Paired_r", vmin=0, vmax=MAX_VALUE, interpolation="none")
            ax[0].set_title("Input")
            ax[1].set_title("Output")
            ax[2].set_title("Prediction")
            plt.show()
            plt.close()
        
        count+=1
        print("#"*100)
        if count > 3:
            break
        