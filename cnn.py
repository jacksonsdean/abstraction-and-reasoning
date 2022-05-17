#%%
import json
import os
import torch
from torch import nn
from tqdm.notebook import trange
from torchvision.transforms import Resize
import random
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
TRAIN_PATH = './ARC/data/training'
EVAL_PATH = './ARC/data/evaluation'

if not os.path.exists(TRAIN_PATH): 
    !git clone https://github.com/fchollet/ARC.git

MAX_GRID = 30
MAX_INPUT_EXAMPLES = 10
MAX_VALUE = 9
USE_ATTENTION = False
USE_PADDING = True
USE_RESIDUAL = False # causes overfitting

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

    data_idx = 0
    num_batches = len(tasks) // batch_size
    if not USE_PADDING:
        XS = []
        YS = []
    else:
        XS = torch.zeros((num_batches, batch_size, 1, MAX_GRID, MAX_GRID), device=device)
        YS = torch.zeros((num_batches, batch_size, 1, MAX_GRID, MAX_GRID), device=device)

    batch_idx = 0
    while data_idx < len(tasks):
        batch_X = []
        batch_Y = []
        for task_idx, task in enumerate(tasks[data_idx:data_idx+batch_size]):
            if group == "both":
                tasks_iterable = enumerate(task["train"]+ task["test"])
            else:
                tasks_iterable = enumerate(task[group])
            for i, example in tasks_iterable:
                example_in = torch.tensor(example["input"], dtype=torch.float32, device=device)
                example_out = torch.tensor(example["output"], dtype=torch.float32, device=device)
                if USE_PADDING:
                    example_in = pad_to_max(example_in)
                    example_out = pad_to_max(example_out)
                    XS[batch_idx, task_idx] = example_in
                    YS[batch_idx, task_idx] = example_out
                batch_X.append(example_in)
                batch_Y.append(example_out)
        
        if not USE_PADDING:
            XS.append(batch_X)
            YS.append(batch_Y)
        
        data_idx += batch_size
        batch_idx +=1
    
    return XS, YS 

#%%
# defines

def show_examples(num = 4):
    model.eval()
    with torch.no_grad():
        # show examples:
        count= 0 
        xs,ys = get_batches(eval_tasks, "both", shuffle=True, batch_size=1)
        for x, y  in zip(xs,ys):
            if not USE_PADDING:
                for train_x, train_y in zip(x,y):
                    train_x = train_x.unsqueeze(0).unsqueeze(0)
                    train_y = train_y.unsqueeze(0).unsqueeze(0)

                    y_hat = model(train_x)
                    if y_hat.shape != train_y.shape:
                        y_hat, train_y = pad_output(y_hat, train_y)

                    y_hat = torch.round(y_hat).squeeze(0).squeeze(0).squeeze(0)
                    train_y = torch.round(train_y).squeeze(0).squeeze(0).squeeze(0)
                    train_x = train_x.squeeze(0).squeeze(0).squeeze(0)
            else:
                # x = transforms.Lambda(lambda s:s.repeat(1, 3,1,1))(x)
                y_hat = model(x)
                # y_hat = y_hat["out"]
                # y_hat = y_hat.mean(1).unsqueeze(1)
                y_hat = torch.round(y_hat).view(MAX_GRID, MAX_GRID)
                y = torch.round(y).view(MAX_GRID, MAX_GRID)
                # x = transforms.Grayscale()(x)
                x=x.view(MAX_GRID, MAX_GRID)
                train_y = y
                train_x = x

            fig, ax = plt.subplots(1,3)
            ax[0].set_title("Input")
            ax[0].imshow(train_x.cpu().numpy(), cmap="Paired_r", vmin=0, vmax=MAX_VALUE, interpolation="none")
            ax[1].set_title("Output")
            ax[1].imshow(train_y.cpu().numpy(), cmap="Paired_r", vmin=0, vmax=MAX_VALUE, interpolation="none")
            ax[2].set_title("Prediction")
            ax[2].imshow(y_hat.cpu().numpy(), cmap="Paired_r", vmin=0, vmax=MAX_VALUE, interpolation="none")
            correct = (y_hat == train_y).sum().item() / (y_hat.shape[0] * y_hat.shape[1])
            ax[2].set_title(f"Prediction: {correct:.2f}")
            plt.show()
            plt.close()
            
            count+=1
            print("#"*100)
            if count >= num:
                break
            
def pad_output(y_hat, train_y):
    # cheating?
    # pad the output to match the input
    if y_hat.shape[-2] < train_y.shape[-2]:
        y_hat = torch.nn.functional.pad(y_hat, (0,0, train_y.shape[-2] - y_hat.shape[-2], 0), value=-1)
    elif y_hat.shape[-2] > train_y.shape[-2]:
        train_y = torch.nn.functional.pad(train_y, (0,0, y_hat.shape[-2] - train_y.shape[-2], 0), value=-1)
    if y_hat.shape[-1] < train_y.shape[-1]:
        y_hat = torch.nn.functional.pad(y_hat, (0, train_y.shape[-1] - y_hat.shape[-1], 0, 0), value=-1)
    elif y_hat.shape[-1] > train_y.shape[-1]:
        train_y = torch.nn.functional.pad(train_y, (0, y_hat.shape[-1] - train_y.shape[-1], 0, 0), value=-1)
    return y_hat, train_y

def pad_to_max(x):
    x = torch.nn.functional.pad(x, (0,0, MAX_GRID - x.shape[-2], 0), value=-1)
    x = torch.nn.functional.pad(x, (0, MAX_GRID - x.shape[-1] ,0,0), value=-1)
    return x


def evaluate(model):
    model.eval()
    with torch.no_grad():
        xs, ys = get_batches(eval_tasks, "both", shuffle=False, batch_size=batch_size)
        correct = 0
        total = 0
        if not USE_PADDING:
            test_loss = 0
        for x, y  in zip(xs,ys):
            if not USE_PADDING:
                # NO parallelization!!
                for test_x, test_y in zip(x,y):
                    test_x = test_x.unsqueeze(0).unsqueeze(0)
                    test_y = test_y.unsqueeze(0).unsqueeze(0)

                    y_hat = model(test_x)
                    if y_hat.shape != test_y.shape:
                        y_hat, test_y = pad_output(y_hat, test_y)

                    test_loss += criterion(y_hat/MAX_VALUE, test_y/MAX_VALUE)
                    if torch.all(torch.isclose(torch.round(y_hat), test_y)):
                        correct += 1
                    total += 1
            else:
                # x = transforms.Lambda(lambda s:s.repeat(1, 3,1,1))(x)
                y_hat = model(x)
                # y_hat = y_hat["out"]
                # y_hat = y_hat.mean(1).unsqueeze(1)
                test_loss = criterion(y_hat/MAX_VALUE, y/MAX_VALUE)
                for y_hat_, test_y_ in zip(y_hat, y):
                    if y_hat_.shape != test_y_.shape:
                        y_hat_, test_y_ = pad_output(y_hat_, test_y_)
                    if torch.all(torch.isclose(torch.round(y_hat_), test_y_)):
                        correct += 1
                    total += 1                    

        if not USE_PADDING:
            test_loss /= total

    return test_loss.item(), correct/total

#%%
class Sin(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)

class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super().__init__()
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def _conv(self,n_in,n_out):
        return nn.Conv1d(n_in, n_out, kernel_size=1, bias=False)


    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()

class CNN(nn.Module):
    def __init__(self, use_attn=False):
        super(CNN, self).__init__()
        self.use_attn = use_attn
        # self.activation = nn.ReLU()
        self.activation = nn.Tanh()
        # self.activation = Sin()
        self.input = nn.Conv2d(1, 15 if use_attn else 64, 3, padding=1)
        self.hidden = nn.Sequential(
            nn.Conv2d(15 if use_attn else 64, 256, kernel_size=5, stride=1, padding=2),
            self.activation,
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.Dropout(0.5),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # self.activation,
            # nn.Dropout(0.85),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # self.activation,
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # self.activation,
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # self.activation,
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # self.activation,
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            self.activation,
        )
        self.output = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.mh_attention = nn.MultiheadAttention(MAX_GRID, 15)
        self.attention = SelfAttention(15)
        self.fc = nn.Linear(MAX_GRID*MAX_GRID, MAX_GRID*MAX_GRID)
        self.residual = nn.Sequential()
        
    def forward(self, x):
        if USE_RESIDUAL:
            residual = self.residual(x)
        x = self.input(x)
        if self.use_attn:
            x = self.attention(x)
        x = self.hidden(x)
        x = self.output(x)
        if USE_RESIDUAL:
            x+=residual
        x = self.activation(x) 
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.activation(x) 
        x = x.view(x.size(0),1, MAX_GRID, MAX_GRID)
        x = x * MAX_VALUE
        # x = torch.round(x)
        return x

#%%
batch_size = 16
num_epochs = 60000
lr = 1e-5
weight_decay=1e-5

model = CNN(USE_ATTENTION)
# model = resnet18(pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
model.to(device)

pbar = trange(num_epochs)
losses = []
accs = []
val_losses = []

optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

for epoch in pbar:
    try:
        model.train()
        xs,ys = get_batches(train_tasks, "both", shuffle=True, batch_size=batch_size)
        for x, y  in zip(xs,ys):
            optim.zero_grad()
            learn_loss = 0
            if not USE_PADDING:
                for train_x, train_y in zip(x,y):
                    # NO parallelization!!
                    train_x = train_x.unsqueeze(0).unsqueeze(0)
                    train_y = train_y.unsqueeze(0).unsqueeze(0)
                    y_hat = model(train_x)
                    if y_hat.shape != train_y.shape:
                        # pad output to calculate loss
                        y_hat, train_y = pad_to_max(y_hat), pad_to_max(train_y)

                    learn_loss += criterion(y_hat/MAX_VALUE, train_y/MAX_VALUE)
                    
                learn_loss = learn_loss / len(x)
            
            else:
                # with parallelization, faster
                # x = transforms.Lambda(lambda s:s.repeat(1, 3,1,1))(x)
                y_hat = model(x)
                # y_hat= y_hat["out"]
                # y_hat = y_hat.mean(1).unsqueeze(1)
                learn_loss = criterion(y_hat/MAX_VALUE, y/MAX_VALUE)
            
            learn_loss.backward()
            optim.step()

        
        if epoch % 10 == 0:
            test_loss, test_acc = evaluate(model)
            val_losses.append(test_loss)
            accs.append(test_acc)
        pbar.set_description(f'Epoch {epoch} train:{learn_loss.item():.4f} val:{test_loss:.4f} acc:{test_acc:.4f}')
        losses.append(learn_loss.item())

        if epoch %100 ==0:
            show_examples()

    except KeyboardInterrupt:
        break
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
show_examples()
