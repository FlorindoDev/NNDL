import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
from torchvision.utils import make_grid
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

print("torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("cuda.is_available (ROCm usa questa API):", torch.cuda.is_available())

load_dotenv() 
active = os.getenv("USE_GPU", "FALSE").upper() == "TRUE"

if torch.cuda.is_available() and active:
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


print("Selected device:", device)

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)



class FCNN(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = layers

    def forward(self, X):
        X = self.flatten(X)
        logits = self.layers(X)
        return logits


def train_loop(training_data, model, loss_fn, optimizer,batch_size = 32):
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    size = len(train_dataloader.dataset)
    model.train()


    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()

    for batch, (X, Y) in enumerate(tqdm(train_dataloader)):
        X = X.to(device)
        Y = Y.to(device)

        logits = model(X)

        if batch == 0:
            print("Batch on:", X.device, "Logits on:", logits.device)

        loss = loss_fn(logits, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        #     loss_value, current = loss.item(), batch * batch_size + len(X)
        #     print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()


def test_loop(test_data, model, loss_fn,batch_size=32):

    test_dataloader = DataLoader(test_data, batch_size, shuffle=False)
    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



layer = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

model = FCNN(layer).to(device)


print("Model on:", next(model.parameters()).device)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"\n\nEpoch {t+1}\n-------------------------------")
    train_loop(training_data, model, loss_fn, optimizer)

test_loop(test_data, model, loss_fn)

# X, y = next(iter(test_dataloader))
# print(nn.Softmax(dim=1)(model(X[0])))

# def imshow(img):
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()

# prova = DataLoader(test_data, batch_size=32, shuffle=False)
# # get some random training images
# dataiter = iter(prova)
# images, labels = next(dataiter)
# labels
# # show images
# imshow(make_grid(images))