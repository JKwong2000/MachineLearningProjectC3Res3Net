import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 128
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        downsample_x = self.downsample(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(downsample_x + out)


class C3Res3Net(nn.Module):
    def __init__(self):
        super(C3Res3Net, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=(3,3),stride=1,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool_1=nn.AvgPool2d(2)
        self.layer_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool_2 = nn.AvgPool2d(2)
        self.layer_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool_3 = nn.AvgPool2d(2)
        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc1 = nn.Linear(128, 64)

        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.pool_1(out)
        out = self.layer_2(out)
        out = self.pool_2(out)
        out = self.layer_3(out)
        out = self.pool_3(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(self.dropout1(out))
        out = self.fc2(out)
        return out


model = C3Res3Net().to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    return train_loss


def test(dataloader, model, loss_fn, temp_correct):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if correct > temp_correct:
        temp_correct = correct
        torch.save(model.state_dict(), "best_model.pth")
    print(f"Test : \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return temp_correct, test_loss, correct


epochs = 100
temp_correct = 0
train_loss = []
test_loss = []
Accuracy = []
start=time.time()
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    temp_train_loss = train(train_dataloader, model, loss_fn, optimizer)
    temp_correct, temp_test_loss, temp_correct = test(test_dataloader, model, loss_fn, temp_correct)

    train_loss.append(temp_train_loss)
    test_loss.append(temp_test_loss)
    Accuracy.append(temp_correct)
end = time.time()
print("Done! Use {:.2f} seconds".format(end-start))
epoch=[i+1 for i in range(len(train_loss))]
df = pd.DataFrame({"epoch":epoch, "train_loss":train_loss, "test_loss":test_loss,"Accuracy":Accuracy})
df.to_csv("history.csv")

torch.save(model.state_dict(), "final_model.pth")
print("Save PyTorch Model State to final_model.pth")
