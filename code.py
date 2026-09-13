import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_set =CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set =CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader =DataLoader(train_set, batch_size=64, shuffle=True)
test_loader =DataLoader(test_set, batch_size=64, shuffle=False)

#Building CNN
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layers = nn.Sequential(
        nn.Conv2d(3,32,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Conv2d(32,64,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Conv2d(64,128,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.fc_layers = nn.Sequential(
        nn.Linear(4*4*128,256),
        nn.ReLU(),
        nn.Linear(256,10)
    )
  def forward(self,x):
    x=self.conv_layers(x)
    x=x.view(x.size(0), -1) # Corrected flattening
    x=self.fc_layers(x)
    return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters())

#Training
epochs= 10
for epoch in range(epochs):
  epoch_training_loss =0.0
  for images,labels in train_loader:
    optimizer.zero_grad()
    outputs = model.forward(images)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    epoch_training_loss +=loss.item()
  print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_training_loss/len(train_loader)}")

#evaluation\
correct_labels=0
total_labels=0
model.eval()
with torch.no_grad():
  for images,labels in test_loader:
    outputs = model.forward(images)
    _,predicted = torch.max(outputs.data,1)
    total_labels +=labels.size(0)
    correct_labels +=(predicted==labels).sum().item()
accuracy =correct_labels/total_labels
