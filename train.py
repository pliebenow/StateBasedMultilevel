
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import data 
from resNet152 import ResNet152
from load_cifar import load_data


# Training function
def train(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(trainloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# Test function
def test(model, testloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(testloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


#hardware settings
torch.cuda.empty_cache()  # Frees unused cached memory
torch.cuda.memory_summary(device=None, abbreviated=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load data
trainloader, testloader = load_data(cifar10=True,cifar100=False)

# instantiate the model, define loss function and optimizer
model = ResNet152(num_blocks_layer1 = 3,  num_blocks_layer2= 8, num_blocks_layer3=36, num_blocks_layer4=3)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# scheduler to reduce learning rate when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5, verbose=True
)

criterion = nn.CrossEntropyLoss()
torch.backends.cudnn.benchmark = True

model.to(device)

#parameters
num_epochs = 50
running_loss=0
total = 0
correct=0
for epoch in range(num_epochs):
    avg_loss_train, accuracy_train = train(model, trainloader, criterion, optimizer, device)
    avg_loss_test, accuracy_test = test(model, testloader, criterion, device)
    scheduler.step(avg_loss_test)

    print(f"epoch {epoch}, test loss: {avg_loss_test}, accuracy_test: {accuracy_test}")