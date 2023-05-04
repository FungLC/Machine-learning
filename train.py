import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from enum import Enum
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 

# Set the paths for the training and validation directories
TRAIN_DIR = "DATA/train"
VAL_DIR = "DATA/val"

# Define the transformations to be applied to the images
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets
train_dataset = ImageFolder(root=TRAIN_DIR, transform=transform_train)
val_dataset = ImageFolder(root=VAL_DIR, transform=transform_val)

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=4)

# Define the ResNet50 model
class ResNet50(ResNet):
    def __init__(self, num_classes=1000, pretrained=True):
        super(ResNet50, self).__init__(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
        if pretrained:
            self.load_state_dict(torch.hub.load_state_dict_from_url(torchvision.models.resnet.model_urls['resnet50'], ResNet50_Weights.DEFAULT.value, progress=True, check_hash=True))

class ResNet50_Weights(Enum):
    DEFAULT = None
    IMAGENET1K_V1 = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

model = ResNet50(num_classes=1000, pretrained=True)
model.fc = nn.Linear(2048, 20)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.00115, momentum=0.9)

# Define the learning rate scheduler
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__=='__main__':
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    num_epochs = 50
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        # Train the model for one epoch
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        train_loss_history.append(epoch_loss)
        # Create a tensor on the GPU
        epoch_acc_gpu = epoch_acc
        epoch_acc_cpu = epoch_acc_gpu.detach().cpu()
        epoch_acc_array = epoch_acc_cpu.numpy()
        train_acc_history.append(epoch_acc_array)

        # Evaluate the model on the validation set
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = running_loss / len(val_dataset)
        val_epoch_acc = running_corrects.double() / len(val_dataset)
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_epoch_loss, val_epoch_acc))
        val_loss_history.append(val_epoch_loss)   
        val_epoch_acc_gpu = val_epoch_acc
        val_epoch_acc_cpu = val_epoch_acc_gpu.detach().cpu()
        val_epoch_acc_array = val_epoch_acc_cpu.numpy()
        val_acc_history.append(val_epoch_acc_array)

    
        
        # Step the learning rate scheduler
        lr_scheduler.step()


    # Plot the accuracy history
    plot_loss(train_loss_history, val_loss_history)
    plt.figure(figsize=(8, 4))
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    torch.save(model, 'CarsModel.pth')