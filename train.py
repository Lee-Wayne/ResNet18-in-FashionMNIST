import os, time, random
import logging
import numpy as np
from tqdm import tqdm
import torch
import torchvision.models as models
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        self.resnet = models.resnet18(pretrained=True)
        in_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_ftrs, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        return x
    

def Acc_fn(model, inputs, labels):
    _, predicted = torch.max(model(inputs).data, 1)
    return (predicted == labels).sum().item()


def Train(model, criterion, optimizer, device, num_epochs, train_loader, test_loader):
    best_acc = 0.
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(num_epochs):
        ts = time.time()

        # train
        model.train()
        train_acc = 0.
        train_loss = 0.
        train_count = 0
        for images, labels in tqdm(train_loader, total=len(train_loader)):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_count += images.shape[0]
            train_acc += Acc_fn(model, images, labels)
            train_loss += loss.item() * images.shape[0]
        
        train_acc /= float(train_count)
        train_loss /= float(train_count)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # test
        model.eval()
        test_acc = 0.
        test_loss = 0.
        test_count = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, total=len(test_loader)):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_count += images.shape[0]
                test_acc += Acc_fn(model, images, labels)
                test_loss += loss.item() * images.shape[0]
        
        test_acc /= float(test_count)
        test_loss /= float(test_count)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        time_cost = time.time() - ts
        logging.info(
            'Epoch [{}/{}]: Train Loss: {:.3f} | Train Acc: {:.3f} | Test Loss: {:.3f}, Test Acc: {:.3f} || time: {:.1f}'.format(
                epoch+1, num_epochs, train_loss, train_acc, test_loss, test_acc, time_cost))
        
        if best_acc < test_acc:
            best_acc = test_acc
            logging.info(
                'Save model at Epoch [{}/{}] | Test Acc {:.3f}'.format(
                    epoch+1, num_epochs, test_acc))
            
            save_model = model
            state = {
                'model': save_model.state_dict(),
                'history': history
            }
            torch.save(state, 'ResNet18_best.pkl')
            
        scheduler.step()
    
    return history


logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ResNet18_bestacc.log')
    ]
)

if __name__ == '__main__':
    
    # Hyper-parameters
    num_epochs = 20
    num_classes = 10
    lr = 0.01
    batch_size = 256
    gpu_order = '0'
    torch_seed = 2

    # Device configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_order
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    np.random.seed(torch_seed)
    random.seed(torch_seed)
    
    cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # history record
    history = {'train_loss': [], 'test_loss': [],
               'train_acc': [], 'test_acc': []}
    
    # Data Set
    trans = transforms.ToTensor()
    train_dataset = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=False)
    test_dataset = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=False)

    # Data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load pre-trained model
    model = ConvNet(num_classes)
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Train the model
    Train(model, criterion, optimizer, device, num_epochs, train_loader, test_loader)
