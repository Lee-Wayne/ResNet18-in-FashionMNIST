import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt


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


trans = transforms.ToTensor()
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

test_loader = data.DataLoader(mnist_test, batch_size=256, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model
model = ConvNet(num_classes=10)
model = model.to(device)
checkpoint = torch.load('ResNet50_best.pkl')
model.load_state_dict(checkpoint['model'])
history = checkpoint['history']

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


# Plot
plt.figure()
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['train_acc'], label='train_acc', linestyle='--')
plt.plot(history['test_acc'], label='test_acc', linestyle='--')
plt.xticks(range(0, len(history['train_loss'])+1, 4))
plt.xlabel('epoch')
plt.ylim((0.0, 1.0))
plt.title('ResNet18 FashionMNIST')
plt.legend()
plt.savefig('result.png')

plt.show()

