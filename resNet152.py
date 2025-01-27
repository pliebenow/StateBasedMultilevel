import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        
        # First 1x1 convolution to reduce dimensionality
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        
        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        
        # Third 1x1 convolution to restore dimensionality
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection, adjust dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # First 1x1 convolution + activation
        out = F.relu(self.bn2(self.conv2(out)))  # Second 3x3 convolution + activation
        out = self.bn3(self.conv3(out))  # Third 1x1 convolution
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)  # Final activation
        return out

class ResNet152(nn.Module):

    def __init__(self, num_blocks_layer1 = 3,  num_blocks_layer2= 8, num_blocks_layer3=36, num_blocks_layer4=3, num_classes=100):
        super(ResNet152, self).__init__()

        self.intermediate_outputs = []
        
        # Initial Convolution and Max-Pooling (Adjusted for 32x32 images)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Modify the layer architecture based on CIFAR-100
        self.layer1 = self._make_layer(64, 64, num_blocks_layer1, stride=1)   # 64 channels, 3 blocks
        self.layer2 = self._make_layer(64, 128, num_blocks_layer2, stride=2)  # 128 channels, 8 blocks
        self.layer3 = self._make_layer(128, 256, num_blocks_layer3, stride=2)  # 256 channels, 36 blocks
        self.layer4 = self._make_layer(256, 512, num_blocks_layer4, stride=2)  # 512 channels, 3 blocks
        
        # Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)  # Output layer for CIFAR-100 classes

    def install_hook(layer_number, block_number):
        #model.layer2[0].register_forward_hook(hook_fn)
        pass

    # Define a forward hook
    def forward_hook(self, module, input, output):
        #print(f"Hooked {module.__class__.__name__} with input shape {input[0].shape}, output shape {output.shape}")
        self.intermediate_outputs.append(output)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Initial conv layer + batch norm + relu
        x = self.layer1(x)  # First residual layer
        x = self.layer2(x)  # Second residual layer
        x = self.layer3(x)  # Third residual layer
        x = self.layer4(x)  # Fourth residual layer
        x = self.avgpool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.fc(x)  # Final fully connected layer
        return x


random_input = torch.randn(1, 3, 28, 28)
model = ResNet152()

# List to store intermediate outputs
intermediate_outputs = []

    
#for name, layer in model.named_modules():
#    print(name, layer)# Install hooks on each Bottleneck block

# Perform a forward pass
output = model(random_input)

# Output final results
print("Final output shape:", output.shape)
print(f"Number of hooked outputs: {len(intermediate_outputs)}")
#print(intermediate_outputs[0].shape)
#print(intermediate_outputs[1].shape)
#print(intermediate_outputs[2].shape)
#print(intermediate_outputs[3].shape)

