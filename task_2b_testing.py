# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define data transformations (modify as needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset_path = 'c://Users//HP//Downloads//training'

from torchvision.datasets import ImageFolder
train_dataset = ImageFolder(root=dataset_path, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Load the pre-trained VGG-16 model
model = vgg16(pretrained=True)

# Modify the model for your specific number of classes
num_classes = 5  # Change this to your actual number of classes
model.classifier[6] = nn.Linear(4096, num_classes)

# Send the model to the GPU (or CPU)
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10  # Change this as needed
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    class_labels = ["fire", "combat", "humanitarianaid", "militaryvehicles", "destroyedbuilding"]  

# Save the trained model
model_save_path = 'my_pytorch_trained_model.pth'  # Specify the path where you want to save the model
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'class_labels': class_labels,  # Save your class labels for reference
}, model_save_path)

print("Model saved to:", model_save_path)



















