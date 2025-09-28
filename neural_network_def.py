import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Pick GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----Define a simple neural network -----
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self). __init__()
        # Flatten 28by28=784 -> input
        self.fc1 = nn.Linear(28*28, 128) #1ST fully connected layer
        self.fc2 = nn.Linear(128,64) # 2ND fully connected layer
        self.fc3 = nn.Linear(64, 10) # Output layer(10 classes for digits)
        
    def forward(self,x):
        # x shape: (batch_size,1,28,28)
        x = x.view(-1,28*28) # flatten into (batch_size, 784)
        x = F.relu(self.fc1(x)) # hidden layer 1 with ReLU
        x = F.relu(self.fc2(x)) # hidden layer 2 with ReLU
        x = self.fc3(x) # Output logits
        return x

if __name__ == "__main__":
    model = SimpleNN()
    print(model) # Print the model architecture
    
    # Fake batch of 64 MNIST-like images
    dummy_input = torch.randn(64, 1, 28, 28)
    output = model(dummy_input)
    print("Output shape:",output.shape)
    
# Transform: convert to tensor and normalize(0 - 1)
transform = transforms.ToTensor()

# Download + load MNIST
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = SimpleNN()

# Loss: CrossEntropyLoss (good for classification problems)
criterion = nn.CrossEntropyLoss()

# Optimizer: Stochastic Gradient Descent(SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# Trach history
train_losses = []
test_accuracies = []

# ---- Training loop (1 epoch)
EPOCHS = 5 # number of full passes over the dataset
for epoch in range(EPOCHS):
    model.train() # set model to training mode
    running_loss = 0.0
        
    for batch_idx,(data, targets) in enumerate(train_loader):
        data = data.view(data.shape[0], -1).to(device)
        targets = targets.to(device) 
            
            
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
            
            
        running_loss += loss.item()
            
    
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")
    train_losses.append(avg_loss) # store training loss
        
        
    # Testing 
    model.eval()
    correct = 0
    total = 0
     
        
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.view(data.shape[0], -1).to(device)
            targets = targets.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} Test Accuracy: {accuracy:.2f} %")
    test_accuracies.append(accuracy) # store test accuracy


# Training Loss
plt.subplot(1,2,1)
plt.plot(train_losses,marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")


# Test Accuracy
plt.subplot(1,2,1)
plt.plot(test_accuracies,marker='o')
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.show()


# ------------ Visualize Predictions ---------------
model.eval()
examples = enumerate(test_loader)
batch_idx, (example_data,example_targets) = next(examples)


with torch.no_grad():
    example_data = example_data.to(device)
    output = model(example_data.view(example_data.shape[0], -1))
    
    
# Plot first 6 test images with predictions
fig = plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i].cpu().squeeze(), cmap="gray", interpolation="none")
    plt.title(f"Pred {output.data.max(1, keepdim=True)[1][i].item()} | True: {example_targets[i].item()}")
    plt.xticks([])
    plt.yticks([])
   
    
plt.show()
    
        