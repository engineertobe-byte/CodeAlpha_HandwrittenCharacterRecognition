"""
Training script for the CNN model using the MNIST dataset.
Executes entirely locally.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from .model import CharacterCNN


def train_model(epochs: int = 3):
    print("🚀 Starting Local Training Pipeline...")
    
    # 1. Data Preprocessing & Loading (Local storage)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download to ./data if not present, strictly local execution afterwards
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 2. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharacterCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0
                
    # 4. Save Model Locally
    save_path = Path("models/cnn_mnist.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training complete. Model saved locally to {save_path}")


if __name__ == "__main__":
    train_model(epochs=2) # Kept low for quick portfolio demonstration
