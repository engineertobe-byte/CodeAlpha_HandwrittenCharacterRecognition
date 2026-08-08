"""
Inference script to evaluate the trained CNN model on test data.
Demonstrates model accuracy assessment (as requested in ML guidelines).
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from .model import CharacterCNN


def evaluate_model():
    print("🔍 Evaluating Model on Local Test Set...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharacterCNN().to(device)
    
    # Load pre-trained weights
    model_path = Path("models/cnn_mnist.pth")
    if not model_path.exists():
        print("❌ Error: Model weights not found. Run 'python -m src.train' first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = 100 * correct / total
    print(f"✅ Model Accuracy on 10,000 test images: {accuracy:.2f}%")


if __name__ == "__main__":
    evaluate_model()
