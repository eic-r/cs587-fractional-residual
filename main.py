import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import FractionalResNet18
import time

# Set quick flushing for slurm output
sys.stdout.reconfigure(line_buffering=True, write_through=True)

def train(model, trainloader, testloader, optimizer, criterion, device, epochs, file):
    model.train()
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Evaluate test accuracy each epoch
        test_acc = test(model, testloader, device)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.3f}, "
              f"Train Acc={epoch_acc:.2f}%, Test Acc={test_acc:.2f}%")
        
        update_plots(range(1, epoch+2), train_losses, train_accuracies, test_accuracies, file)

def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100. * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    return test_acc

def update_plots(epochs, train_losses, train_accuracies, test_accuracies, file):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(epochs, train_losses, label='Train Loss', color='red')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, train_accuracies, label='Train Acc', color='blue')
    ax2.plot(epochs, test_accuracies, label='Test Acc', color='green')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"plots/{file}.png")
    plt.close(fig)  # Free up memory

def main(*ARGS):
    R"""
    Main.
    """
    #
    parser = argparse.ArgumentParser(description="Main Execution")
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=100,
        help="Batch size.",
    )
    parser.add_argument(
        "--method",
        required=False,
        default="base",
        help="Gradient method: base, GL.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=1,
        help="Derivative alpha.",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        required=False,
        default=20,
        help="Max number of prior gradients stored.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=1e-2,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        required=False,
        default=20,
        help="Number of training epochs.",
    )
    
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    # Parse the command line arguments.
    batch_size = args.batch_size
    method = args.method
    alpha = args.alpha
    max_history = args.max_history
    lr = args.lr
    num_epochs = args.num_epochs

    transform = transforms.Compose([
        transforms.Resize(224),            # Resize to ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("plots", exist_ok=True)

    file = f"{method}_a{alpha}".replace('.', '_')

    if method == "base":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif method == "GL":
        model = FractionalResNet18(alpha=alpha, max_history=max_history)
    else:
        raise ValueError(f"Unknown method: {method}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"\n--- Training with {method} α={alpha} ---")
    run_start = time.time()
    train(model, trainloader, testloader, optimizer, criterion, device, num_epochs, file)
    run_end = time.time()
    print(f"Total training time: {(run_end-run_start):.2f}s")

#
if __name__ == "__main__":
    #
    main()
