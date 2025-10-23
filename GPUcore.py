# import torch
# import torchvision
# import torchvision.transforms as transforms

# # Check GPU (RTX 2050)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")  # Should say 'cuda'

# # Load a tiny batch of MNIST data
# train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)  # Small batch for 4GB VRAM

# # Super simple model
# model = torch.nn.Sequential(
#     torch.nn.Conv2d(1, 4, kernel_size=3),  # Small CNN layer
#     torch.nn.ReLU(),
#     torch.nn.Flatten(),
#     torch.nn.Linear(4 * 26 * 26, 10)      # Output for 10 digits
# ).to(device)  # Move to GPU

# # One training step
# for images, labels in train_loader:
#     images, labels = images.to(device), labels.to(device)  # Data to GPU
#     outputs = model(images)  # GPU cores compute convolutions
#     break  # Just one batch for demo
# print("Done! Your RTX 2050's cores and threads handled the math.")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")