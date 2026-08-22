"""LeNet-5 architecture implemented from scratch in PyTorch, including a custom RBF layer and loss function 
as described in the original 1998 paper by Yann LeCun et al. 
The model is designed for grayscale images of size 32x32 pixels and outputs predictions for 10 classes (digits 0-9).
"""

import torch
import torch.nn as nn


class LeNetRBFSublayer(nn.Module):
    def __init__(self, in_features=84, num_classes=10):
        super(LeNetRBFSublayer, self).__init__()
        # 1. Create the fixed 7x12 bitmap templates for digits 0-9
        # In the 1998 paper, these weights were manually hardcoded and frozen
        # We initialize them here as standard parameters (shape: 10 classes, 84 features)
        self.centers = nn.Parameter(torch.randn(num_classes, in_features), requires_grad=False)

    def forward(self, x):
        # x shape: [batch_size, 84]
        # self.centers shape: [10, 84]

        # 2. Compute the Euclidean distance between the input and each center
        # Unsqueeze to align the dimensions for broadcasting
        x_unsqueezed = x.unsqueeze(1) # Shape: [batch_size, 1, 84]
        centers_unsqueezed = self.centers.unsqueeze(0) # Shape: [1, 10, 120]

        # Calculate squared Euclidean distance
        distances = torch.sum((x_unsqueezed - centers_unsqueezed) ** 2, dim=2)

        return distances # Shape: [batch_size, 10]

class LeNet_5(nn.Module):

    """LeNet_5 architecture implemented from scratch
    
    Layer Breakdown:

    1. Input: 32x32 pixel grayscale image
    2. C1 (Convolution): 5x5 filters, 6 feature maps, output size 28x28x6
    3. S2 (Average Pooling): 2x2 window, stride 2, output size 14x14x6
    4. C3 (Convolution): 5x5 filters, 16 feature maps, partial connections, output size 10x10x16
    5. S4 (Average Pooling): 2x2 window, stride 2, output size 5x5x16
    6. C5 (Fully Connected Conv): 120 neurons
    7. F6 (Fully Connected Layer): 84 neurons
    8. Output: Softmax with 10 classes

    Core Concepts:
    - Local Receptive Fields: Neurons look at small local patches of input
    - Weight Sharing: Filters use the same weights across the entire image to detect identical features anywhere
    - Subsampling: Pooling shrinks spatial size and adds shift invariance
    """

    def __init__(self) -> None:
        super().__init__()
        # in_channels=1 because the img. is grayscale; out_channels=6 because we want to obtain 6 feature maps at exit (6 different filters); kernel_size = 5 because the dimension of every filter is 5x5 pixels; stride = 1 because the filter is moving pixel by pixel; padding = 0 because we don't add zeroed pixels on the edges of the images (valid convolution operation)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d((2, 2), stride=2)
        # in_channels=6 as per the output channels of the first convolutional layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2d((2, 2), stride=2)
        # We will write this the modern way as a basic nn.Linear layer with 120 neurons;
        # Output size of previous pooling was 5x5x16 (which equals exactly 400 total values)
        # We map these 400 flatenned inputs to 120 independent neurons.
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        # we will write this the modern way as a basic nn.Linear layer with 84 neurons; Output size of previous convolution was 1x1x120 (which equals exactly 120 total values); We map these 120 flatenned inputs to 84 independent neurons.
        self.f6 = nn.Linear(in_features=120, out_features=84)
        # The custom rbf layer implemented from the original paper
        self.rbf_layer = LeNetRBFSublayer(in_features=120, num_classes=10)
# Observation: nn.Linear layers don't know how to work with 2D/3D structures (channels, height, weight)
    def forward(self, x):
        # We define the Scaled Tanh activation function as the activation function for the convolutional layers, as per the original LeNet-5 paper.
        scaled_tanh = lambda x: 1.7159 * torch.tanh((2/3) * x)

        x = scaled_tanh(self.conv1(x))
        x = self.avgpool1(x)

        x = scaled_tanh(self.conv2(x))
        x = self.avgpool2(x)

        x = torch.flatten(x, start_dim=1)

        # C5 (convolutional layer) pass + Activation Function
        x = scaled_tanh(self.c5(x)) # Dimension: [batch_size, 120]
        # F6 (fully connected layer) pass + Activation Function
        x = torch.flatten(x, start_dim=1) # Dimension: [batch_size, 120] - flatten the (batch_size, 1, 1) to (batch_size, 120) for the next layer
        # Apply fully connected layer and activation function
        x = self.f6(x)
        # RBF pass through (also does the reduction to the 10 classes 0-9, so it kind of also acts as a nn.Linear)
        output = self.rbf_layer(x)

        return output

    def _lenet_rbf_loss(self, distances, target_labels):
        """Private method for RBF loss computation. Computes the loss based on the distances between the model's output and the target labels."""
        batch_size = distances.size(0)
        correct_class_distances = distances[torch.arange(batch_size), target_labels]
        penalty = torch.logsumexp(-distances, dim=1)
        return torch.mean(correct_class_distances + penalty)

    """Implement a fit method to train the model using the custom RBF loss function. This method will handle the training loop, including forward passes, loss computation, backward passes, and optimizer steps."""
    def fit(self, train_loader, epochs=15, lr=0.05):
        self.train()  # Set the model to training mode
        for epoch in range(epochs):
            """The learning rate schedule is based on the original LeNet-5 paper, which suggests a decreasing learning rate over epochs to ensure convergence."""
            if epoch < 3:
                lr = 0.05
            elif epoch < 9:
                lr = 0.01
            elif epoch < 13:
                lr = 0.005
            elif epoch < 17:
                lr = 0.001
            elif epoch < 21:
                lr = 0.0005
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)

            running_loss = 0.0
            current_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()  # Zero the gradients
                outputs = self.forward(inputs)  # Forward pass
                loss = self._lenet_rbf_loss(outputs, labels)  # Compute the custom RBF loss

                loss.backward() # Backward pass
                optimizer.step()  # Update weights

                running_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f"Epoch: {epoch} | Batch: {batch_idx:03d} | Batch Loss: {loss.item():.4f}")

                epoch_loss = running_loss / len(train_loader)
                print(f"Epoch {epoch} completed | Average Loss: {epoch_loss:.4f}")

    def evaluate(self, test_loader, correct=None, total=None):
        """Evaluates the model on the test dataset and returns the accuracy."""
        self.eval() # Set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.forward(images)
                predicted = torch.argmin(outputs, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        if total == True:
            print(f"Total samples evaluated: {total}")
        if correct == True:
            print(f"Correct predictions: {correct}")

        print(f"Accuracy on the test set: {accuracy:.2f}%")

        return accuracy

    def predict(self, images):
        """Gets a tensor of images and returns the guessed classes"""
        self.eval() # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(images)
            predicted_classes = torch.argmin(outputs, dim=1)

        return predicted_classes

    """Saving and loading the model's state dictionary to/from a file for later use """
    def save(self):
        """Saves the model's state dictionary to a file."""
        torch.save(self.state_dict(), "lenet5_model.pth")
        print("LeNet_5 model saved to lenet5_model.pth")

    def load(self):
        """Loads the model's state dictionary from a file."""
        self.load_state_dict(torch.load("lenet5_model.pth"))
        print("LeNet_5 model loaded from lenet5_model.pth")