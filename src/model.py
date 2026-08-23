"""LeNet-5 architecture implemented from scratch in PyTorch, including a custom RBF layer and loss function
as described in the original 1998 paper by Yann LeCun et al.
The model is designed for grayscale images of size 32x32 pixels and outputs predictions for 10 classes (digits 0-9).
"""

import os

import torch
import torch.nn as nn


# In the 1998 paper the output RBF centers are not learned: each one is a stylized
# 7x12 bitmap of the digit it represents, with +1 for ink and -1 for background.
# 7 * 12 = 84, which is exactly why layer F6 has 84 units. Keeping the templates as
# ASCII art rather than a flat list of numbers means the "each center is a picture of
# its digit" idea stays visible in the source.
DIGIT_GLYPHS = {
    0: ["".join(r) for r in [".......", "..###..", ".#...#.", "#.....#", "#.....#", "#.....#",
                             "#.....#", "#.....#", "#.....#", ".#...#.", "..###..", "......."]],
    1: ["".join(r) for r in [".......", "...#...", "..##...", ".#.#...", "...#...", "...#...",
                             "...#...", "...#...", "...#...", "...#...", ".#####.", "......."]],
    2: ["".join(r) for r in [".......", ".#####.", "#.....#", "......#", ".....#.", "....#..",
                             "...#...", "..#....", ".#.....", "#......", "#######", "......."]],
    3: ["".join(r) for r in [".......", ".#####.", "#.....#", "......#", ".....#.", "..###..",
                             ".....#.", "......#", "......#", "#.....#", ".#####.", "......."]],
    4: ["".join(r) for r in [".......", "....##.", "...#.#.", "..#..#.", ".#...#.", "#....#.",
                             "#######", ".....#.", ".....#.", ".....#.", ".....#.", "......."]],
    5: ["".join(r) for r in [".......", "#######", "#......", "#......", "#####..", ".....#.",
                             "......#", "......#", "......#", "#....#.", ".####..", "......."]],
    6: ["".join(r) for r in [".......", "..####.", ".#.....", "#......", "#......", "#####..",
                             "#....#.", "#.....#", "#.....#", ".#...#.", "..###..", "......."]],
    7: ["".join(r) for r in [".......", "#######", ".....#.", "....#..", "....#..", "...#...",
                             "...#...", "..#....", "..#....", ".#.....", ".#.....", "......."]],
    8: ["".join(r) for r in [".......", "..###..", ".#...#.", "#.....#", ".#...#.", "..###..",
                             ".#...#.", "#.....#", "#.....#", ".#...#.", "..###..", "......."]],
    9: ["".join(r) for r in [".......", "..###..", ".#...#.", "#.....#", "#.....#", ".#...#.",
                             "..####.", "......#", "......#", ".#...#.", "..###..", "......."]],
}

GLYPH_WIDTH, GLYPH_HEIGHT = 7, 12


def build_digit_bitmaps(num_classes=10): # A more visual recommendation by Claude
    """Turns the ASCII glyphs above into a (num_classes, 84) tensor of +1 / -1 values."""
    templates = []
    for digit in range(num_classes):
        rows = DIGIT_GLYPHS[digit]
        assert len(rows) == GLYPH_HEIGHT, f"digit {digit}: expected {GLYPH_HEIGHT} rows, got {len(rows)}"
        for row in rows:
            assert len(row) == GLYPH_WIDTH, f"digit {digit}: row '{row}' is not {GLYPH_WIDTH} wide"
        templates.append([1.0 if pixel == "#" else -1.0 for row in rows for pixel in row])
    return torch.tensor(templates, dtype=torch.float32)


class LeNetRBFSublayer(nn.Module):
    """Output layer of LeNet-5: one Euclidean RBF unit per class.

    Each unit computes the squared distance between the 84-dimensional F6 activation
    and that class's fixed bitmap template. A *small* distance therefore means a
    *confident* prediction, which is why inference uses argmin rather than argmax.
    """

    def __init__(self, in_features=84, num_classes=10):
        super(LeNetRBFSublayer, self).__init__()
        expected = GLYPH_WIDTH * GLYPH_HEIGHT
        assert in_features == expected, f"the paper's RBF centers are {expected}-dimensional, got {in_features}"

        # The centers are constants, not parameters - registering them as a buffer keeps
        # them in the state_dict and moves them with .to(device) while leaving them out
        # of .parameters(), so no optimizer can ever touch them.
        self.register_buffer("centers", build_digit_bitmaps(num_classes))

    def forward(self, x):
        # x shape: [batch_size, 84]
        # self.centers shape: [10, 84]

        # Unsqueeze to align the dimensions for broadcasting
        x_unsqueezed = x.unsqueeze(1)                    # Shape: [batch_size,  1, 84]
        centers_unsqueezed = self.centers.unsqueeze(0)   # Shape: [        1, 10, 84]

        # Squared Euclidean distance from each sample to each of the 10 templates
        distances = torch.sum((x_unsqueezed - centers_unsqueezed) ** 2, dim=2)

        return distances  # Shape: [batch_size, 10]


class LeNet_5(nn.Module):

    """LeNet_5 architecture implemented from scratch.

    Layer Breakdown:

    1. Input: 32x32 pixel grayscale image
    2. C1 (Convolution): 5x5 filters, 6 feature maps, output size 28x28x6
    3. S2 (Subsampling): 2x2 window, stride 2, output size 14x14x6
    4. C3 (Convolution): 5x5 filters, 16 feature maps, output size 10x10x16
    5. S4 (Subsampling): 2x2 window, stride 2, output size 5x5x16
    6. C5 (Convolution): 5x5 filters over a 5x5 input, so it collapses to 120x1x1
    7. F6 (Fully Connected Layer): 84 neurons = the 7x12 RBF template size
    8. Output: 10 Euclidean RBF units returning distances, so predict with argmin

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
        # C5 is written as a convolution, as in the paper. Its 5x5 kernel exactly covers the
        # 5x5 input, so the output is 120x1x1 - mathematically a fully connected layer, but
        # keeping it a Conv2d makes that coincidence explicit.
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        # F6 maps the 120 C5 features onto the 84 dimensions of the RBF templates (7x12).
        self.f6 = nn.Linear(in_features=120, out_features=84)
        # The custom rbf layer implemented from the original paper
        self.rbf_layer = LeNetRBFSublayer(in_features=84, num_classes=10)

    def forward(self, x):
        # We define the Scaled Tanh activation function as the activation function for the convolutional layers, as per the original LeNet-5 paper.
        scaled_tanh = lambda x: 1.7159 * torch.tanh((2 / 3) * x)

        x = scaled_tanh(self.conv1(x))
        x = self.avgpool1(x)

        x = scaled_tanh(self.conv2(x))
        x = self.avgpool2(x)

        # C5 pass + activation. Input is still 4D here: [batch_size, 16, 5, 5]
        x = scaled_tanh(self.c5(x))                # Dimension: [batch_size, 120, 1, 1]

        # Drop the two trailing singleton dimensions so nn.Linear can take over
        x = torch.flatten(x, start_dim=1)          # Dimension: [batch_size, 120]

        # F6 pass + activation. The squashing matters here: it bounds the output to
        # +/-1.7159, the same scale as the +/-1 RBF templates it is compared against.
        x = scaled_tanh(self.f6(x))                # Dimension: [batch_size, 84]

        # RBF pass through (also does the reduction to the 10 classes 0-9, so it kind of also acts as a nn.Linear)
        output = self.rbf_layer(x)

        return output                              # Dimension: [batch_size, 10] of distances

    def _lenet_rbf_loss(self, distances, target_labels):
        """Private method for RBF loss computation. Computes the loss based on the distances between the model's output and the target labels."""
        batch_size = distances.size(0)
        correct_class_distances = distances[torch.arange(batch_size), target_labels]
        # Penalty term: log( sum_j e^-D_j ). Pushing this down forces the distances to the
        # *wrong* classes up, so the loss cannot be minimised by collapsing every distance
        # to zero. The paper adds a positive constant j inside the log as an extra guard;
        # it is omitted here. Note this whole expression is exactly cross-entropy over
        # the negated distances.
        penalty = torch.logsumexp(-distances, dim=1)
        return torch.mean(correct_class_distances + penalty)

    def fit(self, train_loader, epochs=15):
        """Trains the model using the custom RBF loss. Handles the training loop, including
        forward passes, loss computation, backward passes and optimizer steps."""
        self.train()  # Set the model to training mode
        optimizer = torch.optim.SGD(self.parameters(), lr=0.0005)

        for epoch in range(epochs):
            # The learning rate schedule is the one reported in the original LeNet-5 paper:
            # 0.0005 for the first two passes, then 0.0002, 0.0001, 0.00005 and 0.00001.
            if epoch < 2:
                lr = 0.0005
            elif epoch < 5:
                lr = 0.0002
            elif epoch < 8:
                lr = 0.0001
            elif epoch < 12:
                lr = 0.00005
            else:
                lr = 0.00001
            for group in optimizer.param_groups:
                group["lr"] = lr

            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()  # Zero the gradients
                outputs = self.forward(inputs)  # Forward pass
                loss = self._lenet_rbf_loss(outputs, labels)  # Compute the custom RBF loss

                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                running_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f"Epoch: {epoch} | Batch: {batch_idx:03d} | LR: {lr} | Batch Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch} completed | Average Loss: {epoch_loss:.4f}")

    def evaluate(self, test_loader, verbose=False):
        """Evaluates the model on the test dataset and returns the accuracy."""
        self.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.forward(images)
                # argmin, not argmax: the RBF layer returns distances, so smaller is better
                predicted = torch.argmin(outputs, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        if verbose:
            print(f"Total samples evaluated: {total}")
            print(f"Correct predictions: {correct}")

        print(f"Accuracy on the test set: {accuracy:.2f}%")

        return accuracy

    def predict(self, images):
        """Gets a tensor of images and returns the guessed classes"""
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(images)
            predicted_classes = torch.argmin(outputs, dim=1)

        return predicted_classes

    # Saving and loading the model's state dictionary to/from a file for later use.
    # The default path sits next to this file, so it does not depend on the directory
    # the script or notebook happens to be run from.
    DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lenet5_model.pth")

    def save(self, path=None):
        """Saves the model's state dictionary to a file."""
        import os
        path = path or self.DEFAULT_WEIGHTS
        # Ensure the directory exists before saving
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        torch.save(self.state_dict(), path)
        print(f"LeNet_5 model saved to {path}")

    def load(self, path=None, device=None):
        """Loads the model's state dictionary from a file."""
        path = path or self.DEFAULT_WEIGHTS
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device) # In the case a user inputs a device of choice

        # Load the weights on the specified device (CPU, GPU, etc.)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        self.to(device)

        print(f"LeNet_5 model loaded from {path} onto {device}")
