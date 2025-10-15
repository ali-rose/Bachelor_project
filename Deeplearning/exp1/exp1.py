# =======================================================================
# Part 1: Preliminary Experiments (Based on Sections 3.5 and 3.6)
# =======================================================================

# --- Code for Section 3.5: Fashion-MNIST Dataset Introduction ---

import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from IPython import display

# Use svg format to display plots in notebooks for higher quality
d2l.use_svg_display()

# Define a transformation to convert images to tensors
trans = transforms.ToTensor()

# Download and load the Fashion-MNIST training dataset
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data",  # Directory to store the data
    train=True,      # Specify this is the training set
    transform=trans, # Apply the ToTensor transformation
    download=True    # Download if not found in the root directory
)

# Download and load the Fashion-MNIST test dataset
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data",  # Directory to store the data
    train=False,     # Specify this is the test set
    transform=trans, # Apply the ToTensor transformation
    download=True    # Download if not found in the root directory
)

# Display the size of the training and test datasets
print(f"Training set size: {len(mnist_train)}")
print(f"Test set size: {len(mnist_test)}")

# Check the shape of the first image in the training set (Channel, Height, Width)
print(f"Shape of one image: {mnist_train[0][0].shape}")


def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    # Plotting loop
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # If it's a PyTorch tensor
            ax.imshow(img.numpy())
        else:
            # If it's a PIL image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()


# Create a data loader to get a batch of images
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# Display the first 18 images and their corresponding labels
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

# --- Code for Section 3.6: Softmax Regression Implementation from Scratch ---

# Load the data using the d2l library's function
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# Initialize parameters
num_inputs = 784  # 28 * 28 pixels
num_outputs = 10  # 10 classes

# Initialize weights with random normal distribution and biases with zeros
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(X):
    """Softmax operation."""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # Broadcasting is used here


def net(X):
    """The softmax regression model."""
    # Reshape the input image tensor and perform matrix multiplication with weights, then add bias
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    """Cross-entropy loss function."""
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    """Calculate the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # Compare predicted labels with true labels
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """For accumulating sums over n variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    """Evaluate the accuracy of the model on a given dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # (Number of correct predictions, Total number of predictions)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    """The training loop for one epoch."""
    if isinstance(net, torch.nn.Module):
        net.train()  # Set the model to training mode
    # Sum of training loss, sum of training accuracy, number of samples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch's built-in optimizer and loss function
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # Using custom updater
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture parameters
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points to the chart
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """The main training function."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # Assertions to check if the training results are within a reasonable range
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


# Set learning rate
lr = 0.1

def updater(batch_size):
    """Custom updater using stochastic gradient descent."""
    return d2l.sgd([W, b], lr, batch_size)

# Set number of epochs and start training
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net, test_iter, n=6):
    """Predict labels for a few test examples."""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

# Make predictions on the test set
predict_ch3(net, test_iter)


# =======================================================================
# Part 2: Multi-Layer Perceptron (MLP) Experiment
# =======================================================================
import torch
from torch import nn
from d2l import torch as d2l

# --- Base Model and Training Configuration ---

# Define the MLP model using nn.Sequential
# This is the original model for comparison
net_base = nn.Sequential(
    nn.Flatten(),           # Flattens the 28x28 image into a 784x1 vector
    nn.Linear(784, 256),    # First hidden layer
    nn.ReLU(),              # ReLU activation
    nn.Linear(256, 256),    # Second hidden layer
    nn.ReLU(),              # ReLU activation
    nn.Linear(256, 10)      # Output layer
)

def init_weights(m):
    """Initialize weights for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

# Apply the weight initialization to the base model
net_base.apply(init_weights)

# Set hyperparameters
batch_size, lr, num_epochs = 256, 0.1, 20
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net_base.parameters(), lr=lr)

# Load data and train the base model
print("--- Training Base MLP Model ---")
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net_base, train_iter, test_iter, loss, num_epochs, trainer)


# --- Requirement 2: Change number of neurons in hidden layers ---
print("\n--- Training Model with 128, 64, 32 Neurons ---")
net_req2 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)  # The output must be 10 for 10 classes, not 32.
)
net_req2.apply(init_weights)
trainer_req2 = torch.optim.SGD(net_req2.parameters(), lr=0.1)
d2l.train_ch3(net_req2, train_iter, test_iter, loss, 20, trainer_req2)


# --- Requirement 3: Change number of hidden layers ---
print("\n--- Training Model with 4 Layers ---")
net_req3 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 128),  # Added layer
    nn.ReLU(),
    nn.Linear(128, 10)
)
net_req3.apply(init_weights)
trainer_req3 = torch.optim.SGD(net_req3.parameters(), lr=0.1)
d2l.train_ch3(net_req3, train_iter, test_iter, loss, 20, trainer_req3)


# --- Requirement 4: Change learning rate to 0.3 ---
print("\n--- Training Base Model with LR=0.3 ---")
# We reuse the base model structure for this test
net_req4 = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
net_req4.apply(init_weights)
trainer_req4 = torch.optim.SGD(net_req4.parameters(), lr=0.3)
d2l.train_ch3(net_req4, train_iter, test_iter, loss, 20, trainer_req4)


# --- Requirement 5: Change epochs to 25 ---
print("\n--- Training Base Model for 25 Epochs ---")
net_req5 = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
net_req5.apply(init_weights)
trainer_req5 = torch.optim.SGD(net_req5.parameters(), lr=0.1)
d2l.train_ch3(net_req5, train_iter, test_iter, loss, 25, trainer_req5)


# --- Requirement 6: Change activation function to Sigmoid ---
print("\n--- Training Model with Sigmoid Activation ---")
net_req6 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.Sigmoid(),
    nn.Linear(256, 256),
    nn.Sigmoid(),
    nn.Linear(256, 10)
)
net_req6.apply(init_weights)
trainer_req6 = torch.optim.SGD(net_req6.parameters(), lr=0.1)
d2l.train_ch3(net_req6, train_iter, test_iter, loss, 20, trainer_req6)


# --- Requirement 7 & 8: Optimized Model ---
print("\n--- Training Optimized Model ---")

def init_weights_optimized(m):
    """Initialize weights with standard deviation of 0.05."""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.05)

# Use the base model structure
net_optimized = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Apply the new weight initialization
net_optimized.apply(init_weights_optimized)

# Set optimized hyperparameters
batch_size_opt, lr_opt, num_epochs_opt = 256, 0.3, 30
loss_opt = nn.CrossEntropyLoss(reduction='none')
trainer_opt = torch.optim.SGD(net_optimized.parameters(), lr=lr_opt)

# Train the final, optimized model
d2l.train_ch3(net_optimized, train_iter, test_iter, loss_opt, num_epochs_opt, trainer_opt)