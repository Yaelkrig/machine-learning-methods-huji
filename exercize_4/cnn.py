import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            self.resnet18 = resnet18()
        in_features_dim = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        if probing:
            for name, param in self.resnet18.named_parameters():
                param.requires_grad = False
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):
        features = self.resnet18(x)
        ### YOUR CODE HERE ###
        return self.logistic_regression(features)

def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def compute_accuracy(model, data_loader, device, do_best_worst):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: ret_list: List of booleans indicating if each prediction is correct,
             accuracy: The accuracy of the model on the data in data_loader.
    """

    model.eval()
    ret_list = []  # List to store correctness of each prediction
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions = (predicted == labels).float()  # Get correctness of each prediction
            if do_best_worst:
                ret_list.extend(correct_predictions.cpu().numpy().tolist())  # Append correctness to ret_list
            total += len(labels)
            correct += correct_predictions.sum().item()

    accuracy = correct / total
    return ret_list, accuracy


def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    tot_loss = []
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        ### YOUR CODE HERE ###
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        tot_loss.append(loss.item())
    return np.mean(tot_loss)

# Set the random seed for reproducibility
torch.manual_seed(0)
models = []
### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
model = ResNet18(pretrained=False, probing=False)
models.append(model)
# Linear probing
model = ResNet18(pretrained=True, probing=True)
models.append(model)
# Fine-tuning
model = ResNet18(pretrained=True, probing=False)
models.append(model)

transform = model.transform
batch_size = 32

num_of_epochs = 1
learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
path = 'C:/Users/yaelk/ML/exercize_4/whichfaceisreal/'
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

train_accuracies = np.empty((len(models), len(learning_rates), num_of_epochs))
val_accuracies = np.empty((len(models), len(learning_rates), num_of_epochs))
test_accuracies = np.empty((len(models), len(learning_rates), num_of_epochs))

losses = np.empty((len(models), len(learning_rates), num_of_epochs))

best_preds = []
worst_preds = []

for model_idx, model in enumerate(models):
    for learning_rate_idx, learning_rate in enumerate(learning_rates):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        ### Define the loss function and the optimizer
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        ### Train the model
        best_or_worst = False
        if model_idx == 0 and learning_rate_idx == 3:
            print('here')
            best_or_worst = True
        if model_idx == 2 and learning_rate_idx == 2:
            print('best')
            best_or_worst = True
        # Train the model
        for epoch in range(num_of_epochs):
            print(model_idx, learning_rate_idx, epoch)
            # Run a training epoch
            loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
            # Compute the accuracy
            _, train_acc = compute_accuracy(model, train_loader, device, best_or_worst)
            # Compute the validation accuracy
            _, val_acc = compute_accuracy(model, val_loader, device, best_or_worst)
            # Compute the test accuracy
            test_preds_idx, test_acc = compute_accuracy(model, test_loader, device, best_or_worst)
            print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
            # Stopping condition
            ### YOUR CODE HERE ###
            print("test accuracy is ", test_acc)
            train_accuracies[model_idx][learning_rate_idx][epoch] = train_acc
            val_accuracies[model_idx][learning_rate_idx][epoch] = val_acc
            test_accuracies[model_idx][learning_rate_idx][epoch] = test_acc
            losses[model_idx][learning_rate_idx][epoch] = loss
            if model_idx == 0 and learning_rate_idx == 3:
                print(test_preds_idx)
                worst_preds = [not value for value in test_preds_idx]
            if model_idx == 2 and learning_rate_idx == 2:
                print(test_preds_idx)
                best_preds = test_preds_idx

print(train_accuracies)
print(val_accuracies)
print(test_accuracies)

best_preds = np.array(best_preds)
worst_preds = np.array(worst_preds)

indices = np.where((best_preds == 1) & (worst_preds == 0))[0]
common_5_indices = indices[:5]
print(common_5_indices)

test_loader = get_loaders('C:/Users/yaelk/ML/exercize_4/whichfaceisreal/', transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), 32)[2]
for idx in common_5_indices:
    plt.imshow(test_loader.dataset[idx][0].permute(1, 2, 0))
    plt.show()







