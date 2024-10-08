import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from helpers import *


def train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256, grad_layers=[]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)
    # grad_layers = [0, 30, 60, 90, 95, 99]
    trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(),
                                              torch.tensor(train_data['country'].values).long())
    valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(),
                                            torch.tensor(val_data['country'].values).long())
    testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(),
                                             torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []
    grad_magn = {layer: [] for layer in grad_layers}

    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.

        ep_grads = {layer: [] for layer in grad_layers}

        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            #### YOUR CODE HERE ####

            # perform a training iteration

            # move the inputs and labels to the device
            # zero the gradients
            # forward pass
            # calculate the loss
            # backward pass
            # update the weights

            # name the model outputs "outputs"
            # and the loss "loss"

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            layer_idx = 0
            for _, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    if layer_idx in grad_layers:
                        grad_magnitude = torch.norm(m.weight.grad) ** 2
                        ep_grads[layer_idx].append(grad_magnitude.item())
                    layer_idx += 1

            optimizer.step()
            #### END OF YOUR CODE ####

            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()

        for layer in grad_layers:
            if len(ep_grads[layer]) > 0:
                print(grad_magn[layer])
                grad_magn[layer].append(np.mean(ep_grads[layer]))

        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    #### YOUR CODE HERE ####

                    # perform an evaluation iteration

                    # move the inputs and labels to the device
                    # forward pass
                    # calculate the loss

                    # name the model outputs "outputs"
                    # and the loss "loss"
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    #### END OF YOUR CODE ####

                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1],
                                                                                       test_accs[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, grad_magn


def train_sin_data(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    print(train_data.shape)
    trainset = torch.utils.data.TensorDataset(torch.tensor(
        train_data[['long_sin1', 'lat_sin1', 'long_sin2', 'lat_sin2', 'long_sin3', 'lat_sin3', 'long_sin4', 'lat_sin4',
                    'long_sin5', 'lat_sin5', 'long_sin6', 'lat_sin6', 'long_sin7', 'lat_sin7', 'long_sin8', 'lat_sin8',
                    'long_sin9', 'lat_sin9', 'long_sin10', 'lat_sin10']].values).float().to(device),
                                              torch.tensor(train_data['country'].values).long().to(device))
    print(val_data.shape)
    valset = torch.utils.data.TensorDataset(torch.tensor(
        val_data[['long_sin1', 'lat_sin1', 'long_sin2', 'lat_sin2', 'long_sin3', 'lat_sin3', 'long_sin4', 'lat_sin4',
                  'long_sin5', 'lat_sin5', 'long_sin6', 'lat_sin6', 'long_sin7', 'lat_sin7', 'long_sin8', 'lat_sin8',
                  'long_sin9', 'lat_sin9', 'long_sin10', 'lat_sin10']].values).float().to(device),
                                            torch.tensor(val_data['country'].values).long().to(device))

    testset = torch.utils.data.TensorDataset(torch.tensor(
        test_data[['long_sin1', 'lat_sin1', 'long_sin2', 'lat_sin2', 'long_sin3', 'lat_sin3', 'long_sin4', 'lat_sin4',
                   'long_sin5', 'lat_sin5', 'long_sin6', 'lat_sin6', 'long_sin7', 'lat_sin7', 'long_sin8', 'lat_sin8',
                   'long_sin9', 'lat_sin9', 'long_sin10', 'lat_sin10']].values).float().to(device),
                                             torch.tensor(test_data['country'].values).long().to(device))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.

        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()

        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    #### END OF YOUR CODE ####

                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1],
                                                                                       test_accs[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses


if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    output_dim = len(train_data['country'].unique())
    model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    model = nn.Sequential(*model)

    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, grad_magn = \
        train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256)

    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Losses')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Accs.')
    plt.legend()
    plt.show()

    plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values,
                             'Decision Boundaries', implicit_repr=False)
