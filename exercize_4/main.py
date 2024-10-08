import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import helpers
from helpers import *

from NN_tutorial import train_model, train_sin_data

def plot_lr(val_loss, x_ax):
    print(val_loss)
    plt.figure()
    plt.plot(x_ax, val_loss[0], label='lr 1', color='red')
    plt.plot(x_ax, val_loss[1], label='lr 0.01', color='blue')
    plt.plot(x_ax, val_loss[2], label='lr 0.001', color='green')
    plt.plot(x_ax, val_loss[3], label='lr 0.00001', color='orange')

    plt.title('Validation Losses of different learning rate values')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.show()

def plot_ep(val_loss, x_ax):
    print(x_ax)
    print(val_loss)

    values_to_plot = [val_loss[i-1] for i in x_ax]
    plt.figure()
    plt.plot(x_ax, values_to_plot, color='red')

    plt.title('Validation Loss over different epochs')
    # plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.show()

def plot_batch(val_loss_reg, val_loss_batch, x_ax):
    plt.figure()
    plt.plot(x_ax, val_loss_reg, label='not normalized', color='red')
    plt.plot(x_ax, val_loss_batch, label='normalized', color='blue')

    plt.title('Normalized vs Unnormalized')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.show()

def plot_losses(train_losses, val_losses, test_losses):
    epochs = len(train_losses)
    plt.plot(np.arange(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(np.arange(1, epochs + 1), val_losses, label='Validation Loss')
    plt.plot(np.arange(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Losses')
    plt.legend()
    plt.show()


def MLP_task_lr(train_data, val_data, test_data):
    learning_rates = [1., 0.01, 0.001, 0.00001]
    epochs = 50
    train_acc_lr = np.empty((len(learning_rates), epochs))
    valid_acc_lr = np.empty((len(learning_rates), epochs))
    test_acc_lr = np.empty((len(learning_rates), epochs))

    train_loss_lr = np.empty((len(learning_rates), epochs))
    valid_loss_lr = np.empty((len(learning_rates), epochs))
    test_loss_lr = np.empty((len(learning_rates), epochs))

    for lr_idx in range(len(learning_rates)):
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
            train_model(train_data, val_data, test_data, model, lr=learning_rates[lr_idx], epochs=epochs, batch_size=256)
        train_acc_lr[lr_idx] = train_accs
        valid_acc_lr[lr_idx] = val_accs
        test_acc_lr[lr_idx] = test_accs

        train_loss_lr[lr_idx] = train_losses
        valid_loss_lr[lr_idx] = val_losses
        test_loss_lr[lr_idx] = test_losses
    return valid_loss_lr

def MLP_task_ep(train_data, val_data, test_data, num_of_epochs):
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
    train_model(train_data, val_data, test_data, model, lr=0.001, epochs=num_of_epochs, batch_size=256)

    return val_losses

def MLP_batch_norm(train_data, val_data, test_data):
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
    model, reg_train_accs, reg_val_accs, reg_test_accs, reg_train_losses, reg_val_losses, reg_test_losses, grad_magn = \
        train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256)

    batch_model = [nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),   # hidden layer 2
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),   # hidden layer 3
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),   # hidden layer 4
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),   # hidden layer 5
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),   # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    batch_model = nn.Sequential(*batch_model)
    batch_model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, grad_magn = \
        train_model(train_data, val_data, test_data, batch_model, lr=0.001, epochs=50, batch_size=256)
    x_ax_first = [i for i in range(1, 51)]
    plot_batch(reg_val_losses, val_losses, x_ax_first)
    plot_batch(reg_val_accs, val_accs, x_ax_first)



def MLP_batch_size(train_data, val_data, test_data):
    batch_sizes = [1, 16, 128, 1024]
    epochs = [1, 10, 50, 50]

    train_acc_lr = np.empty((len(epochs)))
    valid_acc_lr = np.empty((len(epochs)))
    test_acc_lr = np.empty((len(epochs)))

    train_loss_lr = np.empty((len(epochs)))
    valid_loss_lr = np.empty((len(epochs)))
    test_loss_lr = np.empty((len(epochs)))
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    for idx in range(len(batch_sizes)):
        batch_size = batch_sizes[idx]
        epoch = epochs[idx]
        print("number of epochs ", epoch)
        print("batch-size is ", batch_size)
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
                train_model(train_data, val_data, test_data, model, lr=0.001, epochs=epoch, batch_size=batch_size)
        train_acc_lr[idx] = train_accs[-1]
        valid_acc_lr[idx] = val_accs[-1]
        test_acc_lr[idx] = test_accs[-1]

        train_loss_lr[idx] = train_losses[-1]
        valid_loss_lr[idx] = val_losses[-1]
        test_loss_lr[idx] = test_losses[-1]
        axs[0].plot(range(1, epoch + 1), val_losses, label=f'Batch Size: {batch_size}', marker='o')
        axs[1].plot(range(1, epoch + 1), val_accs, label=f'Batch Size: {batch_size}', marker='o')

    axs[0].set_title('Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].set_title('Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return valid_loss_lr

def plot_6_2_1_3_4(y_ax, train_acc, valid_acc, test_acc, title):
    print(y_ax)
    print(train_acc)
    print(valid_acc)
    print(test_acc)
    plt.plot(y_ax, train_acc, label='Training Accuracy')
    plt.plot(y_ax, valid_acc, label='Validation Accuracy')
    plt.plot(y_ax, test_acc, label='Test Accuracy')
    plt.xlabel(title)
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. ' + title)
    plt.legend()
    plt.show()


def eval_MLP(train_data, val_data, test_data):
    depths = [1, 2, 6, 10, 6, 6, 6]
    widths = [16, 16, 16, 16, 8, 32, 64]
    epochs = [50, 50, 100, 100, 100, 100, 100]
    models = []

    best_val_acc = 0
    best_model = None
    best_width = widths[0]
    best_depth = depths[0]
    best_idx = 0

    worst_val_acc = 100000000
    worst_model = None
    worst_width = widths[0]
    worst_depth = depths[0]
    worst_idx = 0

    train_acc_lr = []
    valid_acc_lr = []
    test_acc_lr = []

    train_loss_lr = []
    valid_loss_lr = []
    test_loss_lr = []

    idx = 0
    for depth, width, epoch in zip(depths, widths, epochs):
        print(idx)
        output_dim = len(train_data['country'].unique())
        layers = list()
        layers.append(nn.Linear(2, width))
        layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())
        for _ in range(1, depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))
        model = nn.Sequential(*layers)
        batch_model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, grad_magn = \
            train_model(train_data, val_data, test_data, model, lr=0.001, epochs=epoch, batch_size=128)

        models.append(model)

        train_acc_lr.append(train_accs)
        valid_acc_lr.append(val_accs)
        test_acc_lr.append(test_accs)

        train_loss_lr.append(train_losses)
        valid_loss_lr.append(val_losses)
        test_loss_lr.append(test_losses)

        # Check if current model has better validation accuracy
        if val_accs[-1] >= best_val_acc:
            best_val_acc = val_accs[-1]
            best_width = width
            best_depth = depth
            best_model = model
            best_idx = idx

        if val_accs[-1] < worst_val_acc:
            worst_val_acc = val_accs[-1]
            worst_width = width
            worst_depth = depth
            worst_model = model
            worst_idx = idx
        idx += 1
    # best model
    print("best model is with width of ", best_width, "and depth of ", best_depth)
    print("and the validation accuracy is ", best_val_acc)

    best_train_loss = train_loss_lr[best_idx]
    best_valid_loss = valid_loss_lr[best_idx]
    best_test_loss = test_loss_lr[best_idx]

    plot_losses(best_train_loss, best_valid_loss, best_test_loss)

    helpers.plot_decision_boundaries(best_model, test_data[['long', 'lat']].values, test_data['country'].values)

    # worst model
    print("worst model is with width of ", worst_width, "and depth of ", worst_depth)
    print("and the validation accuracy is ", worst_val_acc)


    worst_train_loss = train_loss_lr[worst_idx]
    worst_valid_loss = valid_loss_lr[worst_idx]
    worst_test_loss = test_loss_lr[worst_idx]

    plot_losses(worst_train_loss, worst_valid_loss, worst_test_loss)

    helpers.plot_decision_boundaries(worst_model, test_data[['long', 'lat']].values, test_data['country'].values)

    # where width is 16
    hidden_layers_16 = depths[:4]
    print(hidden_layers_16)
    models_width_16 = models[:4]
    train_loss_16 = [row[-1] for row in train_loss_lr[:4]]
    valid_loss_16 = [row[-1] for row in valid_loss_lr[:4]]
    test_loss_16 = [row[-1] for row in test_loss_lr[:4]]
    train_acc_16 = [row[-1] for row in train_acc_lr[:4]]
    val_acc_16 = [row[-1] for row in valid_acc_lr[:4]]
    test_acc_16 = [row[-1] for row in test_acc_lr[:4]]
    plot_6_2_1_3_4(hidden_layers_16, train_acc_16, val_acc_16, test_acc_16, 'number of hidden layers')

    # where depth is 6
    num_of_neurons = widths[2:3] + widths[-3:]

    models_width_6 = models[2:3] + models[-3:]
    train_loss_6 = train_loss_lr[2:3] + train_loss_lr[-3:]
    valid_loss_6 = valid_loss_lr[2:3] + valid_loss_lr[-3:]
    test_loss_6 = test_loss_lr[2:3] + test_loss_lr[-3:]
    train_acc_6 = [row[-1] for row in train_acc_lr[2:3]] + [row[-1] for row in train_acc_lr[-3:]]
    val_acc_6 = [row[-1] for row in valid_acc_lr[2:3]] + [row[-1] for row in valid_acc_lr[-3:]]
    test_acc_6 = [row[-1] for row in test_acc_lr[2:3]] + [row[-1] for row in test_acc_lr[-3:]]
    print(train_acc_6)
    print(val_acc_6)
    print(test_acc_6)
    plot_6_2_1_3_4(num_of_neurons, train_acc_6, val_acc_6, test_acc_6, 'number of nuerons')

    return best_model, best_val_acc, train_acc_lr, worst_model, worst_val_acc, \
           valid_acc_lr, test_acc_lr, train_loss_lr,valid_loss_lr, test_loss_lr

def q_6_2_1_5(train_data, val_data, test_data):
    width = 100
    depth = 4
    epochs = 10
    grad_magnitudes = [0, 30, 60, 90, 95, 99]

    output_dim = len(train_data['country'].unique())
    layers = list()
    layers.append(nn.Linear(2, width))
    layers.append(nn.BatchNorm1d(width))
    layers.append(nn.ReLU())
    for _ in range(1, depth):
        layers.append(nn.Linear(width, width))
        layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width, output_dim))
    model = nn.Sequential(*layers)
    batch_model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, grad_magn = \
        train_model(train_data, val_data, test_data, model, lr=0.001, epochs=epochs, batch_size=256, grad_layers=grad_magnitudes)
    print(grad_magn)

    for layer in grad_magnitudes[1:]:
        if not grad_magn[layer]:  # Check if gradient data is missing for the layer
            grad_magn[layer] = [0] * len(grad_magn[0])
    epochs = range(1, len(grad_magn[0]) + 1)  # Assuming all layers have the same number of epochs

    for layer in grad_magnitudes:
          # Check if gradient data exists for the layer
        plt.plot(epochs, grad_magn[layer], label=f'Layer {layer}')

    plt.xlabel('Epochs')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Magnitudes for Specified Layers')
    plt.legend()
    plt.grid(True)
    plt.show()


def q_6_2_1_7(train_data, val_data, test_data):
    output_dim = len(train_data['country'].unique())
    # sin the train data
    sin_train_data = train_data.copy()

    for idx, alpha in enumerate(np.arange(0.1, 1.1, 0.1), start=1):
        sin_train_data[f'long_sin{idx}'] = np.sin(alpha * train_data['long'].values)
        sin_train_data[f'lat_sin{idx}'] = np.sin(alpha * train_data['lat'].values)

    sin_train_data.drop(columns=['Unnamed: 0', 'long', 'lat', 'country'], inplace=True)

    sin_train_data['country'] = train_data['country']
    # sin the val data
    sin_val_data = val_data.copy()
    for idx, alpha in enumerate(np.arange(0.1, 1.1, 0.1), start=1):
        sin_val_data[f'long_sin{idx}'] = np.sin(alpha * val_data['long'].values)
        sin_val_data[f'lat_sin{idx}'] = np.sin(alpha * val_data['lat'].values)


    sin_val_data.drop(columns=['Unnamed: 0', 'long', 'lat', 'country'], inplace=True)

    sin_val_data['country'] = val_data['country']
    # sin the test data
    sin_test_data = test_data.copy()
    for idx, alpha in enumerate(np.arange(0.1, 1.1, 0.1), start=1):
        sin_test_data[f'long_sin{idx}'] = np.sin(alpha * test_data['long'].values)
        sin_test_data[f'lat_sin{idx}'] = np.sin(alpha * test_data['lat'].values)

    sin_test_data.drop(columns=['Unnamed: 0', 'long', 'lat', 'country'], inplace=True)
    # sin_test_data.drop(columns=['country'], inplace=True)

    sin_test_data['country'] = test_data['country']

    print(sin_train_data)
    print(sin_val_data)
    print(sin_test_data)

    model = [nn.Linear(20, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    model = nn.Sequential(*model)

    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = \
        train_sin_data(sin_train_data, sin_val_data, sin_test_data, model, lr=0.001, epochs=50, batch_size=256)

    helpers.plot_decision_boundaries(model,  test_data[['long', 'lat']].values, test_data['country'].values,
                             'sin', implicit_repr=True)




if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    lrs = [1., 0.01, 0.001, 0.00001]

    valid_loss = MLP_task_lr(train_data, val_data, test_data)
    x_ax_first = [i for i in range(1, 51)]
    plot_lr(valid_loss, x_ax_first)


    x_ax_sec = [1, 5, 10, 20, 50, 100]
    num_of_epochs = 100
    valid_loss = MLP_task_ep(train_data, val_data, test_data, num_of_epochs)
    plot_ep(valid_loss, x_ax_sec)
    print(valid_loss)

    # check what happens after too much epochs

    x_ax_200 = [1, 5, 10, 20, 50, 100, 200]
    num_of_epochs = 200
    valid_loss = MLP_task_ep(train_data, val_data, test_data, num_of_epochs)
    plot_ep(valid_loss, x_ax_200)
    print(valid_loss)


    MLP_batch_norm(train_data, val_data, test_data)

    MLP_batch_size(train_data, val_data, test_data)

    # 6.2.1.1
    best_model, best_val_acc, train_acc_lr, worst_model, worst_val_acc, \
    valid_acc_lr, test_acc_lr, train_loss_lr, valid_loss_lr, test_loss_lr = eval_MLP(train_data, val_data, test_data)

    q_6_2_1_5(train_data, val_data, test_data)

    q_6_2_1_7(train_data, val_data, test_data)
