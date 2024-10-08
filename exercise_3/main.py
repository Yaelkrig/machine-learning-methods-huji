import numpy as np
import sklearn
import torch
import pandas as pd
from torch import nn, optim

from models import Ridge_Regression, Logistic_Regression
from helpers import read_data_demo, plot_decision_boundaries
import matplotlib.pyplot as plt


def read_data_demo_torch(filename='train.csv'):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """

    # the data in pandas dataframe format
    df = pd.read_csv(filename)

    # extract the column names
    col_names = list(df.columns)

    # the data in numpy array format
    data_torch = torch.tensor(df.values)

    return data_torch, col_names


def create_datas(train_path, validation_path, test_path):
    train_set, col_names_train = read_data_demo(train_path)
    X_train = train_set[:, :2].astype(np.float32)
    Y_train = train_set[:, 2:].flatten()

    test_set, col_names_test = read_data_demo(validation_path)
    X_valid = test_set[:, :2].astype(np.float32)
    Y_valid = test_set[:, 2:].flatten()

    test_set, col_names_test = read_data_demo(test_path)
    X_test = test_set[:, :2].astype(np.float32)
    Y_test = test_set[:, 2:].flatten()

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def first_task(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):

    lambdas = [0., 2., 4., 6., 8., 10.]
    models = []
    train_accur = np.empty(len(lambdas))
    valid_accur = np.empty(len(lambdas))
    test_accur = np.empty(len(lambdas))

    for lambd_idx in range(len(lambdas)):
        ridge_reg = Ridge_Regression(lambdas[lambd_idx])
        ridge_reg.fit(X_train, Y_train)

        train_accur[lambd_idx] = np.mean(Y_train == ridge_reg.predict(X_train))

        valid_accur[lambd_idx] = np.mean(ridge_reg.predict(X_valid) == Y_valid)

        test_accur[lambd_idx] = np.mean(ridge_reg.predict(X_test) == Y_test)

        models.append(ridge_reg)

    plt.scatter(lambdas, valid_accur, label='Validation')
    plt.scatter(lambdas, test_accur, label='Test')
    plt.scatter(lambdas, train_accur, label='Train')

    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('Accuracy As Function Of Lambda')
    plt.legend()
    plt.show()

    best_model_idx = np.argmax(valid_accur)
    print("test accuracy of the best model according to the validation set is ", test_accur[best_model_idx])
    plot_decision_boundaries(models[best_model_idx], X_test, Y_test, 'Best Model With Lambda ' + str(lambdas[best_model_idx]))
    worst_model_idx = np.argmin(valid_accur)
    plot_decision_boundaries(models[worst_model_idx], X_test, Y_test, 'Worst Model With Lambda ' + str(lambdas[worst_model_idx]))
    print('train accuracy of the best model is ', train_accur[best_model_idx], ' and of the worst is ', train_accur[worst_model_idx])
    print('validation accuracy of the best model is ', valid_accur[best_model_idx], ' and of the worst is ', valid_accur[worst_model_idx])
    print('test accuracy of the best model is ', test_accur[best_model_idx], ' and of the worst is ', test_accur[worst_model_idx])


def grad_f(x, y):
    df_dx = 2 * (x - 3)
    df_dy = 2 * (y - 5)
    return np.array([df_dx, df_dy])

def gradiant_decent_in_np():
    learning_rate = 0.1
    vec = np.array([0, 0])
    num_of_iterations = 1000

    for i in range(num_of_iterations):
        plt.scatter(vec[0], vec[1], c="blue")
        vec = vec - learning_rate*grad_f(vec[0], vec[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Accuracy As Function Of Lambda')
    plt.show()
    print(vec[0], vec[1])
    return vec


def plot_log_reg(train_loss, valid_loss, test_loss, num_epochs, lr, x_label):
    plt.scatter(range(num_epochs), train_loss, label='Train')
    plt.scatter(range(num_epochs), valid_loss, label='Validation')
    plt.scatter(range(num_epochs), test_loss, label='Testing')

    plt.xlabel('Epoch')
    plt.ylabel(x_label)
    plt.title(x_label + ' As Function Of Epoch, lr is ' + str(lr))
    plt.legend()
    plt.show()


def plot_log_reg_multi(test_accur, valid_accur, lr):
    plt.scatter(lr, test_accur, label='Test')
    plt.scatter(lr, valid_accur, label='Validation')

    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Accuracy As Function Of Learning Rate')
    plt.legend()
    plt.show()


def train_log_reg(dataloader, log_reg_model, device, optimizer, criterion, lambd):
    loss_train_values = []
    ep_train_accur = 0.
    log_reg_model.train()  # set the model to training mode
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = log_reg_model(inputs)
        loss = criterion(outputs.squeeze(), labels) + lambd * torch.sum(log_reg_model.linear.weight**2)
        loss.backward()
        optimizer.step()

        loss_train_values.append(loss.item())
        ep_train_accur += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    return np.mean(loss_train_values), ep_train_accur / (len(dataloader) * dataloader.batch_size)


def evaluate(X, y, log_reg_model, criterion, lambd):
    with torch.no_grad():
        log_reg_model.eval()
        outputs = log_reg_model(X)
        loss = criterion(outputs.squeeze(), y) + lambd * torch.sum(log_reg_model.linear.weight**2)

        outputs = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, dim=1)
        correct_valid_preds = torch.sum(predicted == y).item()
    return loss.item(), correct_valid_preds / len(y)


def logistic_reg(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, epochs,
                 learning_rates, step_size=0, gamma=0., lambd=0.):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_valid_tor = torch.tensor(X_valid, dtype=torch.float32).to(device)
    X_test_tor = torch.tensor(X_test, dtype=torch.float32).to(device)

    Y_train_tor = torch.tensor(Y_train, dtype=torch.long).to(device)
    Y_valid_tor = torch.tensor(Y_valid, dtype=torch.long).to(device)
    Y_test_tor = torch.tensor(Y_test, dtype=torch.long).to(device)

    lr_log_regs = []

    training_accuracies = np.empty((len(learning_rates), epochs))
    training_losses = np.empty((len(learning_rates), epochs))
    validation_accuracies = np.empty((len(learning_rates), epochs))
    validation_losses = np.empty((len(learning_rates), epochs))
    testing_accuracies = np.empty((len(learning_rates), epochs))
    testing_losses = np.empty((len(learning_rates), epochs))

    for lr_idx in range(len(learning_rates)):
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tor, Y_train_tor),
                                                 batch_size=32, shuffle=True)

        n_classes = len(torch.unique(Y_train_tor))
        log_reg = Logistic_Regression(2, n_classes)
        log_reg.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(log_reg.parameters(), lr=learning_rates[lr_idx])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(epochs):
            # training logistic regression, returning loss and accuracy
            loss_train, train_accur = train_log_reg(dataloader, log_reg, device, optimizer, criterion, lambd)

            training_losses[lr_idx][epoch] = loss_train
            training_accuracies[lr_idx][epoch] = train_accur

            # evaluating on validation set
            valid_loss, valid_accur = evaluate(X_valid_tor, Y_valid_tor, log_reg, criterion, lambd)

            validation_losses[lr_idx][epoch] = valid_loss
            validation_accuracies[lr_idx][epoch] = valid_accur

            # evalutaing on test set
            test_loss, test_accur = evaluate(X_test_tor, Y_test_tor, log_reg, criterion, lambd)

            testing_losses[lr_idx][epoch] = test_loss
            testing_accuracies[lr_idx][epoch] = test_accur
            # if we want to do step, gamma will be higher than 0
            if gamma != 0:
                lr_scheduler.step()

        lr_log_regs.append(log_reg)

    best_idx = np.argmax(validation_accuracies[:, -1])

    return lr_log_regs, best_idx, training_losses, validation_losses, testing_losses, \
           training_accuracies, validation_accuracies, testing_accuracies


def log_reg_binary_case(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    num_of_epochs = 10
    learning_rates = [0.1, 0.01, 0.001]
    lr_log_regs, best_idx, training_losses, validation_losses, testing_losses, \
    training_accuracies, validation_accuracies, testing_accuracies =\
        logistic_reg(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, num_of_epochs, learning_rates)
    # 9.3.1
    plot_decision_boundaries(lr_log_regs[best_idx], X_test, Y_test,
                             "best validation accuracy, with learning rate " + str(learning_rates[best_idx]))
#     9.3.2
    plot_log_reg(training_losses[best_idx], validation_losses[best_idx], testing_losses[best_idx], num_of_epochs,
                 learning_rates[best_idx], 'Loss')
#     9.3.3 for the comparison
    print("Best validation accuracy is ", validation_accuracies[best_idx, -1], " where learning rate is ",
          learning_rates[best_idx])
    print("testing accuracy of this model is ", testing_accuracies[best_idx, -1])


def log_reg_multi_case(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    num_of_epochs = 30
    learning_rates = [0.01, 0.001, 0.0003]
    lr_log_regs, best_idx, training_losses, validation_losses, testing_losses, \
    training_accuracies, validation_accuracies, testing_accuracies = \
        logistic_reg(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, num_of_epochs, learning_rates, step_size=5, gamma=0.3)
    # 9.4.1
    plot_log_reg_multi(testing_accuracies[:, -1], validation_accuracies[:, -1], learning_rates)
    # 9.4.2
    print("validation accuracy of best model is ", validation_accuracies[best_idx, -1])
    plot_log_reg(training_losses[best_idx], validation_losses[best_idx], testing_losses[best_idx], num_of_epochs,
                 learning_rates[best_idx], 'Loss')
    plot_log_reg(training_accuracies[best_idx], validation_accuracies[best_idx], testing_accuracies[best_idx], num_of_epochs,
                 learning_rates[best_idx], 'Accuracy')


def trees(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, max_depth):
    tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth)
    tree_classifier.fit(X_train, Y_train)

    train_preds = tree_classifier.predict(X_train)
    train_accur = np.mean(train_preds == Y_train)

    valid_preds = tree_classifier.predict(X_valid)
    validation_accur = np.mean(valid_preds == Y_valid)

    test_preds = tree_classifier.predict(X_test)
    test_accur = np.mean(test_preds == Y_test)

    plot_decision_boundaries(tree_classifier, X_test, Y_test, 'Tree Classifier With Max Depth ' + str(max_depth))
    print("Tree Train Accuracy Is ", train_accur)
    print("Tree Validation Accuracy Is ", validation_accur)
    print("Tree Test Accuracy Is ", test_accur)

def bonus(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    lambdas = [0., 2., 4., 6., 8., 10.]
    lr = [0.01]
    models = []
    trainig_acc = np.empty(len(lambdas))
    validation_acc = np.empty(len(lambdas))
    testing_acc = np.empty(len(lambdas))
    max_idx = 0
    max_accur = 0.
    for lambd_idx in range(len(lambdas)):
        lr_log_regs, best_idx, training_losses, validation_losses, testing_losses, \
        training_accuracies, validation_accuracies, testing_accuracies = \
            logistic_reg(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, 30, lr, step_size=5,
                         gamma=0.3, lambd=lambdas[lambd_idx])
        if validation_accuracies[0][-1] > max_accur:
            max_idx = lambd_idx
            max_accur = validation_accuracies[0, -1]
        trainig_acc[lambd_idx] = training_accuracies[0][-1]
        validation_acc[lambd_idx] = validation_accuracies[0][-1]
        testing_acc[lambd_idx] = testing_accuracies[0][-1]
        models.append(lr_log_regs[0])

    best_model = models[max_idx]
    print("The best model, which has the highest validation accuracy is with lambda ", lambdas[max_idx])
    print("the training accuracy is", trainig_acc[max_idx])
    print("the validation accuracy is ", validation_acc[max_idx])
    print("the testing accuracy is ", testing_acc[max_idx])
    plot_decision_boundaries(best_model, X_test, Y_test, 'bonus, lambda is ' + str(lambdas[max_idx]))


if __name__ == "__main__":

    np.random.seed(42)
    torch.manual_seed(42)
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = create_datas("train.csv", "validation.csv", "test.csv")
    # 6
    print("#6\n")
    first_task(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

    #7
    print("#7\n")
    gradiant_decent_in_np()

    #8

    #9
    # 9.3
    print("#9.3\n")
    log_reg_binary_case(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

    # 9.4
    print("#9.4.2\n")
    X_train_multi, Y_train_multi, X_valid_multi, Y_valid_multi, X_test_multi, Y_test_multi = \
        create_datas("train_multiclass.csv", "validation_multiclass.csv", "test_multiclass.csv")

    log_reg_multi_case(X_train_multi, Y_train_multi, X_valid_multi, Y_valid_multi, X_test_multi, Y_test_multi)

    # 9.4.3
    print("#9.4.3\n")

    trees(X_train_multi, Y_train_multi, X_valid_multi, Y_valid_multi, X_test_multi, Y_test_multi, max_depth=2)

    # 9.4.4
    print("#9.4.4\n")

    trees(X_train_multi, Y_train_multi, X_valid_multi, Y_valid_multi, X_test_multi, Y_test_multi, max_depth=10)

    # bonus 9.4.5
    print("#bonus 9.4.5\n")

    bonus(X_train_multi, Y_train_multi, X_valid_multi, Y_valid_multi, X_test_multi, Y_test_multi)


