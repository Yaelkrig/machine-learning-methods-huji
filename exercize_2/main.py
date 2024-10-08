import numpy as np
from helpers import *
import matplotlib.pyplot as plt

import matplotlib as mpl

def create_data():
    train_set, col_names_train = read_data_demo('train.csv')
    X_train = train_set[:, :2].astype(np.float32)
    Y_train = train_set[:, 2:].flatten()

    test_set, col_names_test = read_data_demo('validation.csv')
    X_valid = test_set[:, :2].astype(np.float32)
    Y_valid = test_set[:, 2:].flatten()

    test_set, col_names_test = read_data_demo('test.csv')
    X_test = test_set[:, :2].astype(np.float32)
    Y_test = test_set[:, 2:].flatten()

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def create_classifier(k, distance_metric, X_train, Y_train, X_test, Y_test):
    knn_classifier = KNNClassifier(k=k, distance_metric=distance_metric)
    knn_classifier.fit(X_train, Y_train)
    y_pred = knn_classifier.predict(X_test)
    knn_classifier.test_accuracy = np.mean(y_pred == Y_test)
    return knn_classifier

def task_1(X_train, Y_train, X_test, Y_test):
    ks = [1, 10, 100, 1000, 3000]
    distance_metrics = ['l1', 'l2']
    classifiers = []
    results = np.empty((len(ks), len(distance_metrics)), dtype=np.float32)
    mean_diff = 0
    for k_idx in range(len(ks)):
        for mat_idx in range(len(distance_metrics)):
            classifier = create_classifier(ks[k_idx], distance_metrics[mat_idx], X_train, Y_train, X_test, Y_test)
            classifiers.append(classifier)
            results[k_idx][mat_idx] = classifier.test_accuracy
        mean_diff += (classifiers[-2].test_accuracy - classifiers[-1].test_accuracy)
    mean_diff /= 5
    print(results)
    print("the mean diff of l1 and l2:", mean_diff)
    return classifiers

def plot_annomal(X_train, X_anomal_test, indices_of_annomal):
    indices_of_normal = np.setdiff1d(np.arange(len(X_anomal_test)), indices_of_annomal)

    plt.scatter(X_anomal_test[indices_of_normal, 0], X_anomal_test[indices_of_normal, 1],
                c='blue', label='Normal', edgecolor="black")
    plt.scatter(X_anomal_test[indices_of_annomal, 0], X_anomal_test[indices_of_annomal, 1],
                c='red', label='Annomal', edgecolor="black")
    plt.scatter(X_train[:, 0], X_train[:, 1],
                c='black', label='Train', alpha=0.01, edgecolor="black")

    plt.xlabel('Longtitude')
    plt.ylabel('Latitude')
    plt.title('Plot Annomal Detection')
    plt.legend()
    plt.show()

def annomal_detection():
    train_set = read_data_demo()[0]
    X_train = train_set[:, :2].astype(np.float32)
    Y_train = train_set[:, 2:].flatten()
    classifier = KNNClassifier(5, 'l2')
    classifier.fit(X_train, Y_train)
    X_anomal_test = read_data_demo('AD_test.csv')[0]

    distances, idx = classifier.knn_distance(X_anomal_test)

    sum_dis = distances.sum(axis=1)
    idx_of_annomal = np.argpartition(sum_dis, -50)[-50:]
    plot_annomal(X_train, X_anomal_test, idx_of_annomal)

def trees_task(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    maximal_depth = [1, 2, 4, 6, 10, 20, 50, 100]
    maximal_leaf_nodes = [50, 100, 1000]
    tree_models = []
    training_accuracies = np.empty((len(maximal_depth)*len(maximal_leaf_nodes)))
    validation_accuracies = np.empty((len(maximal_depth)*len(maximal_leaf_nodes)))
    test_accuracies = np.empty((len(maximal_depth)*len(maximal_leaf_nodes)))
    fifty_leaf = []
    six_depth = []

    i = 0
    for depth in maximal_depth:
        for leaf in maximal_leaf_nodes:
            if leaf == 50:
                fifty_leaf.append(i)
            if depth == 6:
                six_depth.append(i)
            dec_tree = DecisionTreeClassifier(max_depth=depth, max_leaf_nodes=leaf)
            dec_tree.fit(X_train, Y_train)

            y_train_pred = dec_tree.predict(X_train)
            training_accuracies[i] = np.mean(y_train_pred == Y_train)

            y_valid_pred = dec_tree.predict(X_valid)
            validation_accuracies[i] = np.mean(y_valid_pred == Y_valid)

            y_test_pred = dec_tree.predict(X_test)
            test_accuracies[i] = np.mean(y_test_pred == Y_test)

            tree_models.append(dec_tree)

            i += 1

    arg_of_best_validation = np.argmax(validation_accuracies)
    print("best validation accuracy: ", validation_accuracies[arg_of_best_validation])
    print("the tree of best validation accuracy is:")
    print("with maximal depth of", tree_models[arg_of_best_validation].max_depth)
    print("and maximal leaf", tree_models[arg_of_best_validation].max_leaf_nodes)

    print("training accuracy of this tree is:", training_accuracies[arg_of_best_validation])
    print("testing accuracy of this tree is:", test_accuracies[arg_of_best_validation])

    plot_decision_boundaries(tree_models[arg_of_best_validation], X_test, Y_test, 'map of decision tree')

    # 6.5

    best_valid_fifty = np.argmax(validation_accuracies[np.array(fifty_leaf)])
    best_valid_six = np.argmax(validation_accuracies[np.array(six_depth)])

    idx_of_best_fifty = fifty_leaf[best_valid_fifty]
    idx_of_best_six = six_depth[best_valid_six]

    plot_decision_boundaries(tree_models[idx_of_best_fifty], X_test, Y_test, 'best tree of 50 leaf nodes')
    plot_decision_boundaries(tree_models[idx_of_best_six], X_test, Y_test, 'best tree of depth 6')
    print("accuracies of tree", training_accuracies[idx_of_best_six], validation_accuracies[idx_of_best_six]
          , test_accuracies[idx_of_best_six])

    # plot_tree(tree_models[idx_of_best_six], filled=True)
    # plt.show()

def random_forest(X_train, Y_train, X, y):
    model = RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=4)
    model.fit(X_train, Y_train)
    print("accuracies of forest", np.mean(model.predict(X_train) == Y_train),
          np.mean(model.predict(X_valid) == Y_valid),
          np.mean(model.predict(X) == y))

    plot_decision_boundaries(model, X, y, 'random forest')


if __name__ == "__main__":
    np.random.seed(0)
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = create_data()
    print('5.1')
    classifiers = task_1(X_train, Y_train, X_test, Y_test)

    # 5.2 # plot_tree(dec_tree, filled=True)
    #             # plt.show()
    print('\n5.2')
    model_k_max_l2 = classifiers[1]
    model_k_min_l2 = classifiers[9]
    model_k_max_l1 = classifiers[0]
    plot_decision_boundaries(classifiers[1], X_test, Y_test, 'for k_max and l2:')
    plot_decision_boundaries(classifiers[9], X_test, Y_test, 'for k_min and l2:')
    plot_decision_boundaries(classifiers[0], X_test, Y_test, 'for k_max and l1:')

    # 5.3
    print('\n5.3')
    annomal_detection()

#     6

    trees_task(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

#      6.7
    random_forest(X_train, Y_train, X_test, Y_test)


