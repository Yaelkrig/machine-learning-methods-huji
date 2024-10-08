from utils import *
from prophets import *


def case_two_prophets(num_of_games):

    p1_wins = 0

    test_err = 0.

    est_err = 0.

    p1 = Prophet(0.2)

    p2 = Prophet(0.4)

    for exp in range(100):

        trainset_reduced = np.random.choice(train_set[exp, :], size=num_of_games)

        p1_err = compute_error(p1.predict(trainset_reduced), trainset_reduced)

        p2_err = compute_error(p2.predict(trainset_reduced), trainset_reduced)

        if p1_err <= p2_err:

            p1_wins += 1

            test_err += compute_error(p1.predict(test_set), test_set)

        else:

            test_err += compute_error(p2.predict(test_set), test_set)

            est_err += 0.2

    print("Number of times best prophet selected: ", p1_wins)

    print("Average test error of selected prophet: ", test_err / 100.)

    print("Average approximation error: 0.2")

    print("Average estimation error: ", est_err / 100.)


def find_best_emp(prophets_arr, num_of_prophets, trainset_reduced):
    min_empirical_risk = compute_error(prophets_arr[0].predict(trainset_reduced), trainset_reduced)

    min_err_idx = 0

    for prophet_idx in range(1, num_of_prophets):

        cur_p_err = compute_error(prophets_arr[prophet_idx].predict(trainset_reduced), trainset_reduced)

        if cur_p_err < min_empirical_risk:

            min_empirical_risk = cur_p_err

            min_err_idx = prophet_idx

    return min_empirical_risk, min_err_idx


def find_best_prophet(prophets_arr, num_of_prophets):
    best_p = prophets_arr[0]

    best_p_err = best_p.err_prob

    best_p_idx = 0

    for i in range(num_of_prophets):

        if prophets_arr[i].err_prob < best_p_err:

            best_p = prophets_arr[i]

            best_p_err = best_p.err_prob

            best_p_idx = i

    return best_p, best_p_err, best_p_idx


def case_many_prophets(num_of_prophets, num_of_games, min_error_rate, max_error_rate):

    best_p_wins = 0

    test_err = 0.

    est_err = 0.

    general_diff = 0.

    prophets_arr = sample_prophets(num_of_prophets, min_error_rate, max_error_rate)

    best_p, best_p_err, best_p_idx = find_best_prophet(prophets_arr, num_of_prophets)

    worth_less_than_precent = 0

    for exp in range(100):

        trainset_reduced = np.random.choice(train_set[exp, :], size=num_of_games)

        min_empirical_risk, min_err_idx = find_best_emp(prophets_arr, num_of_prophets, trainset_reduced)

        train_general_err = np.absolute(prophets_arr[min_err_idx].err_prob-min_empirical_risk)

        cur_test_err = compute_error(prophets_arr[min_err_idx].predict(test_set), test_set)

        test_general_err = np.absolute(prophets_arr[min_err_idx].err_prob-cur_test_err)

        general_diff += train_general_err-test_general_err

        test_err += cur_test_err

        if min_err_idx == best_p_idx:

            best_p_wins += 1

        else:

            est_err += prophets_arr[min_err_idx].err_prob - best_p_err

            if (prophets_arr[min_err_idx].err_prob - 0.01) <= best_p_err:

                worth_less_than_precent += 1

    print("Number of times best prophet selected: ", best_p_wins)

    print("Number of times prophets who not worth 1% than the best prophet selected: ", worth_less_than_precent)

    print("Average test error of selected prophet: ", test_err / 100.)

    print("Average diff of generalization error on training and on test: ", general_diff/100.)

    print("Average approximation error: ", best_p_err)

    print("Average estimation error: ", est_err / 100.)


def Scenario_1():
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############

    case_two_prophets(1)


def Scenario_2():
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    case_two_prophets(10)


def Scenario_3():
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    case_many_prophets(500, 10, 0, 1)
    print(f"in case the error rates uniformly distribute between [0, 0.5]")
    case_many_prophets(500, 10, 0, 0.5)


def Scenario_4():
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    case_many_prophets(500, 1000, 0, 1)


def Scenario_5():
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############

    output_table_est = np.ndarray((4, 4))
    output_table_app = np.ndarray((4, 4))

    for k_idx, k in enumerate([2, 5, 10, 50]):

        prophets_arr = sample_prophets(k, 0, 0.2)

        best_p, best_p_err, best_p_idx = find_best_prophet(prophets_arr, k)

        for m_idx, m in enumerate([1, 10, 50, 1000]):

            best_p_wins = 0

            test_err = 0.

            est_err = 0.

            for exp in range(100):
                trainset_reduced = np.random.choice(train_set[exp, :], size=m)

                min_empirical_risk, min_err_idx = find_best_emp(prophets_arr, k, trainset_reduced)

                if min_err_idx == best_p_idx:
                    best_p_wins += 1
                    test_err += compute_error(best_p.predict(test_set), test_set)
                else:
                    test_err += compute_error(prophets_arr[min_err_idx].predict(test_set), test_set)
                    est_err += prophets_arr[min_err_idx].err_prob - best_p_err

            output_table_est[k_idx][m_idx] = est_err / 100.

            output_table_app[k_idx][m_idx] = best_p_err

            # print("k is", k, ", and m is", m)
            #
            # print("Number of times best prophet selected: ", best_p_wins)
            #
            # print("Average test error of selected prophet: ", test_err / 100.)
            #
            # print("Average approximation error: ", best_p_err)
            #
            # print("Average estimation error: ", est_err / 100.)
    print(output_table_est)
    print(output_table_app)


def Scenario_6():
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############

    first_hypothesis_class = sample_prophets(5, 0.3, 0.6)
    sec_hypothesis_class = sample_prophets(500, 0.25, 0.6)

    first_best_p, first_best_p_err, first_best_p_idx = find_best_prophet(first_hypothesis_class, 5)
    first_best_p_wins = 0
    first_test_err = 0.
    first_est_err = 0.

    sec_best_p, sec_best_p_err, sec_best_p_idx = find_best_prophet(sec_hypothesis_class, 500)
    sec_best_p_wins = 0
    sec_test_err = 0.
    sec_est_err = 0.

    for exp in range(100):
        trainset_reduced = np.random.choice(train_set[exp, :], size=10)

        fmin_empirical_risk, fmin_err_idx = find_best_emp(first_hypothesis_class, 5, trainset_reduced)
        smin_empirical_risk, smin_err_idx = find_best_emp(sec_hypothesis_class, 500, trainset_reduced)

        if fmin_err_idx == first_best_p_idx:
            first_best_p_wins += 1
            first_test_err += compute_error(first_best_p.predict(test_set), test_set)
        else:
            first_test_err += compute_error(first_hypothesis_class[fmin_err_idx].predict(test_set), test_set)
            # print("first est error", first_est_err)
            first_est_err += (first_hypothesis_class[fmin_err_idx].err_prob - first_best_p_err)

        if smin_err_idx == sec_best_p_idx:
            sec_best_p_wins += 1
            sec_test_err += compute_error(sec_best_p.predict(test_set), test_set)
        else:
            sec_test_err += compute_error(sec_hypothesis_class[smin_err_idx].predict(test_set), test_set)
            # print("sec est error", sec_est_err)
            sec_est_err += sec_hypothesis_class[smin_err_idx].err_prob - sec_best_p_err
    print("first class estimation error is", first_est_err/100)
    print("second class estimation error is", sec_est_err / 100)

    print("first class approximation error is", first_best_p_err)
    print("second class approximation error is", sec_best_p_err)

    # print(first_test_err/100)
    # print(sec_test_err / 100)


if __name__ == '__main__':
    np.random.seed(0)  # DO NOT MOVE / REMOVE THIS CODE LINE!

    # train, validation and test splits for Scenario 1-3, 5
    train_set = create_data(100, 1000)
    test_set = create_data(1, 1000)[0]

    print(f'Scenario 1 Results:')
    Scenario_1()

    print(f'Scenario 2 Results:')
    Scenario_2()

    print(f'Scenario 3 Results:')
    Scenario_3()

    print(f'Scenario 4 Results:')
    Scenario_4()

    print(f'Scenario 5 Results:')
    Scenario_5()

    print(f'Scenario 6 Results:')
    Scenario_6()

