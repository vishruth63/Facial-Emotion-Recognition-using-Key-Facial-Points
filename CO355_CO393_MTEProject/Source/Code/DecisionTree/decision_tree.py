import pandas as pd
import numpy as np
import random as rand
import scipy.stats as stats
import sys
from multiprocessing import Process
from multiprocessing import Queue

import utilities as util
import constants as cnst
import measures

from node import TreeNode

def decision_tree_parallel(examples, attributes, bin_targets, queue):
    root = decision_tree(examples, attributes, bin_targets)
    queue.put(root)

def decision_tree(examples, attributes, bin_targets):
    all_same = check_all_same(bin_targets)

    if all_same:
        return TreeNode(None, True, bin_targets.iloc[0].iloc[0])
    elif not attributes:
        # Majority Value
        return TreeNode(None, True, majority_value(bin_targets))
    else:
        best_attribute = choose_best_decision_attr(examples, attributes, bin_targets)
        tree = TreeNode(best_attribute)
        for vi in range(0, 2):
            examples_i = examples.loc[examples[best_attribute] == vi]
            indices = examples_i.index.values
            bin_targets_i = bin_targets.ix[indices]

            if examples_i.empty:
                # Majority Value
                return TreeNode(None, True, majority_value(bin_targets))
            else:
                attr = set(attributes)
                attr.remove(best_attribute)
                tree.set_child(vi, decision_tree(examples_i, attr, bin_targets_i))

        return tree

def check_all_same(df):
    return df.apply(lambda x: len(x[-x.isnull()].unique()) == 1 , axis = 0).all()

def majority_value(bin_targets):
    res = stats.mode(bin_targets[0].values)[0][0]
    return res

def choose_best_decision_attr(examples, attributes, bin_targets):
    max_gain = -sys.maxsize - 1
    index_gain = -1

    # p and n: training data has p positive and n negative examples
    p = len(bin_targets.loc[bin_targets[0] == 1].index)
    n = len(bin_targets.loc[bin_targets[0] == 0].index)

    for attribute in attributes:
        examples_pos = examples.loc[examples[attribute] == 1]
        examples_neg = examples.loc[examples[attribute] == 0]
        index_pos = examples_pos.index.values
        index_neg = examples_neg.index.values

        p0 = n0 = p1 = n1 = 0

        for index in index_pos:
            if bin_targets[0][index] == 1:
                p1 = p1 + 1
            else:
                n1 = n1 + 1

        for index in index_neg:
            if bin_targets[0][index] == 1:
                p0 = p0 + 1
            else:
                n0 = n0 + 1

        curr_gain = gain(p, n, p0, n0, p1, n1)
        if curr_gain > max_gain:
            index_gain = attribute
            max_gain = curr_gain

    if max_gain == -sys.maxsize - 1:
        raise ValueError('Index gain is original value...')

    return index_gain

def gain(p, n, p0, n0, p1, n1):
    return get_info_gain(p, n) - get_remainder(p, n, p0, n0, p1, n1)

def get_info_gain(p, n):
    if p + n == 0:
        return 0
    term_1 = float(p / (p + n))
    term_2 = float(n / (p + n))
    return stats.entropy([term_1, term_2], base=2)

# Remainder(attribute) = (p0 + n0)/(p + n) * I(p0, n0) + (p1 + n1)/(p + n) * I(p1, n1)
def get_remainder(p, n, p0, n0, p1, n1):
    return ((p0 + n0)/(p + n)) * get_info_gain(p0, n0) + ((p1 + n1)/(p + n)) * get_info_gain(p1, n1) if p+n != 0 else 0


def compare_pred_expect(predictions, expectations):
    confusion_matrix = pd.DataFrame(0, index=cnst.EMOTIONS_INDICES, columns=cnst.EMOTIONS_INDICES)
    predictions, expectations = predictions.reset_index(drop=True), expectations.reset_index(drop=True)

    for index in predictions.index.values:
        e = expectations.iloc[index] - 1
        p = predictions.iloc[index] - 1
        confusion_matrix.loc[p, e] += 1

    return confusion_matrix

def choose_prediction_random(pred_proc_depth):
    predictions, proc, depths = zip(*pred_proc_depth)
    occurrences = [index for index, value in enumerate(predictions) if value == 1]
    if len(occurrences) == 1:
        return occurrences[0]
    elif len(occurrences) == 0:
        return rand.randint(0, 5)
    else:
        return rand.choice(occurrences)


def choose_prediction_optimised(pred_proc_depth):
    predictions, proc, depths  = zip(*pred_proc_depth)
    indexes = [index for index, value in enumerate(predictions) if value == 1]

    if len(indexes) == 1:
        return indexes[0]
    elif len(indexes) == 0:
        res = 0
        MAX = 0
        max_depth_indexes = []
        for i in range(0, len(depths)):
            if depths[i] > MAX:
                MAX = depths[i]
                del max_depth_indexes[:]
                max_depth_indexes.append(i)
            elif depths[i] == MAX:
                max_depth_indexes.append(i)
        if len(max_depth_indexes) == 1:
            res = max_depth_indexes[0]
        else:
            min_proc = 100
            min_proc_index = 0
            for i in max_depth_indexes:
                if (proc[i] < min_proc):
                    min_proc = proc[i]
                    min_proc_index = i
            res = min_proc_index
        return res
    else:
        res = 0
        MIN = 10000
        max_depth_indexes = []
        for i in indexes:
            if depths[i] < MIN:
                MIN = depths[i]
                del max_depth_indexes[:]
                max_depth_indexes.append(i)
            elif depths[i] == MIN:
                max_depth_indexes.append(i)
        if len(max_depth_indexes) == 1:
            res = max_depth_indexes[0]
        else:
            max_proc = 0
            max_proc_index = 0
            for i in max_depth_indexes:
                if (proc[i] > max_proc):
                    max_proc = proc[i]
                    max_proc_index = i
            res = max_proc_index
        return res

def test_trees(T_P, x2):

    T, P = zip(*T_P)

    predictions = []

    for i in x2.index.values:
        example = x2.loc[i]
        T_P_D = []
        for j in range(0, len(T_P)):
            prediction, depth = TreeNode.dfs_with_depth(T[j], example)
            T_P_D.append([prediction, P[j], depth])

        prediction_choice = choose_prediction_optimised(T_P_D)
        predictions.append(prediction_choice + 1)

    return pd.DataFrame(predictions)

def visualise(df_labels, df_data, N):
    for e in cnst.EMOTIONS_LIST:
        root = decision_tree(df_data, set(cnst.AU_INDICES), util.filter_for_emotion(df_labels, cnst.EMOTIONS_DICT[e]))
        TreeNode.plot_tree(root, e)

def apply_d_tree_parallel(df_labels, df_data, N):
    print(">> Running decision tree algorithm on multiple processes.\n")

    res = pd.DataFrame(0, index=cnst.EMOTIONS_INDICES, columns=cnst.EMOTIONS_INDICES)

    segments = util.preprocess_for_cross_validation(N)

    for test_seg in segments:
        print(">> Starting fold... from:", test_seg)
        print()

        T = []
        # Split data into 90% Training and 10% Testing
        test_df_data, test_df_targets, train_df_data, train_df_targets = util.divide_data(test_seg, N, df_data, df_labels)

        # Further split trainig data into 90% Training and 10% Validation data
        K = train_df_data.shape[0]
        segs = util.preprocess_for_cross_validation(K)
        validation_data, validation_targets, train_data, train_targets = util.divide_data(segs[-1], K, train_df_data, train_df_targets)

        processes = []
        queue_list = []

        for e in cnst.EMOTIONS_LIST:
            print("Building decision tree for emotion...", e)
            train_binary_targets = util.filter_for_emotion(train_df_targets, cnst.EMOTIONS_DICT[e])

            q = Queue()
            queue_list.append(q)

            process = Process(target=decision_tree_parallel, args=(train_df_data, set(cnst.AU_INDICES), train_binary_targets, q))
            processes.append(process)
            process.start()

        for p in processes:
            p.join()

        for q in queue_list:
            T.append(q.get())

        percentage = []
        T_P = []
        for e in cnst.EMOTIONS_LIST:
            print("\nValidation phase for emotion: ", e)
            validation_binary_targets = util.filter_for_emotion(validation_targets, cnst.EMOTIONS_DICT[e])
            results = []
            # Calculate how accurate each tree is when predicting emotions
            for i in validation_data.index.values:
                results.append(TreeNode.dfs2(T[cnst.EMOTIONS_DICT[e]- 1], validation_data.loc[i], validation_binary_targets.loc[i].at[0]))
            ones = results.count(1)
            percentage.append(ones/len(results))
            print("Validation phase ended. Priority levels have been set.")

        print("All decision trees built.\n")

        T_P = list(zip(T, percentage))

        predictions = test_trees(T_P, test_df_data)
        confusion_matrix = compare_pred_expect(predictions, test_df_targets)

        print(confusion_matrix)
        # Print accuracy for each fold
        diag = sum(pd.Series(np.diag(confusion_matrix),
                            index=[confusion_matrix.index, confusion_matrix.columns]))
        sum_all = confusion_matrix.values.sum()
        accuracy = (diag/sum_all) * 100
        print("Accuracy:", accuracy)

        res = res.add(confusion_matrix)

    # res = res.div(10)
    res = res.div(res.sum(axis=1), axis=0)

    for e in cnst.EMOTIONS_LIST:
        print("----------------------------------- MEASUREMENTS -----------------------------------")
        print(measures.compute_binary_confusion_matrix(res, cnst.EMOTIONS_DICT[e]))

    return res

def apply_d_tree(df_labels, df_data, N):
    print(">> Running decision tree algorithm on a single process.\n")

    res = pd.DataFrame(0, index=cnst.EMOTIONS_INDICES, columns=cnst.EMOTIONS_INDICES)

    segments = util.preprocess_for_cross_validation(N)
    total_accuracy = 0
    for test_seg in segments:
        print(">> Starting fold... from:", test_seg)
        print()

        T = []
        # Split data into 90% Training and 10% Testing
        test_df_data, test_df_targets, train_df_data, train_df_targets = util.divide_data(test_seg, N, df_data, df_labels)

        # Further split trainig data into 90% Training and 10% Validation data
        K = train_df_data.shape[0]
        segs = util.preprocess_for_cross_validation(K)
        validation_data, validation_targets, train_data, train_targets = util.divide_data(segs[-1], K, train_df_data, train_df_targets)

        # Train Trees
        for e in cnst.EMOTIONS_LIST:
            print("Building decision tree for emotion: ", e)
            train_binary_targets = util.filter_for_emotion(train_df_targets, cnst.EMOTIONS_DICT[e])
            root = decision_tree(train_data, set(cnst.AU_INDICES), train_binary_targets)
            print("Decision tree built. Now appending...")
            T.append(root)

        percentage = []
        T_P = []
        for e in cnst.EMOTIONS_LIST:
            print("\nValidation phase for emotion: ", e)
            validation_binary_targets = util.filter_for_emotion(validation_targets, cnst.EMOTIONS_DICT[e])
            results = []
            # Calculate how accurate each tree is when predicting emotions
            for i in validation_data.index.values:
                results.append(TreeNode.dfs2(T[cnst.EMOTIONS_DICT[e]- 1], validation_data.loc[i], validation_binary_targets.loc[i].at[0]))
            ones = results.count(1)
            percentage.append(ones/len(results))
            print("Validation phase ended. Priority levels have been set.")

        print("All decision trees built.\n")

        # List containing (Tree, Percentage) tuples
        T_P = list(zip(T, percentage))

        predictions = test_trees(T_P, test_df_data)
        confusion_matrix = compare_pred_expect(predictions, test_df_targets)

        print(confusion_matrix)
        # Print accuracy for each fold
        diag = sum(pd.Series(np.diag(confusion_matrix),
                            index=[confusion_matrix.index, confusion_matrix.columns]))
        sum_all = confusion_matrix.values.sum()
        accuracy = (diag/sum_all) * 100
        total_accuracy += accuracy
        print("Accuracy:", accuracy)

        res = res.add(confusion_matrix)
        print("Folding ended.\n")
        print()
    print("Total accuracy:", accuracy)
    res = res.div(res.sum(axis=1), axis=0)
    print(res)
    return res


    res = res.div(res.sum(axis=1), axis=0)

    for e in cnst.EMOTIONS_LIST:
        print("----------------------------------- MEASUREMENTS -----------------------------------")
        print(measures.compute_binary_confusion_matrix(res, cnst.EMOTIONS_DICT[e]))

    return res
