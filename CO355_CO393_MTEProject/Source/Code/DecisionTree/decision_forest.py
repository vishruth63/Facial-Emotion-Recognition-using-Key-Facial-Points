import pandas as pd
import numpy as np
import random as rand
import sys
from multiprocessing import Process
from multiprocessing import Queue

import decision_tree as dtree
import utilities as util
import constants as cnst
import measures

from node import TreeNode

'''
    N - number of trees in the forest
    K - number of examples (df_data) used to train each tree
'''
def split_in_random(train_df_data, train_df_targets, N = 10, K=500):
    df = pd.concat([train_df_targets, train_df_data], axis=1)
    samples = []
    for i in range(N):
        sample = df.sample(K, replace=True)
        sample_target = sample.iloc[:, :1]
        sample_data = sample.iloc[:, 1:]
        samples.append((sample_target.reset_index(drop=True), sample_data.reset_index(drop=True)))
    return samples

def choose_majority_vote_random(predictions_depths):
    all_emotion_prediction, depths = zip(*predictions_depths)
    M = max(all_emotion_prediction)
    occurrences = [index for index, value in enumerate(all_emotion_prediction) if value == M]

    if len(occurrences) == 1:
        return occurrences[0]
    elif len(occurrences) == 0:
        return rand.randint(0, 5)
    else:
        return rand.choice(occurrences)

def choose_majority_vote_optimised(predictions_depths):
    all_emotion_prediction, depths = zip(*predictions_depths)
    M = max(all_emotion_prediction)
    occurrences = [index for index, value in enumerate(all_emotion_prediction) if value == M]

    if len(occurrences) == 1:
        return occurrences[0]
    elif len(occurrences) == 0:
        MAX = 0
        index = 0
        for i in range(0, len(depths)):
            if depths[i] > MAX:
                MAX = depths[i]
                index = i
        return index
    else:
        MIN = sys.maxsize
        index = 0
        for i in occurrences:
            if depths[i] < MIN:
                MIN = depths[i]
                index = i
        return index


'''
    x2 = test_df_data
'''
def test_forest_trees(forest_T, x2):
    predictions = []
    for i in x2.index.values:
        example = x2.loc[i]
        all_emotion_prediction = []
        for T in forest_T:
            emotion_prediction = []
            depths = []
            for tree in T:
                # how emotion votes
                prediction, depth = TreeNode.dfs_with_depth(tree, example)
                emotion_prediction.append(prediction)
                depths.append(depth)
            sum_per_emotion = sum(emotion_prediction)
            sum_depths = sum(depths)
            all_emotion_prediction.append((sum_per_emotion, sum_depths))

        prediction_choice = choose_majority_vote_optimised(all_emotion_prediction)
        predictions.append(prediction_choice + 1)
    return pd.DataFrame(predictions)

def apply_d_forest_parallel(df_labels, df_data, N):
    print(">> Running decision forest algorithm on multiple processes.\n")

    res = pd.DataFrame(0, index=cnst.EMOTIONS_INDICES, columns=cnst.EMOTIONS_INDICES)

    segments = util.preprocess_for_cross_validation(N)

    for test_seg in segments:
        print(">> Starting fold... from:", test_seg)
        print()

        forest_T = []
        test_df_data, test_df_targets, train_df_data, train_df_targets = util.divide_data(test_seg, N, df_data, df_labels)

        samples = split_in_random(train_df_data, train_df_targets)
        print("Building decision forest...")
        for e in cnst.EMOTIONS_LIST:
            T= []

            processes = []
            queue_list = []

            for (sample_target, sample_data) in samples:
                print("Building decision tree for emotion...", e)
                train_binary_targets = util.filter_for_emotion(sample_target, cnst.EMOTIONS_DICT[e])

                q = Queue()
                queue_list.append(q)

                process = Process(target=dtree.decision_tree_parallel, args=(sample_data, set(cnst.AU_INDICES), train_binary_targets, q))
                processes.append(process)
                process.start()

            for p in processes:
                p.join()

            for q in queue_list:
                T.append(q.get())

            forest_T.append(T)

        predictions_forest = test_forest_trees(forest_T, test_df_data)
        confusion_matrix = dtree.compare_pred_expect(predictions_forest, test_df_targets)
        print("----------------------------------- CONFUSION MATRIX -----------------------------------\n")
        print(confusion_matrix)
        res = res.add(confusion_matrix)

    diag_res = sum(pd.Series(np.diag(res),
                        index=[res.index, res.columns]))
    sum_all_res = res.values.sum()
    accuracy_res = (diag_res/sum_all_res) * 100
    print("-----------------------------------  AVERAGE ACCURACY -----------------------------------\n:", accuracy_res)

    # res = res.div(10)
    res = res.div(res.sum(axis=1), axis=0)
    for e in cnst.EMOTIONS_LIST:
        print("----------------------------------- MEASUREMENTS -----------------------------------")
        print(measures.compute_binary_confusion_matrix(res, cnst.EMOTIONS_DICT[e]))

    return res

'''
    Computes a confusion matrix using decison forests,
    improving the prediction accuracy.
'''
def apply_d_forest(df_labels, df_data, N):
    print(">> Running decision forest algorithm on a single process.\n")

    res = pd.DataFrame(0, index=cnst.EMOTIONS_INDICES, columns=cnst.EMOTIONS_INDICES)

    segments = util.preprocess_for_cross_validation(N)

    for test_seg in segments:
        print(">> Starting fold... from:", test_seg)
        print()

        forest_T = []
        test_df_data, test_df_targets, train_df_data, train_df_targets = util.divide_data(test_seg, N, df_data, df_labels)

        samples = split_in_random(train_df_data, train_df_targets)
        print("Building decision forest...")
        for e in cnst.EMOTIONS_LIST:
            T= []
            for (sample_target, sample_data) in samples:
                print("Building decision tree for emotion...", e)
                train_binary_targets = util.filter_for_emotion(sample_target, cnst.EMOTIONS_DICT[e])
                root = dtree.decision_tree(sample_data, set(cnst.AU_INDICES), train_binary_targets)
                print("Decision tree built. Now appending...\n")
                T.append(root)
            forest_T.append(T)

        predictions_forest = test_forest_trees(forest_T, test_df_data)
        confusion_matrix = dtree.compare_pred_expect(predictions_forest, test_df_targets)
        print("----------------------------------- CONFUSION MATRIX -----------------------------------\n")
        print(confusion_matrix)
        res = res.add(confusion_matrix)

    diag_res = sum(pd.Series(np.diag(res),
                        index=[res.index, res.columns]))
    sum_all_res = res.values.sum()
    accuracy_res = (diag_res/sum_all_res) * 100
    print("-----------------------------------  AVERAGE ACCURACY -----------------------------------\n:", accuracy_res)

    # res = res.div(10)
    res = res.div(res.sum(axis=1), axis=0)
    for e in cnst.EMOTIONS_LIST:
        print("----------------------------------- MEASUREMENTS -----------------------------------")
        print(measures.compute_binary_confusion_matrix(res, cnst.EMOTIONS_DICT[e]))

    return res
