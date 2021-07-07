import pandas as pd
import scipy.io as sio
import constants as cnst
import os

from node import TreeNode

pd.options.mode.chained_assignment = None

#  ------------------ INPUT ------------------

'''
    Loading data from mat files
'''
def load_raw_data(path):
    print("Loading raw data...")
    mat_contents = sio.loadmat(path)
    data = mat_contents['x']   # entries/lines which contain the activated AU/muscles
    labels = mat_contents['y'] # the labels from 1-6 of emotions for each entry in data
    print("Raw data loaded.\n")
    return labels, data

def load_raw_data_clean():
    return load_raw_data(cnst.CLEAN_DATA_PATH)

def load_raw_data_noisy():
    return load_raw_data(cnst.NOISY_DATA_PATH)


# ------------------ DATA MANIPULATION ------------------

'''
    Converting data to DataFrame format
'''
def to_dataframe(labels, data):
    print("Converting to data frame started...")
    df_labels = pd.DataFrame(labels)
    df_data = pd.DataFrame(data, columns=cnst.AU_INDICES)
    print("Converting to data frame done.\n")
    return df_labels, df_data
'''
    Filter a vector in df format to be 1 where we have
    this certain emotion and 0 otherwise
    emotion is an int
'''
def filter_for_emotion(df, emotion):
    print("Filtering to binary targets for emotion...", emotion)
    df_filter = df.copy(deep=True)
    df_filter.loc[(df_filter[0] > emotion) | (df_filter[0] < emotion), 0] = 0
    df_filter.loc[df_filter[0] == emotion, 0] = 1
    print("Filtering done.")
    return df_filter

'''
    Computes list [(start, end)] of the limits of K segments
    used in cross validation in a df of length N
'''
def preprocess_for_cross_validation(N, K = 10):
    seg_size = int(N/K)     # Size of a block of examples/targets
    segments = [(i - seg_size, i - 1) for i in range(seg_size, N-+1, seg_size)]
    segments[-1] = (segments[-1][0], N-1)
    return segments

"""
    Slices the given df_data and df_labels [from_index : to_index + 1]
"""
def slice_segments(from_index, to_index, df_data, df_labels):
    return df_data[from_index : to_index + 1], df_labels[from_index : to_index + 1]

'''
    Divides the given df_data and df_labels the following away
    test_data, test_labels = (test_seg/N) %
    train_data, train_labels = (N-test_seg/N) %
'''
def divide_data(test_seg, N, df_data, df_labels):
    (test_start, test_end) = test_seg
    test_df_data, test_df_targets = slice_segments(test_start, test_end, df_data, df_labels)

    if test_start == 0:
        # test is first segment
        train_df_data, train_df_targets = slice_segments(test_end + 1, N-1, df_data, df_labels)
    elif test_end == N - 1:
        # test is last segment
        train_df_data, train_df_targets = slice_segments(0, test_start-1, df_data, df_labels)
    else:
        # test is middle segment
        data_p1,targets_p1 = slice_segments(0, test_start-1, df_data, df_labels)
        data_p2, targets_p2 = slice_segments(test_end+1, N-1, df_data, df_labels)

        train_df_data = pd.concat([data_p1, data_p2], axis=0)
        train_df_targets = pd.concat([targets_p1, targets_p2], axis=0)

    return test_df_data, test_df_targets, train_df_data, train_df_targets

# ------------------ TREE SERIALIZATION ------------------

def save_trees_to_file(trees):
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trees'), exist_ok=True)
    for i in range(len(trees)):
        TreeNode.save_tree(trees[i], 'trees/tree_' + str(i))

def save_forest_to_file(trees):
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trees'), exist_ok=True)
    for i in range(len(trees)):
        TreeNode.save_tree(trees[i], 'trees/forest_' + str(i))

def load_trees(nr_of_trees):
    trees = []
    for i in range(nr_of_trees):
        trees.append(TreeNode.load_tree("trees/tree_" + str(i)))
    return trees

def load_forest(nr_of_trees):
    trees = []
    for i in range(nr_of_trees):
        trees.append(TreeNode.load_tree("trees/forest_" + str(i)))
    return trees
