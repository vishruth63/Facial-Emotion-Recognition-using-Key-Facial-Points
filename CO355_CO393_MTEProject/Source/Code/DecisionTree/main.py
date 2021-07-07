import time
import numpy as np
import sys

import cross_validation
import utilities as util
import decision_tree as dtree
import decision_forest as dforest

WRONG_ARGUMENTS_MSG = 'Wrong arguments.'

def convert_arguments():

    algorithm = None

    if (len(sys.argv) == 1):
        algorithm = dforest.apply_d_forest

    elif (len(sys.argv) == 2):
        tree_or_forest = sys.argv[1]
        if tree_or_forest == 'tree':
            algorithm = dtree.apply_d_tree
        elif tree_or_forest == 'forest':
            algorithm = dforest.apply_d_forest
        elif tree_or_forest in ['visualisation', 'visualization', 'visual', 'visualize']:
            algorithm = dtree.visualise
        else:
            print()
            sys.exit()

    elif (len(sys.argv) == 3):
        tree_or_forest = sys.argv[1]
        parallel = sys.argv[2]
        if (tree_or_forest != 'tree' and tree_or_forest != 'forest') or (parallel != 'multi' and parallel != 'single'):
            print(WRONG_ARGUMENTS_MSG)
            sys.exit()
        elif tree_or_forest == 'tree' and parallel == 'single':
            algorithm = dtree.apply_d_tree
        elif tree_or_forest == 'forest' and parallel == 'single':
            algorithm = dforest.apply_d_forest
        elif tree_or_forest == 'tree' and parallel == 'multi':
            algorithm = dtree.apply_d_tree_parallel
        else:
            algorithm = dforest.apply_d_forest_parallel
    else:
        print(WRONG_ARGUMENTS_MSG)
        sys.exit()
    return algorithm

# Testing
def main():

    algorithm = convert_arguments()

    START_TIME = time.time()

    labels, data = util.load_raw_data_clean()
    A = np.array(labels)
    labels = [row[0] for row in A]
    df_labels, df_data = util.to_dataframe(labels, data)

    # Number of examples
    N = df_labels.shape[0]
    print("----------------------------------- LOADING COMPLETED -----------------------------------\n")

    print("----------------------------------- CONFUSION_MATRIX ------------------------------------\n")
    res = algorithm(df_labels, df_data, N)
    print(res)

    print("----------------------------------- TOTAL EXECUTION TIME -----------------------------------\n")
    END_TIME = time.time()
    print(END_TIME - START_TIME)

if __name__ == "__main__": main()
