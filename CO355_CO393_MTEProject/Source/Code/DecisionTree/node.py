import plot
from multiprocessing import Queue
import pickle

# Tree Structure - TreeNode in Decision Tree
_node_index = 0
_edges = []
_labels = {}
_file_index = 0

class TreeNode:

    
    def __init__(self, node_label, leaf=False, value=None):
        self.op = node_label
        self.kids = [None] * 2
        self.leaf = leaf
        self.value = value
        global _node_index
        self.index = _node_index
        _node_index += 1

    def __str__(self):
        if self.leaf:
            return str(self.value)
        return str(self.op)

    def preorder_traversal(self):
        if self.op == None:
            if self.leaf:
                return self.value
            else:
                return "null"
        else:
            left = ""

            if self.kids[0] == None:
                left = "null"
            else:
                left = self.kids[0].preorder_traversal()

            if self.kids[1] == None:
                right = "null"
            else:
                right = self.kids[1].preorder_traversal()
            return str(self.op) + ", " + left + ", " + right

    def set_child(self, index, child):
        self.kids[index] = child

    def get_child(self, index):
        return self.kids[index]

    @staticmethod
    def dfs2(root, example, expectation):
        if root.leaf:
            is_correct = root.value == expectation
            return 1 if is_correct else 0
        else:
            index = root.op
            if example.ix[index] == 0:
                return TreeNode.dfs2(root.kids[0], example, expectation)
            else:
                return TreeNode.dfs2(root.kids[1], example, expectation)

    @staticmethod
    def dfs_parallel(root, example, queue):
        value = TreeNode.dfs(root, example)
        queue.put(value)

    @staticmethod
    def dfs(root, example):
        if root.leaf:
            return root.value
        else:
            index = root.op
            if example.loc[index] == 0:
                return TreeNode.dfs(root.kids[0], example)
            else:
                return TreeNode.dfs(root.kids[1], example)

    @staticmethod
    def dfs_with_depth(root, example, depth = 1):
        if root.leaf:
            return root.value, depth
        else:
            index = root.op
            if example.loc[index] == 0:
                return TreeNode.dfs_with_depth(root.kids[0], example, depth + 1)
            else:
                return TreeNode.dfs_with_depth(root.kids[1], example, depth + 1)

    @staticmethod
    def _dfs_pure(root):
        global _edges
        if root.leaf:
            _labels[root.index] = root.value
        else:
            _labels[root.index] = root.op
            for kid in root.kids:
                _edges.append((root.index, kid.index))
                TreeNode._dfs_pure(kid)

    @staticmethod
    def plot_tree(root, emotion="default_emotion"):
        global _file_index,_edges, _node_index, _labels
        _labels, _edges, _node_index = {}, [], 0
        TreeNode._dfs_pure(root)
        _file_index += 1
        plot.visualize_tree(_edges, _file_index, emotion=emotion, labels=_labels)

    @staticmethod
    def traverse(root):
        current_level = [root]
        while current_level:
            print(' '.join(str(node) for node in current_level))
            next_level = list()
            for n in current_level:

                if n.op == "'#'":
                    continue

                if n.kids[0]:
                    next_level.append(n.kids[0])
                else:
                    next_level.append(TreeNode("'#'"))
                if n.kids[1]:
                    next_level.append(n.kids[1])
                else:
                    next_level.append(TreeNode("'#'"))
            current_level = next_level
    @staticmethod
    def save_tree(tree, name):
        with open(str(name) + ".p", 'wb') as f:
            pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load_tree(name):
        with open(str(name) + ".p", "rb") as f:
            return pickle.load(f)
