#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Empty file"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea"]

import os
import json
import numpy as np
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# PARAMETERS
# =============================================================================

def _parse_args():
    """Analyses the received parameters and returns them organised.
    
    Takes the list of strings received at sys.argv and generates a
    namespace assigning them to objects.
    
    Returns
    -------
    out : namespace
        The namespace with the values of the received parameters
        assigned to objects.
    """
    
    # Generate the parameter analyser
    parser = ArgumentParser(description = __doc__,
                            formatter_class = RawDescriptionHelpFormatter)
    
    # Add arguments
    parser.add_argument("name",
                        choices=["BO", "IP", "KSC", "PU", "SV"],
                        help="Abbreviated name of the dataset.")
    
    # Return the analysed parameters
    return parser.parse_args()

# Tools
# =============================================================================

def dataframe_to_list_representation(num_classes, dataframe):
    current_tree = 0
    tree = []
    trees = [[] for i in range(num_classes)]
    for _, column in dataframe.iterrows():
        if column['tree_index'] > current_tree:
            class_num = current_tree%num_classes
            trees[class_num].append(tree)
            current_tree = column['tree_index']
            tree = []
        if column['split_feature']:
            node = [int(column['threshold']),
                    int(column['split_feature'].split('_')[1])]
        else:
            node = [column['value']]
        tree.append(node)
    trees[-1].append(tree)
    return trees

# Node class
# =============================================================================

class Node():
    """Node class to generate and manage decision trees
    
    Attributes
    ----------
    value : float, int or None
        For a leaf node the value is a float and corresponds to the
        prediction. For a non-leaf node the value is an int and
        corresponds to the comparison value. `None` represents an empty
        node.
    leaf : bool
        `True` for leaf nodes and `False` for non-leaf nodes.
    feature : int or None
        The number of the feature to compare with for non-leaf nodes
        and `None` for leaf nodes.
    left : Node or None
        The left child for non-leaf nodes and `None` for leaf nodes.
    right : Node or None
        The right child for non-leaf nodes and `None` for leaf nodes.
    
    Methods
    -------
    """
    
    def __init__(self, value=None, feature=None, left=None, right=None):
        """Inits a new node instance
        
        For a leaf node, `value` must be a float. If it receives a
        `float` as `value` it ignores the rest of the arguments.
        
        For a non-leaf node, `value` must be an int, and `feature` must
        be an int. For also filling both or one of the children, `left`
        and/or `right`, must be Node instances.
        
        If it receives `None` as `value` it generates an empty node.
        This is to allow working with incomplete trees.
        
        Parameters
        ----------
        value : float or int, optional (default: None)
            For a leaf node the value is a float and corresponds to the
            prediction. For a non-leaf node the value is an int and
            corresponds to the comparison value. `None` represents an
            empty node.
        feature : int, optional (default: None)
            The number of the feature to compare with for non-leaf
            nodes and `None` for leaf nodes.
        left : Node, optional (default: None)
            The left child for non-leaf nodes and `None` for leaf
            nodes.
        right : Node, optional (default: None)
            The right child for non-leaf nodes and `None` for leaf
            nodes.
        """
        
        if value is None:
            
            # It is an empty node
            self.leaf = True
            self.value = None
            self.feature = None
            self.left = None
            self.right = None
        
        elif isinstance(value, float):
            
            # It is a leaf node
            self.leaf = True
            self.value = value
            self.feature = None
            self.left = None
            self.right = None
        
        elif isinstance(value, int) and isinstance(feature, int):
            
            # It is a non-leaf node
            self.leaf = False
            self.value = value
            self.feature = feature
            
            if isinstance(left, Node):
                
                # It has a left child
                self.left = left
            
            else:
                
                # Left child is empty
                self.left = None
            
            if isinstance(right, Node):
                
                # It has a right child
                self.right = right
            
            else:
                
                # Right child is empty
                self.right = None
        
        else:
            
            raise TypeError(f"Incorrect parameters:\n{self.__doc__}")
    
    def insert_child(self, path, value, feature=None):
        """Inserts a new child node
        
        To insert a leaf node, `value` must be a float and `feature`
        should be empty or `None`.
        
        To insert a non-leaf node, `value` and `feature` must be ints.
        
        Parameters
        ----------
        path : str
            It represents the path where the node must be inserted
            within the tree as a string composed by `l` and `r` chars
            representing left or right movements.
        value : float or int
            For a leaf node the value is a float and corresponds to the
            prediction. For a non-leaf node the value is an int and
            corresponds to the comparison value.
        feature : int, optional (default: None)
            The number of the feature to compare with for non-leaf
            nodes and `None` for leaf nodes.
        """
        
        assert isinstance(path, str), "Parameter `path` must be of type `str`."
        err_msg = "The path should only consist on `l's` or `r's`."
        
        if not path:
            raise ValueError(err_msg)
        elif len(path) == 1:
            if path == 'l':
                self.left = Node(value, feature)
            elif path == 'r':
                self.right = Node(value, feature)
            else:
                raise ValueError(err_msg)
        else:
            if path[0] == 'l':
                self.left.insert_child(path[1:], value, feature)
            elif path[0] == 'r':
                self.right.insert_child(path[1:], value, feature)
            else:
                raise ValueError(err_msg)
    
    def reduce(self):
        """Returns reduced version of the tree as a new tree"""
        
        if self.leaf:
            
            # Return a leaf node with the same value
            return Node(self.value)
        
        else:
            
            # Calculate reduced children
            left = self.left.reduce()
            right = self.right.reduce()
            
            # If they are leafs with the same value
            if left.leaf and right.leaf:
                if left.value == right.value:
                    
                    # Return a leaf node with this value
                    return Node(left.value)
            
            # Return this node with the reduced children
            return Node(self.value, self.feature, left, right)
    
    def add_value(self, value):
        """Adds the received value to the value of every leaf node"""
        if self.leaf:
            
            # For leaf nodes, add the value
            self.value += value
        
        else:
            
            # For non-leaf nodes, propagate the value
            self.left.add_value(value)
            self.right.add_value(value)
    
    def dump(self):
        """Returns the lists representation of the tree"""
        
        if self.leaf:
            
            # Return the representation of a leaf node
            return [[self.value]]
        
        else:
            
            # Get children representations
            if self.left:
                left = self.left.dump()
            else:
                left = [[None]]
            if self.right:
                right = self.right.dump()
            else:
                right = [[None]]
            
            # Return the representation of the node and children
            return [[self.value, self.feature]] + left + right
    
    def fill(self, tree):
        """Fills the tree structure from a lists representation
        ¡ASUMES CORRECTNESS!
        """
        
        if len(tree[0]) == 1:
            
            # The entire tree if a leaf node
            self.leaf = True
            self.value = tree[0][0]
            self.feature = None
            self.left = None
            self.right = None
        
        else:
            
            # Generate the first node
            self.leaf = False
            self.value = tree[0][0]
            self.feature = tree[0][1]
            
            # Fill with the rest of nodes
            path = 'l'
            for node in tree[1:]:
                
                # For every node
                value = node[0]
                
                if len(node) == 1:
                    
                    # Leaf node
                    feature = None
                    
                    # Calculate next path
                    next_path = path[:path.rfind('l')] + 'r'
                    
                else:
                    
                    # Non-leaf node
                    feature = node[1]
                    
                    # Calculate next path
                    next_path = path + 'l'
                
                # Insert node
                self.insert_child(path, value, feature)
                path = next_path
    
    def inference(self, pixel):
        """Performs the inference process of `pixel`"""
        
        if self.leaf:
            
            # Return the prediction
            if self.value:
                return self.value
            else:
                return 0.0
        
        else:
            
            if pixel[self.feature] <= self.value:
                
                # Take left child
                return self.left.inference(pixel)
                
            else:
                
                # Take right child
                return self.right.inference(pixel)
    
    def _printable(self):
        """Returns a list of str to generate the tree representation"""
        
        if self.leaf:
            
            # Return a list with the value as a string
            if self.value is not None:
                printable = [f"{self.value:.3f}"]
            else:
                printable = ["XXXXX"]
        
        else:
            
            # Get children `printables`
            if self.left:
                left = self.left._printable()
            else:
                left = ["XXXXX"]
            if self.right:
                right = self.right._printable()
            else:
                right = ["XXXXX"]
            
            # Calculate spaces of the left child
            idx = 2
            for n, c in enumerate(left[0]):
                if c != ' ' and c != '_':
                    break
                idx += 1
            
            # Add left structure to node lines
            first_line = " "*(len(left[0]) - 2)
            second_line = " "*idx + "_"*(len(left[0]) - idx - 2)
            third_line = " "*idx + '|' + " "*(len(left[0]) - idx - 1)
            
            # Add cenrat structure to node lines
            first_line += f"f:{self.feature:03d}"
            second_line += f"{self.value:05d}"
            third_line += ' '
            
            # Calculate spaces of the right child
            idx = 0
            for n, c in enumerate(right[0]):
                if c != ' ' and c != '_':
                    break
                idx += 1
            
            # Add right structure to node lines
            first_line += " "*(len(right[0])  - 2)
            second_line += "_"*(idx + 1) + " "*(len(right[0]) - idx - 3)
            third_line += " "*(idx + 2) + '|' + " "*(len(right[0]) - idx - 3)
            
            # Generate the current node lines
            printable = [first_line, second_line, third_line]
            
            # Prepare children `printables` to be merged
            if len(left) > len(right):
                for i in range(len(left) - len(right)):
                    right.append(' '*len(right[0]))
            else:
                for i in range(len(right) - len(left)):
                    left.append(' '*len(left[0]))
            
            # Merge the children `printables`
            for left_line, right_line in zip(left, right):
                printable.append(left_line + ' ' + right_line)
        
        # Return current node printable
        return printable
    
    def __str__(self):
        """String representation of the node and its childs"""
        
        str = ""
        for line in self._printable():
            str += line + "\n"
        return str

# GBDT class
# =============================================================================

class GBDT():
    """GBDT class to manage gradient boosting decision trees"""
    
    def __init__(self, num_classes, dataframe=None):
        
        # Generates an empty GBDT instance
        self.trees = [[] for c in range(num_classes)]
        
        if dataframe is not None:
            
            # Fills the GBDT instance with the dataframe
            trees = dataframe_to_list_representation(num_classes, dataframe)
            self.fill(trees)
    
    def add_tree(self, class_num, Node):
        self.trees[class_num].append(Node)
    
    def reduce(self):
        for class_num, class_trees in enumerate(self.trees):
            for tree_num, tree in reversed(list(enumerate(class_trees))):
                reduced_tree = tree.reduce()
                if reduced_tree.leaf and tree_num > 0:
                    value = reduced_tree.value
                    self.trees[class_num][tree_num - 1].add_value(value)
                    del self.trees[class_num][tree_num]
                else:
                    self.trees[class_num][tree_num] = reduced_tree
    
    def dump(self):
        """Returns a lists representation of the trees"""
        return [[tree.dump() for tree in class_trees]
                for class_trees in self.trees]
    
    def save(self, file):
        """Saves the lists representation of the trees"""
        
        with open(file, 'w') as f:
            json.dump(self.dump(), f, indent=4)
    
    def fill(self, trees):
        """Fills the trees structure from the lists representation"""
        self.trees = [[Node() for tree in class_trees]
                      for class_trees in trees]
        for class_num, class_trees in enumerate(trees):
            for tree_num, tree in enumerate(class_trees):
                self.trees[class_num][tree_num].fill(tree)
    
    def load(self, file):
        """Loads the trees from the lists representation
        ¡ASUMES tree CORRECTNESS!
        """
        with open(file, 'r') as f:
            trees = json.load(f)
        self.fill(trees)
    
    def predict(self, pixels):
        """Performs the inference process of `pixels`"""
        
        predictions = []
        for pixel in pixels:
            prediction = []
            for class_trees in self.trees:
                value = 0.0
                for tree in class_trees:
                    value += tree.inference(pixel)
                prediction.append(value)
            predictions.append(prediction)
        return np.array(predictions)
    
#    def inference(self, pixel):
#        """Performs the inference process of `pixel`"""
#
#        predictions = []
#        for class_trees in self.trees:
#            prediction = 0.0
#            for tree in class_trees:
#                prediction += tree.inference(pixel)
#            predictions.append(prediction)
#        return np.argmax(predictions)
#
#    def predict(self, pixels, labels):
#        """Performs `pixels` inference and returns accuracy"""
#
#        hits = 0
#        for pixel, label in zip(pixels, labels):
#            if self.inference(pixel) == label:
#                hits += 1
#        return hits/len(pixels)
    
    def __str__(self):
        printable = ""
        for class_num, class_trees in enumerate(self.trees):
            printable += f"Trees of class {class_num}\n" + '_'*80 + "\n\n"
            for tree in class_trees:
                printable += str(tree) + "\n"
        return printable

# MAIN FUNCTION
# =============================================================================

def main(name):
    """Main function
    
    Parameters
    ----------
    args : namespace
        The received parameters.
    """
    
    print("\nIrreducible")
    print('_'*80 + "\n")

    tree = Node(12345, 32)

    tree.insert_child("l", 3456, 45)
    tree.insert_child("r", 34567, 23)

    tree.insert_child("ll", 45678, 12)
    tree.insert_child("lr", 0.86)
    tree.insert_child("rl", 0.14)
    tree.insert_child("rr", 0.23)

    tree.insert_child("lll", 0.94)
    tree.insert_child("llr", 0.06)

    print(tree)

    print("\nAdded 0.01 to every leaf node:\n")
    tree.add_value(0.01)
    print(tree)

    print("\nReduced tree:\n")
    print(tree.reduce())

    print("\nDumped tree:\n")
    print(tree.dump())

    print("\nDumped and loaded on new tree:\n")
    dumped_tree = tree.dump()
    new_tree = Node()
    new_tree.load(dumped_tree)
    print(new_tree)

    print("\nReducible")
    print('_'*80 + "\n")

    tree = Node(12345, 32)

    tree.insert_child("l", 3456, 45)
    tree.insert_child("r", 34567, 23)

    tree.insert_child("ll", 45678, 12)
    tree.insert_child("lr", 0.86)
    tree.insert_child("rl", 0.14)
    tree.insert_child("rr", 0.23)

    tree.insert_child("lll", 0.86)
    tree.insert_child("llr", 0.86)

    print(tree)

    print("\nReduced tree:\n")
    print(tree.reduce())

    print("\nDumped reduced tree:\n")
    print(tree.reduce().dump())

    print("\nDumped and loaded on new tree:\n")
    dumped_tree = tree.reduce().dump()
    new_tree = Node()
    new_tree.load(dumped_tree)
    print(new_tree)

if __name__ == "__main__":
    
    # Parse args
    args = _parse_args()
    
    # Launch main function
    main(args.name)
