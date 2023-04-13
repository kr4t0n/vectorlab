"""
In frequent tree module, a frequent tree could be set up to be
known as a FTree, which the nodes are represented as a combination
of words to be calculated the times they appeared in the certain
scope.
"""

import numpy as np

from functools import reduce
from sklearn.base import TransformerMixin

from ..base import KVNode
from ..utils._check import check_valid_float, check_pairwise_1d_array


class FreqTreeNode(KVNode):
    r"""A nodes structure for FreqTree implementation.

    It is base node structure for FreqTree. The node contains
    one parent node and multiple children nodes. Since this
    FreqTreeNode contains multiple children, we use a dictionary
    to store all these nodes to achieve better efficiency when
    searching the child node.

    Parameters
    ----------
    key : str
        The key used to identify current node.
    value : int
        The valus stored in current node.

    Attributes
    ----------
    parent_ : FreqTreeNode
        The parent node for the current node.
    children_ : dict
        The children nodes for the current node.
    """

    def __init__(self, key, value):

        super().__init__(key, value)

        self.parent_ = None
        self.children_ = {}

        return

    def value_add_one(self):
        r"""The dummy function add one to the node value.
        """

        self.value_ += 1

        return

    def is_child(self, key):
        r"""Whether the given key exists in the children.

        Parameters
        ----------
        key : str
            The key to be tested whether existing in the children.

        Returns
        -------
        bool
            Whether the key is in the children or not.
        """

        return key in self.children_

    def is_parent(self, key):
        r"""Whether the given key is the parent.

        Parameters
        ----------
        key : str
            The key to be test whether being the parent.

        Returns
        -------
        bool
            Whether the key is the parent or not.
        """

        return key == self.parent_.key_


class FreqTree(TransformerMixin):
    r"""A tree structure that every node is a FreqTreeNode.

    The idea is to construct a prefix tree that every node
    stored not only a key but also a value. The key represents
    the node identity, while the depth of this node implies the
    location that this key appeared in the original target. The
    value represents the probability that the particular key
    appeared in this certain location among all the targets.

    Such tree can also be called as a frequency tree and used
    to encoding the input target.

    Attributes
    ----------
    root_ : FreqTreeNode
        The root node of the FreqTree.
    split_token_ : str
        The default token used to split the target into chunks of tokens.
    wild_token_ : str
        The wild token used to replace the merged tokens.
    freq_threshold_ : float, optional
        The threshold that nodes' frequency under this threshold
        will be merged.
    failed_safe_ : str, optional
        The failed safe string used to replace the input which
        cannot be encoded.
    X_ : array_like, shape (n_samples)
        The input array.
    transformed_X_ : array_like, shape (n_samples)
        The transformed array.
    """

    def __init__(self, split_token='', wild_token='*',
                 freq_threshold=0.1,
                 failed_safe='invalid'):

        freq_threshold = check_valid_float(
            freq_threshold,
            lower=0.0, upper=1.0,
            variable_name='frequency threshold'
        )

        self.root_ = FreqTreeNode(None, np.Inf)

        self.split_token_ = split_token
        self.wild_token_ = wild_token
        self.freq_threshold_ = freq_threshold
        self.failed_safe_ = failed_safe

        return

    def build_tree(self, targets):
        r"""Build frequency tree from multiple targets.

        To build a frequency tree, you also need to specify the
        split token to split the targets into chunks of tokens.
        After splitting, we build every target from the root
        of the tree. The level of node representing the location
        that key of node appeared in the target, while the value
        of that node referring to the number of appearance. In this
        case, the frequency tree is more alike to a prefix tree.

        Parameters
        ----------
        targets : list
            The list of targets to be used to set up a frequency tree.
        """

        for target in targets:

            cur_node = self.root_

            for field in target.strip().split(self.split_token_):

                if not cur_node.is_child(field):
                    cur_node.children_[field] = FreqTreeNode(field, 0)

                cur_node.children_[field].parent_ = cur_node
                cur_node = cur_node.children_[field]
                cur_node.value_add_one()

            if '<END>' not in cur_node.children_:
                end_node = FreqTreeNode('<END>', np.Inf)
                cur_node.children_['<END>'] = end_node

        return

    def _back_freq_transform(self, node):
        r"""The backward frequency transformation.

        It is a helper function to transform the frequency tree
        from a counting based tree to a frequency based tree. The
        frequency is calculated as the counting value of current
        node over the counting value of the parent node. Therefore,
        we try to transform in a reverse order manner, from the bottom
        to the top, which is called a backward frequency transformation.

        Parameters
        ----------
        node : FreqTreeNode
            The entrance node to do backward frequency transformation.
        """

        for child in list(node.children_.keys()):
            self._back_freq_transform(node.children_[child])

        if node.parent_ is None:
            node.value_ = np.Inf
        elif node.parent_.value_ == np.Inf:
            node.value_ = 1.0
        else:
            node.value_ /= float(node.parent_.value_)

        return

    def _forward_freq_transform(self, node):
        r"""The forward frequency transformation.

        It is a helper function to transform the frequency tree
        from a counting based tree to a frequency based tree. While the
        backward frequency transformation transforms the value of node
        from counting to frequency. For calculation convenience in other
        operations such as merge_nodes, we use the frequency that a certain
        node appeared from the root of tree rather than from their parent
        to be their actual frequency. Therefore, we also need to forward
        frequency transformation again, from the top to the bottom, calculated
        as the frequency of current node times the frequency of parent, which
        is called a forward frequency transformation.

        NOTICE: We treat the root node frequency as infinity while the first
        tier children of root all have the same frequency as one.

        Parameters
        ----------
        node : FreqTreeNode
            The entrance node to do forward frequency transformation.
        """

        if node.parent_ is None:
            node.value_ = np.Inf
        elif node.parent_.value_ == np.Inf:
            node.value_ = 1.0
        else:
            node.value_ *= float(node.parent_.value_)

        for child in list(node.children_.keys()):
            self._forward_freq_transform(node.children_[child])

        return

    def freq_transform(self):
        r"""The frequency transformation.

        This function transforms the frequency tree from a counting based
        tree to a frequency based tree. It first uses the backward frequency
        transformation to transform the values and uses the forward frequency
        transformation to update the values.
        """

        self._back_freq_transform(self.root_)
        self._forward_freq_transform(self.root_)

        return

    def _merge_nodes(self, parent, nodes):
        r"""Merge multiple nodes from the same parent.

        When merging nodes have multiple names, we use a wild character
        as the name of the new node. If the merging nodes have the same
        name, the new node will inherit it. The value of the new node is
        simply the sum of all the merging nodes.

        Remember, this merging operation will spread to all the sub-nodes
        involved. When a merging happens to some of the nodes, all their
        children will be checked whether they have the same name or not.
        For all the same named children, we also need to use this function
        to merge them accordingly.

        Parameters
        ----------
        parent : FreqTreeNode
            The parent node of all merging nodes.
        nodes : np.ndarray
            The array of FreqTreeNode that will be merged.
        """

        if nodes.shape[0] == 0:
            return

        nodes_keys = np.unique(np.array([node.key_ for node in nodes]))

        if nodes_keys.shape[0] > 1:
            nodes_key = self.wild_token_
        else:
            nodes_key = nodes_keys[0]

        # If the new nodes_key already exists, we also merge that node
        if nodes_key in parent.children_:
            if parent.children_[nodes_key] not in nodes:
                nodes = np.concatenate(
                    (
                        nodes,
                        [parent.children_[nodes_key]]
                    )
                )

        nodes_values = 0
        sub_nodes = []
        for node in nodes:

            nodes_values += node.value_

            if node.key_ in parent.children_:
                parent.children_.pop(node.key_)

            for child in node.children_:
                sub_nodes.append(node.children_[child])

        new_node = FreqTreeNode(nodes_key, nodes_values)
        new_node.parent_ = parent
        parent.children_[nodes_key] = new_node

        sub_nodes = np.array(sub_nodes)
        sub_nodes_keys = np.array([sub_node.key_ for sub_node in sub_nodes])

        for sub_node_key in np.unique(sub_nodes_keys):
            merge_sub_nodes = sub_nodes[sub_nodes_keys == sub_node_key]

            self._merge_nodes(new_node, merge_sub_nodes)

        return

    def _merge_nodes_helper(self, node, freq_threshold=0.1):
        r"""The merge nodes helper function.

        It is a helper function that merges nodes with the same
        parent according to their frequency. When the sub_nodes
        have the frequency under the threshold, we want to merge
        these nodes to create a new node to represent them.
        However, we only consider to merge the nodes whereas they
        have the common children. In other words, a special node
        with all its children different from others will not be merged.

        Parameters
        ----------
        node : FreqTreeNode
            The entrance node to do merge nodes operation.
        freq_threshold : float, optional
            The threshold that nodes' frequency under this threshold
            will be merged.
        """

        sub_nodes = []
        sub_nodes_values = []

        for child in node.children_:

            sub_nodes.append(node.children_[child])
            sub_nodes_values.append(node.children_[child].value_ / node.value_)

        sub_nodes = np.array(sub_nodes)
        sub_nodes_values = np.array(sub_nodes_values)

        if np.sum(sub_nodes_values <= freq_threshold) > 1:

            less_freq_sub_nodes = sub_nodes[sub_nodes_values <= freq_threshold]
            less_freq_sub_nodes_children = [
                list(less_freq_sub_node.children_.keys())
                for less_freq_sub_node in less_freq_sub_nodes
            ]

            symbols, counts = np.unique(
                reduce(lambda x, y: x + y, less_freq_sub_nodes_children),
                return_counts=True
            )
            single_symbol = symbols[counts == 1]

            merge_sub_nodes = less_freq_sub_nodes[
                ~np.array(
                    [
                        np.isin(
                            less_freq_sub_node_children,
                            single_symbol
                        ).all()
                        for less_freq_sub_node_children
                        in less_freq_sub_nodes_children
                    ]
                )
            ]

            self._merge_nodes(node, merge_sub_nodes)

        for child in node.children_:
            self._merge_nodes_helper(
                node.children_[child], freq_threshold
            )

        return

    def merge_nodes(self):
        r"""The merge nodes function.

        The entrance point to merge the nodes with the frequency
        value under the threshold.
        """

        for child in self.root_.children_.values():
            self._merge_nodes_helper(child, self.freq_threshold_)

        return

    def fit(self, X, Y=None):
        r"""Fit a FreqTree

        All the input data is provided by X, while Y is set to None
        to be ignored. In FreqTree, this function will copy the input X
        as the attribute and fit the FreqTree.

        It is a combination of operations of `build_tree`, `freq_transform`
        and `merge_nodes`.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        self : object
            FreqTree class object itself.
        """

        X, Y = check_pairwise_1d_array(X, Y, dtype=np.str_)

        self.X_ = X

        # FreqTree fit process
        self.build_tree(self.X_)
        self.freq_transform()
        self.merge_nodes()

        return self

    def _transform_one(self, target):
        r"""Transform one using fitted FreqTree

        After the construction of a FreqTree, since we have merged
        a lot of nodes with the frequency under the threshold and
        replace them with a wild character. Therefore, we also
        want to encode the input that replace the merged tokens with
        accordingly. This transform function will serve this purpose.

        Parameters
        ----------
        target : str
            The input target.

        Returns
        -------
        transformed_target : str
            The transformed target.
        """

        cur_node = self.root_
        transformed_target = None

        encoded_fields = []
        for field in target.strip().split(self.split_token_):

            if field in cur_node.children_:
                cur_node = cur_node.children_[field]
                encoded_fields.append(cur_node.key_)
            elif self.wild_token_ in cur_node.children_:
                cur_node = cur_node.children_[self.wild_token_]
                encoded_fields.append(cur_node.key_)
            else:
                transformed_target = self.failed_safe_
                break
        else:

            if '<END>' in cur_node.children_:
                transformed_target = self.split_token_.join(encoded_fields)
            else:
                transformed_target = self.failed_safe_

        return transformed_target

    def transform(self, X, Y=None):
        r"""Transform using fitted FreqTree

        All the input data is provided by X, while Y is set to None
        to be ignored. In FreqTree, this function actually transform
        the input data X to transfromed_X.

        After the construction of a FreqTree, since we have merged
        a lot of nodes with the frequency under the threshold and
        replace them with a wild character. Therefore, we also
        want to encode the input that replace the merged tokens with
        accordingly. This transform function will serve this purpose.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        transformed_X : array_like, shape (n_samples)
            The transformed input array.
        """

        X, Y = check_pairwise_1d_array(X, Y, dtype=np.str_)

        self.transformed_X_ = np.vectorize(
            self._transform_one, otypes=[np.str_]
        )(X)

        return self.transformed_X_

    def _repr_helper(self, node, return_str,
                     indent='', indent_width=4):
        r"""Representation helper function of FreqTree.

        The representation helper function that will try to
        illustrate the tree structure of FreqTree.

        Parameters
        ----------
        node : FreqTreeNode
            The entrance point to print the structure of the tree.
        return_str : list
            The list of tokens to be concatenated as a whole representation.
        indent : str, optional
            The indent symbol.
        indent_width : int, optional
            The indent width.
        """

        return_str.append(str(node.key_) + ', ' + str(node.value_) + '\n')

        if len(node.children_) != 0:
            children = list(node.children_.keys())

            for child in children[:-1]:
                return_str.append(indent + '├' + '-' * indent_width)
                self._repr_helper(
                    node.children_[child],
                    return_str,
                    indent + '│' + ' ' * indent_width
                )

            child = children[-1]
            return_str.append(indent + '└' + '-' * indent_width)
            self._repr_helper(
                node.children_[child],
                return_str,
                indent + ' ' * (indent_width + 1)
            )

        return

    def __repr__(self):
        r"""Representation function of FreqTree.

        The representation function that will be called if print
        function is used.

        Returns
        -------
        str
            The string to be printed.
        """

        return_str = []
        self._repr_helper(self.root_, return_str)

        return ''.join(return_str)
