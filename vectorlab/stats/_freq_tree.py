"""
In frequent tree module, a frequent tree could be set up to be
known as a FTree, which the nodes are represented as a combination
of words to be calculated the times they appeared in the certain
scope.
"""

import warnings
import numpy as np

from functools import reduce

from ..base import KVNode, SLMixin
from ..utils._check import check_valid_float


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

        super(FreqTreeNode, self).__init__(key, value)

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


class FreqTree(SLMixin):
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
    """

    def __init__(self):

        self.root_ = FreqTreeNode(None, np.Inf)

        return

    def build_tree(self, targets, split_token=''):
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
        split_token : str, optional
            The token used to split the target into chunks of tokens.
        """

        self.split_token_ = split_token

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
        \* as the name of the new node. If the merging nodes have the same
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
            nodes_key = '*'
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

        freq_threshold = check_valid_float(
            freq_threshold,
            lower=0.0, upper=1.0,
            variable_name='frequency threshold'
        )

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

    def merge_nodes(self, freq_threshold=0.1):
        r"""The merge nodes function.

        The entrance point to merge the nodes with the frequency
        value under the threshold.

        Parameters
        ----------
        freq_threshold : float, optional
            The threshold that nodes' frequency under this threshold
            will be merged.
        """

        for child in self.root_.children_.values():
            self._merge_nodes_helper(child, freq_threshold)

        return

    def encode(self, targets,
               split_token=None,
               failed_safe='invalid'):
        r"""Encode the input targets into a more general form.

        After the construction of a FreqTree, since we have merged
        a lot of nodes with the frequency under the threshold and
        replace them with a wild character \*. Therefore, we also
        want to encode the input that replace the merged tokens with
        \* accordingly. This encode function will serve this purpose.

        Parameters
        ----------
        targets : list
            The list of inputs that need to be encoded.
        split_token : str, optional
            The split token used to split the input.
        failed_safe : str, optional
            The failed safe string used to replace the input which
            cannot be encoded.

        Returns
        -------
        list
            The list of encoded inputs.
        """

        if split_token is None:
            split_token = self.split_token_
        else:
            warnings.warn(
                f'The original split token is {self.split_token_}, '
                f'and the new split token is {split_token}. This '
                f'may cause undesired results.'
            )

        encoded_targets = []

        for target in targets:

            cur_node = self.root_

            encoded_fields = []
            for field in target.strip().split(split_token):

                if field in cur_node.children_:
                    cur_node = cur_node.children_[field]
                    encoded_fields.append(cur_node.key_)
                elif '*' in cur_node.children_:
                    cur_node = cur_node.children_['*']
                    encoded_fields.append(cur_node.key_)
                else:
                    encoded_targets.append(failed_safe)
                    break
            else:

                if '<END>' in cur_node.children_:
                    encoded_targets.append(split_token.join(encoded_fields))
                else:
                    encoded_targets.append(failed_safe)

        return encoded_targets

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
