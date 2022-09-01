"""
In degree graph module, a graph could be set where the same level
annotations are inflected to the same node, while the edges
representing the ordering relationship between two annotations.
"""

import numpy as np

from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.base import TransformerMixin

from ..base import KVNode
from ..utils._check import check_valid_int, check_pairwise_1d_array


class DegreeGraph(TransformerMixin):
    r"""A graph based structure that every node representing the same
    annotation on the same level.

    The idea is to construct a graph that every node stored not only
    a key but also a value representing the position that appears in
    the original target. Therefore, the same annotation on the same
    position will reflect to the same node. Therefore, the degree of
    one node could be seen as the variety of the difference that linked
    to that particular node.

    Attributes
    ----------
    nodes_ : np.ndarray, shape (n)
        The nodes appeared in the DegreeGraph.
    layers_ : np.ndarray, shape (n)
        The position of every node in the nodes.
    edges_ : np.ndarray, shape (2, e)
        The edge between two nodes with their index appeared in nodes.
    graph_dict_ : defaultdict
        The dictionary representation of a graph.
    entrance_nodes_ : np.ndarray, shape (m)
        The entrance node to the DegreeGraph.
    split_token_ : str
        The default token used to split the target into chunks of tokens.
    wild_token_ : str
        The wild token used to replace the merged tokens.
    deg_threshold_ : int, optional
            The threshold that nodes' variety over this threshold will
            be merged.
    failed_safe_ : str, optional
        The failed safe string used to replace the input which
        cannot be encoded.
    X_ : array_like, shape (n_samples)
        The input array.
    transformed_X_ : array_like, shape (n_samples)
        The transformed array.
    """

    def __init__(self, split_token='', wild_token='*',
                 deg_threshold=10,
                 failed_safe='invalid'):

        deg_threshold = check_valid_int(
            deg_threshold,
            lower=1,
            variable_name='degree threshold'
        )

        self.nodes_ = np.empty(0, dtype=np.object_)
        self.layers_ = np.empty(0, dtype=np.int_)
        self.edges_ = np.empty((2, 0), dtype=np.int_)
        self.graph_dict_ = defaultdict(list)

        self.entrance_nodes_ = np.empty(0, dtype=np.object_)

        self.split_token_ = split_token
        self.wild_token_ = wild_token
        self.deg_threshold_ = deg_threshold
        self.failed_safe_ = failed_safe

        return

    def _update_layers(self):
        r"""Update nodes layers according to nodes_.
        """

        self.layers_ = np.array(
            [_.value_ for _ in self.nodes_],
            dtype=np.int_
        )

        return

    def build_graph(self, targets):
        r"""Build degree graph from multiple targets.

        To build a degree graph, you also need to specify the split token
        to split the targets into chunks of tokens. After splitting, we
        build every target from scratch. The layer of node represents the
        location the token appeared, while edges reflect the ordering
        relationship between two nodes.

        Parameters
        ----------
        targets : list
            The list of targets to be used to set up a degree graph.
        split_token : str, optional
            The token used to split the target into chunks of tokens.
        """

        nodes_dict = {}
        edges = []

        for target in targets:
            fields = target.strip().split(self.split_token_) + ['<END>']

            pre_node = None
            for layer, field in enumerate(fields):
                node = KVNode(field, layer)

                if node not in nodes_dict:
                    nodes_dict[node] = len(nodes_dict)

                if pre_node is None:
                    pre_node = node
                else:
                    edges.append([nodes_dict[pre_node], nodes_dict[node]])
                    pre_node = node

        self.nodes_ = np.array(
            sorted(nodes_dict, key=lambda x: nodes_dict[x]),
            dtype=np.object_
        )
        self.edges_ = np.unique(edges, axis=0).T

        self._update_layers()

        return

    def _gen_graph_dict(self):
        r"""Generate the dictionary structure of DegreeGraph.
        """

        self.graph_dict_ = defaultdict(list)

        for start_node in np.unique(self.edges_[0]):
            end_nodes = self.edges_[1][self.edges_[0] == start_node]

            self.graph_dict_[self.nodes_[start_node]] = \
                self.nodes_[end_nodes]

        layers = np.array([_.value_ for _ in self.nodes_], dtype=np.int_)
        self.entrance_nodes_ = self.nodes_[layers == 0]

        return

    def merge_nodes(self, verbose=False):
        r"""Merge multiple nodes from the same layer.

        We try to merge the nodes based on the degree threshold. The merged
        nodes are on the same level, with the same upper and lower nodes.
        In other words, we try to merge the nodes with a large degree,
        therefore they will not be various nodes on the same layer making
        the tree have the large width.

        When merging nodes have multiple names, we use a wild character
        as the name of the new node. If there already exists a wild
        character on the same level, the node will be merged into it.

        Parameters
        ----------
        deg_threshold : int, optional
            The threshold that nodes' variety over this threshold will
            be merged.
        verbose : bool, optional
            Since the merging nodes inside DegreeGraph could be time
            consuming, when using verbose optional, a progress bar would
            show.
        """

        max_layer = np.amax(self.layers_)

        for start_layer in range(max_layer - 1):
            inter_layer = start_layer + 1

            start_nodes = np.where(self.layers_ == start_layer)[0]

            for start_node in tqdm(
                start_nodes,
                desc=f'Processing Layer {start_layer}',
                total=start_nodes.shape[0],
                ascii=True,
                disable=not verbose
            ):

                inter_nodes = \
                    self.edges_[1][self.edges_[0] == start_node]
                end_nodes, end_nodes_cnts = np.unique(
                    self.edges_[1][np.isin(self.edges_[0], inter_nodes)],
                    return_counts=True
                )

                for end_node in end_nodes[
                    end_nodes_cnts >= self.deg_threshold_
                ]:

                    common_inter_nodes = self.edges_[0][
                        (
                            np.isin(self.edges_[0], inter_nodes)
                        ) & (
                            self.edges_[1] == end_node
                        )
                    ]

                    if KVNode(self.wild_token_, inter_layer) not in self.nodes_:
                        self.nodes_ = np.concatenate((
                            self.nodes_,
                            [KVNode(self.wild_token_, inter_layer)]
                        ))

                    wild_character_node = np.where(
                        self.nodes_ == KVNode(self.wild_token_, inter_layer)
                    )[0][0]

                    upper = self.edges_[
                        :,
                        np.isin(self.edges_[1], common_inter_nodes)
                    ]
                    lower = self.edges_[
                        :,
                        np.isin(self.edges_[0], common_inter_nodes)
                    ]

                    self.edges_[1][
                        (
                            self.edges_[0] == start_node
                        ) & (
                            np.isin(self.edges_[1], common_inter_nodes)
                        ) & (
                            ~np.isin(
                                self.edges_[1],
                                lower[
                                    :,
                                    ~np.isin(
                                        lower[1],
                                        end_nodes[
                                            end_nodes_cnts >=
                                            self.deg_threshold_
                                        ]
                                    )
                                ][0]
                            )
                        )
                    ] = wild_character_node

                    self.edges_[0][
                        (
                            self.edges_[1] == end_node
                        ) & (
                            np.isin(self.edges_[0], common_inter_nodes)
                        ) & (
                            ~np.isin(
                                self.edges_[0],
                                upper[
                                    :,
                                    ~np.isin(
                                        upper[0],
                                        start_node
                                    )
                                ][1]
                            )
                        )
                    ] = wild_character_node

                    self.edges_ = np.hstack(
                        (
                            self.edges_,
                            np.array(
                                [
                                    [start_node, wild_character_node],
                                    [wild_character_node, end_node]
                                ]
                            )
                        )
                    )

                    self.edges_ = np.unique(self.edges_, axis=1)

            self._update_layers()

        self._gen_graph_dict()

        return

    def fit(self, X, Y=None, verbose=False):
        r"""Fit a DegGraph

        All the input data is provided by X, while Y is set to None
        to be ignored. In DegGraph, this function will copy the input X
        as the attribute and fit the DegGraph.

        It is a combination of operations of `build_graph` and `merge_nodes`.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.
        verbose : bool, optional
           Since the merging nodes inside DegreeGraph could be time
            consuming, when using verbose optional, a progress bar would
            show.

        Returns
        -------
        self : object
            DegGraph class object itself.
        """

        X, Y = check_pairwise_1d_array(X, Y, dtype=np.str_)

        self.X_ = X

        # DegGraph fit process
        self.build_graph(self.X_)
        self.merge_nodes(verbose=verbose)

        return self

    def _encode(self, graph_dict, reachable_nodes,
                fields, layer, encoded_fields,
                is_found=False):
        r"""The helper function for encode method.

        We try to encode the input targets by walking on the graph
        dictionary. Since we may or may not use the wild character
        on the certain layer, we try to encode the inputs in an
        iterative manner by trying out every possible encoding.

        Parameters
        ----------
        graph_dict : dictionary
            The dictionary representation of a graph.
        reachable_nodes : TYPE
            Current reachable nodes for certain layer of fields.
        fields : list
            The list of fields which could compose the whole target.
        layer : int
            The current layer to be encoded.
        encoded_fields : list
            The encoded fields so far.
        is_found : bool, optional
            If found a possible solution or not.

        Returns
        -------
        is_found : bool
            If found a possible solution or not.
        """

        if layer == len(fields):

            if KVNode('<END>', layer) in reachable_nodes:
                return True
            else:
                return False

        if KVNode(fields[layer], layer) in reachable_nodes:
            encoded_fields.append(fields[layer])
            is_found = self._encode(
                graph_dict, graph_dict[KVNode(fields[layer], layer)],
                fields, layer + 1, encoded_fields,
                is_found
            )

            if not is_found:
                encoded_fields.pop()

        if not is_found and KVNode(self.wild_token_, layer) in reachable_nodes:
            encoded_fields.append(self.wild_token_)
            is_found = self._encode(
                graph_dict, graph_dict[KVNode(self.wild_token_, layer)],
                fields, layer + 1, encoded_fields,
                is_found
            )

            if not is_found:
                encoded_fields.pop()

        return is_found

    def _transform_one(self, target):
        r"""Transform one using fitted DegGraph

        After the construction of a DegGraph, since we have merged
        a lot of nodes with the degree over the threshold and
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

        transformed_target = None

        layer = 0
        reachable_nodes = self.entrance_nodes_

        encoded_fields = []
        fields = target.strip().split(self.split_token_)

        is_found = self._encode(
            self.graph_dict_, reachable_nodes,
            fields, layer, encoded_fields
        )

        if is_found:
            transformed_target = self.split_token_.join(encoded_fields)
        else:
            transformed_target = self.failed_safe_

        return transformed_target

    def transform(self, X, Y=None):
        r"""Transform using fitted DegGraph

        All the input data is provided by X, while Y is set to None
        to be ignored. In DegGraph, this function actually transform
        the input data X to transfromed_X.

        After the construction of a DegGraph, since we have merged
        a lot of nodes with the degree over the threshold and
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
