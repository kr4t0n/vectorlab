"""
In degree graph module, a graph could be set where the same level
annotations are inflected to the same node, while the edges
representing the ordering relationship between two annotations.
"""

import numpy as np

from tqdm.auto import tqdm
from collections import defaultdict

from ..base import SLMixin, KVNode


class DegreeGraph(SLMixin):
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
    """

    def __init__(self):

        self.nodes_ = np.empty(0, dtype=np.object_)
        self.layers_ = np.empty(0, dtype=np.int_)
        self.edges_ = np.empty((2, 0), dtype=np.int_)
        self.graph_dict_ = defaultdict(list)

        self.entrance_nodes_ = np.empty(0, dtype=np.object_)

        return

    def _update_layers(self):
        r"""Update nodes layers according to nodes_.
        """

        self.layers_ = np.array(
            [_.value_ for _ in self.nodes_],
            dtype=np.int_
        )

        return

    def build_graph(self, targets, split_token=''):
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
            fields = target.strip().split(split_token) + ['<END>']

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

    def merge_nodes(self, deg_threshold=10, verbose=False):
        r"""Merge multiple nodes from the same layer.

        We try to merge the nodes based on the degree threshold. The merged
        nodes are on the same level, with the same upper and lower nodes.
        In other words, we try to merge the nodes with a large degree,
        therefore they will not be various nodes on the same layer making
        the tree have the large width.

        When merging nodes have multiple names, we use a wild character
        \* as the name of the new node. If there already exists a wild
        character \* on the same level, the node will be merged into it.

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
                desc='Processing Layer {}'.format(start_layer),
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

                for end_node in end_nodes[end_nodes_cnts >= deg_threshold]:

                    common_inter_nodes = self.edges_[0][
                        (
                            np.isin(self.edges_[0], inter_nodes)
                        ) & (
                            self.edges_[1] == end_node
                        )
                    ]

                    if KVNode('*', inter_layer) not in self.nodes_:
                        self.nodes_ = np.concatenate((
                            self.nodes_,
                            [KVNode('*', inter_layer)]
                        ))

                    wild_character_node = np.where(
                        self.nodes_ == KVNode('*', inter_layer)
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
                                            end_nodes_cnts >= deg_threshold
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
            is_found = self._encode(graph_dict,
                                    graph_dict[KVNode(fields[layer], layer)],
                                    fields, layer + 1, encoded_fields,
                                    is_found)

            if not is_found:
                encoded_fields.pop()

        if not is_found and KVNode('*', layer) in reachable_nodes:
            encoded_fields.append('*')
            is_found = self._encode(graph_dict,
                                    graph_dict[KVNode('*', layer)],
                                    fields, layer + 1, encoded_fields,
                                    is_found)

            if not is_found:
                encoded_fields.pop()

        return is_found

    def encode(self, targets,
               split_token='',
               failed_safe='invalid'):
        r"""Encode the input targets into a more general form.

        After the construction of a DegreeGraph, since we have merged
        a lot of nodes with the degree over the degree threshold and
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

        encoded_targets = []

        for target in targets:

            layer = 0
            reachable_nodes = self.entrance_nodes_

            encoded_fields = []
            fields = target.strip().split(split_token)

            is_found = self._encode(self.graph_dict_, reachable_nodes,
                                    fields, layer, encoded_fields)

            if is_found:
                encoded_targets.append(split_token.join(encoded_fields))
            else:
                encoded_targets.append(failed_safe)

        return encoded_targets
