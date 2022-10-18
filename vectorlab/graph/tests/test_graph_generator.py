import pytest

from vectorlab.graph import (
    graph_er_generator,
    graph_sbm_generator,
    graph_pa_generator,
    graph_mesh_generator,
    graph_random_generator
)

n_nodes = 100
n_clusters = 2
n_init_nodes = 50
layers_nodes = [50, 50]


def graph_er_generator_efficiency():
    _ = graph_er_generator(n_nodes)


def test_graph_er_generator_efficiency(benchmark):
    benchmark(graph_er_generator_efficiency)


def graph_sbm_generator_efficiency():
    _ = graph_sbm_generator(n_nodes, n_clusters)


def test_graph_sbm_generator_efficiency(benchmark):
    benchmark(graph_sbm_generator_efficiency)


def graph_pa_generator_efficiency():
    _ = graph_pa_generator(n_nodes, n_init_nodes)


def test_graph_pa_generator_efficiency(benchmark):
    benchmark(graph_pa_generator_efficiency)


def graph_mesh_generator_efficiency():
    _ = graph_mesh_generator(layers_nodes)


def test_graph_mesh_generator_efficiency(benchmark):
    benchmark(graph_mesh_generator_efficiency)


def graph_random_generator_efficiency():
    _ = graph_random_generator(n_nodes)


def test_graph_random_generator_efficiency(benchmark):
    benchmark(graph_random_generator_efficiency)
