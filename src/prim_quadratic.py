"""Quadratic time Prim's algorithm."""

import sys
from typing import Iterable, TextIO

# (from, weight, to) == (to, weight, from)
Edge = tuple[int, float, int]


class Graph:
    """Adjecensy matrix representation of a graph."""

    no_nodes: int
    matrix: list[list[float]]

    def __init__(self, edges: Iterable[Edge]):
        """
        Create a new graph.

        The code doesn't do any consistency checks on the
        input, so be careful!
        """
        edge_list = list(edges)
        self.no_nodes = max(max(a, b) for (a, _, b) in edge_list) + 1
        self.matrix = [
            [float("inf")] * self.no_nodes
            for _ in range(self.no_nodes)
        ]
        for a, w, b in edge_list:
            self.matrix[a][b] = w
            self.matrix[b][a] = w

    def to_dot(self, f: TextIO = sys.stdout) -> None:
        """Print graph to f in .dot format."""
        print("graph { rankdir=LR;", file=f)
        for i in range(self.no_nodes):
            for j in range(i+1, self.no_nodes):
                w = self.matrix[i][j]
                if w < float("inf"):
                    print(f'{i} -- {j} [label="{w}"]', file=f)
        print("}", file=f)


def prim(graph: Graph) -> list[tuple[int, float, int]]:
    """Run the O(v²) Prim's algorithm."""
    # Initial tree
    tree: list[Edge] = []
    in_tree = [False] * graph.no_nodes

    # Start with node 0 in the tree
    in_tree[0] = True
    # Initial distance to nodes outside tree.
    dist = graph.matrix[0]
    # Which node gives us that distance
    src = [0] * graph.no_nodes

    # Now, iteratively, pick the next node and
    # add it to the tree. The next node should always
    # be the one we can reach with the lowest weight.

    for _ in range(graph.no_nodes - 1):
        # Get the best edge out
        best = dist.index(min(dist))  # O(v)

        # Add the destination node to the tree and the edge
        # to the tree, then remove this node as a candidate
        # for future steps.
        tree.append((src[best], dist[best], best))
        in_tree[best] = True
        dist[best] = float("inf")

        # Update the edges out of the tree. Also O(v)
        new_edges = [
            float("inf") if in_tree[i] else w
            for i, w in enumerate(graph.matrix[best])
        ]
        src = [
            src[i] if dist[i] < new_edges[i] else best
            for i in range(graph.no_nodes)
        ]
        dist = [
            min(a, b) for a, b in zip(dist, new_edges)
        ]

    return tree


if __name__ == '__main__':
    graph = Graph(((0, 0.1, 2), (0, 2, 1), (2, 12, 3),
                   (3, 2, 0), (3, 0.2, 4), (4, 1, 5),
                   (5, 1, 2)))
    # graph.to_dot()
    Graph(prim(graph)).to_dot()
