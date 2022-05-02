"""Prim's algorithm running in time O(e log e)."""

import sys
import heapq
from typing import Iterable, Iterator, TextIO, Any
from dataclasses import dataclass


@dataclass
class Edge:
    """This class just put the < order on edges."""

    src: int
    w: float
    dst: int

    def __lt__(self, other: Any) -> bool:
        """Tell us if self has lower weight than other."""
        assert isinstance(other, Edge)
        return self.w < other.w

    def __iter__(self) -> Iterator[Any]:
        """Unpacking edge."""
        yield self.src
        yield self.w
        yield self.dst


class Graph:
    """Adjecency list representation of a graph."""

    no_nodes: int
    edges: list[list[Edge]]

    def __init__(self, edges: Iterable[Edge]):
        """
        Create a new graph.

        The code doesn't do any consistency checks on the
        input, so be careful!
        """
        edge_list = list(edges)
        self.no_nodes = max(max(a, b) for (a, _, b) in edge_list) + 1
        self.edges = [
            [] for _ in range(self.no_nodes)
        ]
        for a, w, b in edge_list:
            self.edges[a].append(Edge(a, w, b))
            self.edges[b].append(Edge(b, w, a))

    def to_dot(self, f: TextIO = sys.stdout) -> None:
        """Print graph to f in .dot format."""
        print("graph { rankdir=LR;", file=f)
        for i, out_edges in enumerate(self.edges):
            for _, w, j in out_edges:
                if j > i:  # Only output one of the two
                    print(f'{i} -- {j} [label="{w}"]', file=f)
        print("}", file=f)


def prim(graph: Graph) -> list[Edge]:
    """Run the O(e log e) Prim's algorithm."""
    # Initial tree
    tree: list[Edge] = []
    seen = {0}  # Start with node 0 in the tree

    heap: list[Edge] = graph.edges[0][:]
    heapq.heapify(heap)

    # Now we have a heap of all the edges going out of the tree.
    # If we pick them in order of lowest weight, we get the
    # minimal tree.
    #
    # If we don't delete edges (_,_,b) where b is already in the
    # tree--and that would be a lot of extra work, you might get
    # such an edge out. Don't add that to the tree, b is already
    # there via a cheaper path; just throw those edges away.
    #
    # When you get a new edges (a, w, b), you need
    # to insert all the edges out of b: (b,w',c). You can skip edges
    # where c is already in the tree if you want, it will be a little
    # faster, but it won't be a problem if you add them. They will be
    # filtered away when you see them later anyway.

    # FIXME: Algorithm needed here!
    ...

    return tree


if __name__ == '__main__':
    graph = Graph(
        (
            Edge(0, 0.1, 2),
            Edge(0, 2, 1),
            Edge(2, 12, 3),
            Edge(3, 2, 0),
            Edge(3, 0.2, 4),
            Edge(4, 1, 5),
            Edge(5, 1, 2)
        )
    )
    # graph.to_dot()
    Graph(prim(graph)).to_dot()
