"""Prim's algorithm running in time O((v + e) log v)."""

import sys
from typing import Iterable, TextIO, Optional

Node = int
Weight = float
Edge = tuple[Node, Weight, Node]


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
            self.edges[a].append((a, w, b))
            self.edges[b].append((b, w, a))

    def to_dot(self, f: TextIO = sys.stdout) -> None:
        """Print graph to f in .dot format."""
        print("graph { rankdir=LR;", file=f)
        for i, out_edges in enumerate(self.edges):
            for _, w, j in out_edges:
                if j > i:  # Only output one of the two
                    print(f'{i} -- {j} [label="{w}"]', file=f)
        print("}", file=f)


# Heap ###################################################
def _parent(i: int) -> int:
    """Get the parent index of i."""
    return (i - 1) // 2


def _left(i: int) -> int:
    """Get the left child of i."""
    return 2 * i + 1


def _right(i: int) -> int:
    """Get the right child of i."""
    return 2 * i + 2


def _get_optional(x: list[Node], i: int) -> Optional[Node]:
    """Get x[i] if i is a valid index, otherwise None."""
    try:
        return x[i]
    except IndexError:
        return None


class MinHeap:
    """
    Min-heap implementation using a binary heap.

    We don't use the heapq implementation since it doesn't
    give us the decrease weight functionality that we need.

    It also doesn't implement the full set of heap operations,
    since we only need a subset of them.
    """

    # We want a map from nodes dst to edges (src,w,dst)
    # so we can see what the cheapest way is to get to dst
    # from some node src already in the tree.
    #
    # At the same time, we want a heap for the dst nodes
    # weighted by w. To do this, we have a map from each
    # dst to the weights and src (self._weights and self._src),
    # indexed by node id. The nodes are in a heap,
    # src._nodes where they move around when we change their weight,
    # so we also need a map from a node id to its position in
    # the heap. The self._index list handles that, and it
    # contains None or node ids as nodes we have deleted
    # are no longer in the heap, but their id still exist
    # as an index. This also lets us recognise if we
    # try to insert an endge (src,w,dst) where dst is already
    # in the tree.
    #
    # The self._src can also contain None, but
    # this is for a different purpose; when we initialise the
    # heap there are no sources that leads into the nodes from
    # the tree, that is empty at this point. It means that we
    # have to check if the source node is None before we add
    # an edge to the tree, but it has the added benefit that
    # we can create a minimal spanning forest if the nodes
    # are not all connected in the input graph.

    # This is the list we use for the heap structure,
    # the others contain auxilary data
    _nodes: list[Node]
    _index: list[Optional[int]]  # Maps nodes to indices in _nodes
    _weights: list[Weight]      # The weight to get to a given node
    _src: list[Optional[Node]]  # The node that connects the tree to a node

    def __bool__(self) -> bool:
        """Tell us if there are more elements."""
        return bool(self._nodes)

    def _weight(self, i: int) -> Weight:
        """Get the weight for the node at index i."""
        return self._weights[self._nodes[i]]

    def _swap(self, i: int, j: int) -> None:
        """
        Swap index i and j.

        When swapping indices, we also need to update
        the mapping from nodes to index.
        """
        ni, nj = self._nodes[i], self._nodes[j]
        self._nodes[i], self._nodes[j] = self._nodes[j], self._nodes[i]
        self._index[ni], self._index[nj] = j, i  # ni sits at j now and nj at i

    def _min_child(self, i: int) -> Optional[int]:
        """Get the smallest child of i, if there is any."""
        l, r = _left(i), _right(i)

        # Get values, and handle out-of-bound at the same time
        l_node = _get_optional(self._nodes, l)
        if l_node is None:
            return None
        r_node = _get_optional(self._nodes, r)
        if r_node is None:
            return l

        # We have two values to pick the smallest from
        return l if self._weights[l_node] < self._weights[r_node] else r

    def _fix_up(self, i: int) -> None:
        """Move the value at nodes[i] up to its correct location."""
        while i > 0:
            p = _parent(i)
            if not self._weight(i) < self._weight(p):
                break  # we don't have to move up
            self._swap(i, p)
            i = p

    def _fix_down(self, i: int) -> None:
        """Move the value at nodes[i] down to its correct location."""
        while i < len(self._nodes):
            child = self._min_child(i)
            if child is None or self._weight(i) < self._weight(child):
                break
            self._swap(i, child)
            i = child

    def pop(self) -> tuple[Optional[Node], Weight, Node]:
        """Remove the smallest value and return it."""
        node = self._nodes[0]
        self._nodes[0], self._nodes[-1] = \
            self._nodes[-1], self._nodes[0]
        self._nodes.pop()
        self._fix_down(0)
        self._index[node] = None  # We no longer have this node
        return (self._src[node], self._weights[node], node)

    def decrease_weight(self, n: Node, w: Weight, src: Node) -> None:
        """Decrease the key for n according to the edge (src,w,n)."""
        # Only decrease if node is still here and the weight gets lowered!
        i = self._index[n]
        if i is not None and self._weight(i) > w:
            self._src[n] = src
            self._weights[n] = w
            self._fix_up(i)

    def __init__(self, no_nodes: int) -> None:
        """Initialise a new heap where all nodes have key inf."""
        # No need to heapify since all have the same key.
        self._nodes = list(range(no_nodes))
        self._index = list(range(no_nodes))
        self._weights = [float("inf")] * no_nodes
        self._src = [None] * no_nodes


# Prim ###################################################


def prim(graph: Graph) -> list[Edge]:
    """Run the O(e log e) Prim's algorithm."""
    # Initial tree
    tree: list[Edge] = []

    heap = MinHeap(graph.no_nodes)

    # We have an empty tree and a heap that contains all the
    # nodes. The nodes have infinite distance to the tree, but
    # that's okay. If we ask for a node with minimal weight we
    # will get an arbitrary one of them, and that is fine for
    # the algorithm.
    #
    # To build the tree, pop a minimal node out one at a time.
    # The heap.pop() function will give you an edge (src,w,dst),
    # but src will be None if dst doesn't have an edge from
    # the current tree. The first time you pop a node, src will
    # be None. If the nodes in the graph are not all connected,
    # it can happen more than once, but in the example this doesn't
    # happen.
    #
    # So, keep poping edges (src,w,dst) and if src is not None,
    # add the edge to the tree. Then consider all the new edges
    # out of dst, graph.edges[dst]. They might provide a cheaper
    # way to get to new nodes, so decrease the cost of getting to
    # each destination. Don't worry if the destination is already
    # in the tree or if the destination already has a cheaper path;
    # the decrease_weight() method will only update the weight
    # if you truely have a cheaper route to a node outside of
    # the tree.

    # FIXME: Algorithm needed here!
    ...

    return tree


if __name__ == '__main__':
    graph = Graph(
        (
            (0, 0.1, 2),
            (0, 2, 1),
            (2, 12, 3),
            (3, 2, 0),
            (3, 0.2, 4),
            (4, 1, 5),
            (5, 1, 2)
        )
    )
    # graph.to_dot()
    Graph(prim(graph)).to_dot()
