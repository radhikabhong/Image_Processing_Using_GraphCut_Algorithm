"""
Edmonds-Karp algorithm for maximum flow problems.
"""

from collections import deque
import networkx as nx

"""
Utility classes and functions for network flow algorithms.
"""

def build_residual_network(graph, capacity):
    
    residual_nw = nx.DiGraph()
    residual_nw.add_nodes_from(graph)

    inf = float("inf")
    # Extract edges with positive capacities. Self loops excluded.
    edge_list = [
        (u, v, attr)
        for u, v, attr in graph.edges(data=True)
        if u != v and attr.get(capacity, inf) > 0
    ]
    
    inf = (
        3
        * sum(
            attr[capacity]
            for u, v, attr in edge_list
            if capacity in attr and attr[capacity] != inf
        )
        or 1
    )
    if graph.is_directed():
        for u, v, attr in edge_list:
            r = min(attr.get(capacity, inf), inf)
            if not residual_nw.has_edge(u, v):
                # Both (u, v) and (v, u) must be present in the residual
                # network.
                residual_nw.add_edge(u, v, capacity=r)
                residual_nw.add_edge(v, u, capacity=0)
            else:
                # The edge (u, v) was added when (v, u) was visited.
                residual_nw[u][v]["capacity"] = r
    else:
        for u, v, attr in edge_list:
            # Add a pair of edges with equal residual capacities.
            r = min(attr.get(capacity, inf), inf)
            residual_nw.add_edge(u, v, capacity=r)
            residual_nw.add_edge(v, u, capacity=r)

    # Record the value simulating infinity.
    residual_nw.graph["inf"] = inf

    return residual_nw


def edmonds_karp_core(residual_nw, s, t, cutoff):
    """Implementation of the Edmonds-Karp algorithm."""
    R_nodes = residual_nw.nodes
    R_pred = residual_nw.pred
    R_succ = residual_nw.succ

    inf = residual_nw.graph["inf"]

    def augment(path):
        """Augment flow along a path from s to t."""
        # Determine the path residual capacity.
        flow = inf
        it = iter(path)
        u = next(it)
        for v in it:
            attr = R_succ[u][v]
            flow = min(flow, attr["capacity"] - attr["flow"])
            u = v

        # Augment flow along the path.
        it = iter(path)
        u = next(it)
        for v in it:
            R_succ[u][v]["flow"] += flow
            R_succ[v][u]["flow"] -= flow
            u = v
        return flow

    def bidirectional_bfs():
        """Bidirectional breadth-first search for an augmenting path."""
        pred = {s: None}
        q_s = [s]
        succ = {t: None}
        q_t = [t]
        while True:
            q = []
            if len(q_s) <= len(q_t):
                for u in q_s:
                    for v, attr in R_succ[u].items():
                        if v not in pred and attr["flow"] < attr["capacity"]:
                            pred[v] = u
                            if v in succ:
                                return v, pred, succ
                            q.append(v)
                if not q:
                    return None, None, None
                q_s = q
            else:
                for u in q_t:
                    for v, attr in R_pred[u].items():
                        if v not in succ and attr["flow"] < attr["capacity"]:
                            succ[v] = u
                            if v in pred:
                                return v, pred, succ
                            q.append(v)
                if not q:
                    return None, None, None
                q_t = q

    # Look for shortest augmenting paths using breadth-first search.
    flow_value = 0
    while flow_value < cutoff:
        v, pred, succ = bidirectional_bfs()
        if pred is None:
            break
        path = [v]
        # Trace a path from s to v.
        u = v
        while u != s:
            u = pred[u]
            path.append(u)
        path.reverse()
        # Trace a path from v to t.
        u = v
        while u != t:
            u = succ[u]
            path.append(u)
        flow_value += augment(path)

    return flow_value


def edmonds_karp_impl(graph, s, t, capacity, residual, cutoff):
    """Implementation of the Edmonds-Karp algorithm."""

    if residual is None:
        residual_nw = build_residual_network(graph, capacity)
    else:
        residual_nw = residual

    # Initialize/reset the residual network.
    for u in residual_nw:
        for e in residual_nw[u].values():
            e["flow"] = 0

    if cutoff is None:
        cutoff = float("inf")
    residual_nw.graph["flow_value"] = edmonds_karp_core(residual_nw, s, t, cutoff)

    return residual_nw


def edmonds_karp(
    graph, s, t, capacity="capacity", residual=None, value_only=False, cutoff=None
):
    
    residual_nw = edmonds_karp_impl(graph, s, t, capacity, residual, cutoff)
    residual_nw.graph["algorithm"] = "edmonds_karp"
    return residual_nw