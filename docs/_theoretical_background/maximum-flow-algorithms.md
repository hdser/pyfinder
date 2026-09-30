---
layout: default
title: Maximum Flow Algorithms
parent: Theoretical Background
nav_order: 2
---

# Maximum Flow Algorithms in PyFinder

PyFinder implements multiple maximum flow algorithms, each with specific advantages for different network scenarios. This section details each algorithm's implementation, use cases, and performance characteristics.

## 1. Preflow Push Algorithm (Default)

The Preflow Push algorithm is PyFinder's default choice due to its excellent practical performance and ability to handle token-specific flows efficiently.

### Implementation Details

```python
class PreflowPush:
    def __init__(self, graph: Union[NetworkXGraph, GraphToolGraph]):
        self.graph = graph
        self.heights = {}
        self.excess = defaultdict(int)

    def compute_flow(self, source: str, sink: str) -> Tuple[int, Dict]:
        self._initialize_preflow(source)
        vertices = [v for v in self.graph.nodes() if v != source and v != sink]
        
        while self._has_active_vertex(vertices):
            v = self._get_active_vertex(vertices)
            if not self._push(v):
                self._relabel(v)

        return sum(self.excess[sink]), self._get_flow_dict()
```

### Key Features
- Maintains height function for vertices
- Local operations (push and relabel)
- Efficient handling of multiple token types
- Height-based optimization for intermediate nodes

### Performance Characteristics
```
Time Complexity: O(V²E)
Space Complexity: O(V² + E)
```

## 2. Edmonds-Karp Algorithm

The Edmonds-Karp algorithm is particularly effective for networks with relatively uniform token distributions.

### Implementation Details

```python
def edmonds_karp(self, source: str, sink: str) -> Tuple[int, Dict]:
    flow = 0
    flow_dict = defaultdict(lambda: defaultdict(int))

    while True:
        path = self._bfs_path(source, sink)
        if not path:
            break
            
        path_flow = self._compute_path_flow(path)
        flow += path_flow
        self._augment_flow(path, path_flow, flow_dict)

    return flow, dict(flow_dict)
```

### Optimizations for Token Networks
- BFS modified to respect token constraints
- Path flow computation considers token capacities
- Efficient handling of intermediate nodes

## 3. Boykov-Kolmogorov Algorithm

Specialized implementation for sparse trust networks with clustered trust relationships.

### Implementation Details

```python
class BoykovKolmogorov:
    def __init__(self, graph):
        self.graph = graph
        self.source_tree = set()
        self.sink_tree = set()
        self.orphans = deque()
        
    def compute_max_flow(self, source: str, sink: str):
        self._initialize_trees(source, sink)
        while self._grow_trees():
            self._augment_path()
            self._adopt_orphans()
```

### Key Features
- Maintains two search trees
- Efficient for clustered trust relationships
- Optimized for sparse networks

## 4. Dinitz Algorithm

Particularly effective for networks with high token diversity.

### Implementation Details

```python
class Dinitz:
    def __init__(self, graph):
        self.graph = graph
        self.level = {}
        self.next_edge = {}
        
    def compute_flow(self, source: str, sink: str):
        total_flow = 0
        while self._build_level_graph(source, sink):
            flow = self._blocking_flow(source, sink)
            if not flow:
                break
            total_flow += flow
```

### Optimization Features
- Level graph construction optimized for token constraints
- Blocking flow computation handles intermediate nodes
- Efficient path finding in residual network

## Performance Comparison

| Algorithm | Time Complexity | Space Complexity | Best Use Case |
|-----------|----------------|------------------|---------------|
| Preflow Push | O(V²E) | O(V² + E) | General purpose, large networks |
| Edmonds-Karp | O(VE²) | O(V + E) | Uniform token distribution |
| Boykov-Kolmogorov | O(VE²) | O(V + E) | Sparse, clustered networks |
| Dinitz | O(V²E) | O(V + E) | High token diversity |

## Implementation Considerations

### 1. Token-Specific Flow Handling

```python
def _compute_residual_capacity(self, u: str, v: str, token: str) -> int:
    """Compute residual capacity considering token constraints."""
    edge_data = self.graph.get_edge_data(u, v)
    if not edge_data:
        return 0
        
    capacity = edge_data.get('capacity', 0)
    current_flow = self.flow_dict[u].get(v, {}).get(token, 0)
    return capacity - current_flow
```

### 2. Intermediate Node Management

```python
def _create_intermediate_nodes(self, graph: NetworkXGraph) -> None:
    """Create and manage intermediate nodes for token flows."""
    for u, v, data in graph.edges(data=True):
        token = data['token']
        capacity = data['capacity']
        
        # Create intermediate node
        intermediate = f"{u}_{token}"
        self.graph.add_node(intermediate)
        
        # Split edge through intermediate node
        self.graph.add_edge(u, intermediate, capacity=capacity)
        self.graph.add_edge(intermediate, v, capacity=capacity)
```

### 3. Balance Conservation

```python
def _verify_balance_conservation(self, flow_dict: Dict) -> bool:
    """Verify that flow satisfies balance constraints."""
    for node in self.graph.nodes():
        if '_' not in node:  # Skip intermediate nodes
            incoming = sum(flow_dict.get(u, {}).get(node, 0) 
                         for u in self.graph.predecessors(node))
            outgoing = sum(flow_dict.get(node, {}).get(v, 0) 
                          for v in self.graph.successors(node))
            if incoming != outgoing and node not in (self.source, self.sink):
                return False
    return True
```

## Library-Specific Optimizations

### NetworkX Implementation

```python
class NetworkXFlow(BaseFlow):
    def __init__(self, graph: nx.DiGraph):
        self.g_nx = graph
        
    def compute_flow(self, source: str, sink: str, algorithm: str = 'preflow_push'):
        if algorithm == 'preflow_push':
            return nx.algorithms.flow.preflow_push(self.g_nx, source, sink)
        # ... other algorithm implementations
```

### graph-tool Implementation

```python
class GraphToolFlow(BaseFlow):
    def __init__(self, graph: Graph):
        self.g_gt = graph
        
    def compute_flow(self, source: str, sink: str, algorithm: str = 'push_relabel'):
        if algorithm == 'push_relabel':
            return gt.flow.push_relabel_max_flow(self.g_gt, 
                                               self.g_gt.vertex(source), 
                                               self.g_gt.vertex(sink))
        # ... other algorithm implementations
```

Would you like me to continue with more detailed documentation about:
1. The visualization components and their integration with the algorithms
2. The dashboard implementation and its interaction with the flow computation
3. The testing and benchmarking framework
4. Additional implementation details for specific algorithms?