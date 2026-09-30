---
layout: default
title: Network Flow Problem in Trust-Based Networks
parent: Theoretical Background
nav_order: 1
---

# Network Flow Problem in Trust-Based Networks

## Introduction

The network flow problem in trust-based networks presents unique challenges beyond traditional flow networks. In PyFinder, we address these challenges through a specialized implementation that handles multiple tokens, trust relationships, and balance constraints.

## Problem Definition

Given a trust-based network G = (V, E), where:
- V: Set of accounts (vertices)
- E: Set of trust relationships (edges)
- T: Set of tokens, where each token t ∈ T has its own flow constraints
- B(v, t): Balance of account v for token t
- Trust(u, v, t): Trust relationship from u to v for token t

The goal is to find the maximum possible flow from a source account s to a sink account t while respecting:
1. Trust relationships
2. Token balances
3. Conservation of value

## Mathematical Formulation

For each edge (u, v) ∈ E and token t ∈ T:

1. **Capacity Constraints**:
   ```
   0 ≤ f(u, v, t) ≤ min(B(u, t), Trust(u, v, t))
   ```

2. **Balance Conservation**:
   ```
   ∀v ∈ V - {s,t}, ∀t ∈ T:
   Σ f(u, v, t) = Σ f(v, w, t)
   ```

3. **Token Conservation**:
   ```
   ∀u ∈ V, ∀t ∈ T:
   Σ f(u, v, t) ≤ B(u, t)
   ```

## Implementation Approach

### 1. Intermediate Node Architecture

To handle token-specific flows, we implement an intermediate node structure:

```
Original Network:
A ---(Token1, 100)---> B

Transformed Network:
A --> A_Token1(100) --> B
```

This transformation:
- Ensures balance constraints per token
- Maintains trust relationship integrity
- Simplifies flow computation

### 2. Multi-Token Flow

Our implementation handles multiple tokens through:

```python
class TokenFlow:
    def __init__(self, token_address: str, capacity: int):
        self.token = token_address
        self.capacity = capacity
        self.flow = 0

class Edge:
    def __init__(self):
        self.token_flows = {}  # Map: token -> TokenFlow
```

### 3. Balance Conservation

Balance conservation is enforced through:
1. Intermediate nodes limiting total outflow
2. Per-token capacity constraints
3. Flow validation at each step

## Example Network

Consider this trust network:

```
     Token1(100)
A --------------> B
|                 |
| Token2(50)      | Token1(75)
|                 |
v                 v
C --------------> D
     Token2(30)
```

Transformed into:

```
                 Token1(100)
A --> A_Token1 --------------> B
|     A_Token2      B_Token1   |
|        |             |       |
|        v             v       |
|     Token2(50)   Token1(75)  |
|        |             |       |
v        v             v       v
C --> C_Token2 --------------> D
            Token2(30)
```
