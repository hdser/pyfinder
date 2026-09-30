---
layout: home
title: Home
nav_order: 1
---

# PyFinder Documentation
This documentation provides a comprehensive overview of the PyFinder project, which implements network flow algorithms for analyzing and optimizing value transfers in trust-based networks. It covers both theoretical foundations and practical implementations, including a sophisticated dashboard for visualization and analysis.

## Table of Contents

1. [Theoretical Background]({{ site.baseurl }}{% link _theoretical_background/index.md %})
   - [Network Flow Problem]({{ site.baseurl }}{% link _theoretical_background/network-flow-problem.md %})
   - [Maximum Flow Algorithms]({{ site.baseurl }}{% link _theoretical_background/maximum-flow-algorithms.md %})
   - [Balance Conservation]({{ site.baseurl }}{% link _theoretical_background/balance-conservation.md %})
   - [Intermediate Node Architecture]({{ site.baseurl }}{% link _theoretical_background/intermediate-node-architecture.md %})

2. [Implementation]({{ site.baseurl }}{% link _implementation/index.md %})
   - [Data Ingestion]({{ site.baseurl }}{% link _implementation/data-ingestion.md %})
   - [Graph Implementation]({{ site.baseurl }}{% link _implementation/graph-implementation.md %})
   - [Flow Analysis]({{ site.baseurl }}{% link _implementation/flow-analysis.md %})
   - [Visualization]({{ site.baseurl }}{% link _implementation/visualization.md %})

3. [Dashboard]({{ site.baseurl }}{% link _dashboard/index.md %})
   - [Architecture]({{ site.baseurl }}{% link _dashboard/architecture.md %})
   - [Components]({{ site.baseurl }}{% link _dashboard/components.md %})
   - [Interactive Visualization]({{ site.baseurl }}{% link _dashboard/interactive-visualization.md %})
   - [Configuration]({{ site.baseurl }}{% link _dashboard/configuration.md %})

4. [Benchmarks]({{ site.baseurl }}{% link _benchmarks/index.md %})
   - [Performance Analysis]({{ site.baseurl }}{% link _benchmarks/performance-analysis.md %})
   - [Algorithm Comparison]({{ site.baseurl }}{% link _benchmarks/algorithm-comparison.md %})
   - [Library Comparison]({{ site.baseurl }}{% link _benchmarks/library-comparison.md %})

## Project Overview

PyFinder tackles the complex challenge of finding optimal paths for value transfers in trust-based networks while respecting trust relationships and balance constraints. The project is built on two main pillars:

### Core Implementation
- Network flow algorithms using both NetworkX and graph-tool
- Intermediate node architecture for balance conservation
- Multiple data source support (CSV, PostgreSQL)
- Comprehensive visualization tools

### Interactive Dashboard
- Real-time flow analysis
- Interactive network visualization
- Multiple algorithm support
- Configurable parameters
- Result export capabilities

## Key Features

- **Multiple Flow Algorithms**: Implementation of various maximum flow algorithms:
  - Preflow Push (Default)
  - Edmonds-Karp
  - Shortest Augmenting Path
  - Boykov-Kolmogorov
  - Dinitz

- **Balance Conservation**: Novel intermediate node architecture ensuring proper balance constraints

- **Data Source Flexibility**: Support for:
  - CSV files
  - PostgreSQL database
  - Environment-based configuration

- **Visualization**: Sophisticated visualization tools for:
  - Network structure
  - Flow paths
  - Transfer patterns
  - Performance metrics

## Getting Started

For quick start guide and installation instructions, see:
- [Installation Guide]({{ site.baseurl }}{% link _implementation/installation.md %})
- [Quick Start]({{ site.baseurl }}{% link _implementation/quick-start.md %})
- [Configuration]({{ site.baseurl }}{% link _implementation/configuration.md %})