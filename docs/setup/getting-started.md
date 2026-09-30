---
layout: default
title: Getting Started
nav_order: 2
---

# Getting Started with PyFinder
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

## Installation

### Prerequisites

Before installing PyFinder, ensure you have:

- Python 3.8 or higher
- pip package manager
- (Optional) PostgreSQL for database integration
- (Optional) graph-tool library for enhanced performance

### Basic Installation

```bash
pip install pyfinder
```

### Development Installation

For development purposes:

```bash
git clone https://github.com/yourgithubusername/pyfinder.git
cd pyfinder
pip install -e ".[dev]"
```

## Quick Start Guide

### 1. Basic Flow Analysis

```python
from pyfinder import GraphManager

# Initialize with CSV files
manager = GraphManager(('trusts.csv', 'balances.csv'))

# Analyze flow
result = manager.analyze_flow(
    source='0x123...',
    sink='0x456...'
)

# Unpack results
flow_value, paths, edge_flows, original_flows = result
```

### 2. Using the Dashboard

```python
from pyfinder.dashboard import create_dashboard

# Create and display dashboard
dashboard = create_dashboard()
dashboard.show()
```

### 3. Database Integration

```python
# PostgreSQL configuration
db_config = {
    'host': 'localhost',
    'port': '5432',
    'dbname': 'circles',
    'user': 'user',
    'password': 'pass'
}

# Initialize with PostgreSQL
manager = GraphManager((db_config, 'queries'))
```

## Configuration

### Environment Variables

For PostgreSQL integration:

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=circles
POSTGRES_USER=user
POSTGRES_PASSWORD=pass
```

### Data File Format

#### Trust Relationships (CSV)

```csv
truster,trustee
0x123...,0x456...
0x789...,0xabc...
```

#### Account Balances (CSV)

```csv
account,tokenAddress,demurragedTotalBalance
0x123...,0x456...,1000000000000000000
0x789...,0xabc...,2000000000000000000
```

## Next Steps

- Learn about [Network Flow Theory](./network-flow-theory)
- Explore [Advanced Features](./advanced-features)
- Check out [API Reference](./api-reference)
- See [Examples](./examples)