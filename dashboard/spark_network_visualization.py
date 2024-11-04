import panel as pn
import param
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import monotonically_increasing_id, col, desc
from graphframes import GraphFrame
from pyvis.network import Network
import json
from pathlib import Path
import tempfile
import os
from typing import Dict, List, Optional, Tuple
import webbrowser
import traceback

from .visualization_component import VisualizationComponent
from src.graph import NetworkXGraph, GraphToolGraph



class SparkNetworkVisualization:
    """Component for large-scale network visualization using Spark and PyVis."""
    
    def __init__(self, spark_config: Optional[Dict[str, str]] = None):
        self.spark = self._initialize_spark(spark_config)
        self._verify_graphframes()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def _initialize_spark(self, config: Optional[Dict[str, str]] = None) -> SparkSession:
        """Initialize Spark session with GraphFrames."""
        builder = SparkSession.builder \
            .appName("NetworkVisualization") \
            .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12")
            
        if config:
            for key, value in config.items():
                builder = builder.config(key, value)
                
        return builder.getOrCreate()
    
    def _verify_graphframes(self):
        """Verify GraphFrames is properly loaded."""
        try:
            from graphframes import GraphFrame
            self.graphframes_available = True
        except ImportError:
            self.graphframes_available = False
            raise ImportError("GraphFrames not available. Check Spark configuration.")
            
    def process_network_data(self, 
                           edges_data: List[Tuple[str, str]], 
                           node_attributes: Optional[Dict[str, Dict]] = None,
                           max_nodes: int = 100) -> Network:
        """
        Process network data using Spark and create PyVis visualization.
        
        Args:
            edges_data: List of (source, target) tuples
            node_attributes: Optional dict of node attributes
            max_nodes: Maximum number of nodes to display
            
        Returns:
            PyVis Network object
        """
        # Create Spark DataFrame from edges
        schema = StructType([
            StructField("truster", StringType(), True),
            StructField("trustee", StringType(), True)
        ])
        
        edges_df = self.spark.createDataFrame(
            [(src, tgt) for src, tgt in edges_data],
            schema=schema
        )
        
        # Extract nodes and create IDs
        nodes_truster = edges_df.select(col("truster").alias("node")).distinct()
        nodes_trustee = edges_df.select(col("trustee").alias("node")).distinct()
        nodes = nodes_truster.union(nodes_trustee).distinct()
        vertices = nodes.withColumn("id", monotonically_increasing_id())
        
        # Create edge mapping
        node_to_id = vertices.select("node", "id")
        df_truster = node_to_id.withColumnRenamed("node", "truster").withColumnRenamed("id", "src")
        df_trustee = node_to_id.withColumnRenamed("node", "trustee").withColumnRenamed("id", "dst")
        
        edges = edges_df.join(df_truster, on="truster", how="inner") \
                       .join(df_trustee, on="trustee", how="inner") \
                       .select("src", "dst")
                       
        # Create GraphFrame
        g = GraphFrame(vertices, edges)
        
        # Run analysis algorithms
        clusters = g.labelPropagation(maxIter=15)
        pagerank = g.pageRank(resetProbability=0.15, maxIter=20)
        
        # Calculate degrees
        in_degrees = g.inDegrees
        out_degrees = g.outDegrees
        
        # Combine metrics
        metrics = (pagerank.vertices
                  .join(clusters, "id")
                  .join(in_degrees, "id", "outer")
                  .join(out_degrees, "id", "outer")
                  .join(node_to_id, "id")
                  .fillna(0))
                  
        # Select top nodes
        top_nodes = (metrics
                    .withColumn("total_degree", col("inDegree") + col("outDegree"))
                    .withColumn("score", col("pagerank") * col("total_degree"))
                    .orderBy(desc("score"))
                    .limit(max_nodes))
                    
        # Collect data
        top_node_ids = set(top_nodes.select("id").rdd.flatMap(lambda x: x).collect())
        filtered_edges = edges.filter(
            (col("src").isin(top_node_ids)) & 
            (col("dst").isin(top_node_ids))
        ).collect()
        
        filtered_nodes = top_nodes.collect()
        
        # Create visualization
        net = self._create_pyvis_network(filtered_nodes, filtered_edges)
        
        return net
    
    def _create_pyvis_network(self, nodes, edges) -> Network:
        """Create PyVis network with enhanced visualization settings."""
        net = Network(
            height="900px",
            width="100%",
            directed=True,
            bgcolor="#ffffff"
        )
        
        # Configure physics and layout
        options = {
            "physics": {
                "enabled": True,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -5000,
                    "centralGravity": 0.01,
                    "springLength": 20,
                    "springConstant": 0.8,
                    "damping": 0.4,
                    "avoidOverlap": 1.0
                },
                "stabilization": {
                    "enabled": True,
                    "iterations": 1000,
                    "updateInterval": 100
                }
            },
            "nodes": {
                "font": {"size": 16, "color": "#000000"}
            },
            "edges": {
                "smooth": False,
                "color": {"inherit": False}
            },
            "interaction": {
                "dragNodes": True,
                "hover": True,
                "zoomView": True,
                "navigationButtons": True
            }
        }
        
        net.set_options(json.dumps(options))
        
        # Add nodes with enhanced styling
        colors = [
            "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
            "#FFFF33", "#A65628", "#F781BF", "#999999", "#66C2A5"
        ]
        
        for node in nodes:
            cluster = int(node["label"]) % len(colors)
            degree = float(node["total_degree"])
            pagerank = float(node["pagerank"])
            
            size = min(50, max(20, 10 + (degree ** 0.5) * 3))
            
            net.add_node(
                node["id"],
                label=node["node"],
                color=colors[cluster],
                size=size,
                title=f"Cluster: {cluster}\nDegree: {int(degree)}\nPageRank: {pagerank:.4f}",
                group=cluster
            )
            
        # Add edges with styling
        for edge in edges:
            net.add_edge(
                edge["src"],
                edge["dst"],
                color={"color": "#666666", "opacity": 0.4},
                width=1
            )
            
        return net
    
    def save_visualization(self, net: Network, filename: str = "network.html"):
        """Save the network visualization to a file."""
        output_path = self.temp_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.generate_html()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(net.html)
        return output_path
    
    def cleanup(self):
        """Clean up temporary files and stop Spark session."""
        import shutil
        shutil.rmtree(self.temp_dir)
        self.spark.stop()

class SparkVisualizationComponent(VisualizationComponent):
    def __init__(self, **params):
        super().__init__(**params)
        self.spark_initialized = False
        self.temp_dir = Path(tempfile.mkdtemp())
        self._create_spark_controls()
        self.current_html = None

    def _create_spark_controls(self):
        """Create Spark-specific controls."""
        self.spark_status = pn.pane.Markdown(
            "Initializing Spark visualization...",
            styles={'color': 'gray'}
        )
        
        self.node_limit = pn.widgets.IntSlider(
            name='Node Limit',
            value=100,
            start=10,
            end=500,
            step=10
        )
        
        self.refresh_button = pn.widgets.Button(
            name="Refresh View",
            button_type="primary"
        )
        
        self.download_button = pn.widgets.Button(
            name="Download Visualization",
            button_type="success"
        )
        
        # Create visualization pane for network view using HTML pane
        self.viz_pane = pn.pane.HTML(
            "",
            height=800,
            sizing_mode='stretch_width'
        )
        
        self.controls = pn.Row(
            self.node_limit,
            self.refresh_button,
            self.download_button,
            self.spark_status,
            sizing_mode='stretch_width'
        )
        
        # Set up callbacks
        self.refresh_button.on_click(self._refresh_visualization)
        self.download_button.on_click(self._download_visualization)
        self.node_limit.param.watch(self._handle_limit_change, 'value')

    def _refresh_visualization(self, event=None):
        """Refresh the network visualization."""
        if hasattr(self, 'graph_manager') and self.graph_manager is not None:
            self._create_spark_visualization(self.graph_manager.graph)
        else:
            self.spark_status.object = "Graph manager not initialized."
            self.spark_status.styles = {'color': 'red'}

    def _handle_limit_change(self, event):
        """Handle changes to node limit."""
        self._refresh_visualization()

    def _download_visualization(self, event=None):
        """Save and open current visualization in browser."""
        try:
            if self.current_html is None:
                self.spark_status.object = "No visualization available for download"
                self.spark_status.styles = {'color': 'red'}
                return

            # Save the current HTML content to a file
            viz_path = self.temp_dir / 'network.html'
            with open(viz_path, 'w', encoding='utf-8') as f:
                f.write(self.current_html)

            # Open in browser
            webbrowser.open(f'file://{viz_path}')

            self.spark_status.object = "Visualization downloaded and opened in browser"
            self.spark_status.styles = {'color': 'green'}

        except Exception as e:
            print(f"Error downloading visualization: {str(e)}")
            self.spark_status.object = f"Error downloading: {str(e)}"
            self.spark_status.styles = {'color': 'red'}

    def initialize_graph(self, graph_manager):
        """Initialize graph with Spark visualization."""
        try:
            self.graph_manager = graph_manager

            if not self.spark_initialized:
                self._initialize_spark()

            # Create initial visualization
            self._create_spark_visualization(graph_manager.graph)

            self.spark_status.object = "Graph visualization ready"
            self.spark_status.styles = {'color': 'green'}

        except Exception as e:
            print(f"Error initializing graph: {str(e)}")
            self.spark_status.object = f"Error: {str(e)}"
            self.spark_status.styles = {'color': 'red'}

    def _initialize_spark(self):
        """Initialize Spark session."""
        try:
            self.spark = SparkSession.builder \
                .appName("NetworkVisualization") \
                .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
                .getOrCreate()

            self.spark_initialized = True
            print("Spark initialized successfully")

        except Exception as e:
            print(f"Error initializing Spark: {str(e)}")
            raise

    def _create_spark_visualization(self, graph):
        """Create network visualization using PyVis."""
        try:
            self.spark_status.object = "Creating visualization..."
            self.spark_status.styles = {'color': 'blue'}

            # Use your old script's logic to process the graph and create PyVis visualization
            # Start by converting the graph to edges_data
            if isinstance(graph, GraphToolGraph):
                edges_data = [(str(graph.vertex_id[e.source()]), str(graph.vertex_id[e.target()])) for e in graph.g_gt.edges()]
            else:
                edges_data = list(graph.g_nx.edges())

            # Now proceed with the old script's logic
            spark = self.spark

            # Create Spark DataFrame from edges
            schema = StructType([
                StructField("truster", StringType(), True),
                StructField("trustee", StringType(), True)
            ])

            df = spark.createDataFrame(
                [(src, tgt) for src, tgt in edges_data],
                schema=schema
            )

            # Create vertices and edges
            nodes_truster = df.select(col("truster").alias("node")).distinct()
            nodes_trustee = df.select(col("trustee").alias("node")).distinct()
            nodes = nodes_truster.union(nodes_trustee).distinct()
            vertices = nodes.withColumn("id", monotonically_increasing_id())
            node_to_id = vertices.select("node", "id")

            df_truster = node_to_id.withColumnRenamed("node", "truster").withColumnRenamed("id", "src")
            df_trustee = node_to_id.withColumnRenamed("node", "trustee").withColumnRenamed("id", "dst")

            edges = df.join(df_truster, on="truster", how="inner") \
                      .join(df_trustee, on="trustee", how="inner") \
                      .select("src", "dst")

            # Create GraphFrame and run enhanced community detection
            g = GraphFrame(vertices, edges)

            # Run Label Propagation for community detection
            clusters = g.labelPropagation(maxIter=15).withColumnRenamed("label", "cluster")

            # Calculate both in and out degrees
            in_degrees = g.inDegrees
            out_degrees = g.outDegrees

            # Join degrees and calculate total degree
            degrees = out_degrees.join(in_degrees, "id", "outer").fillna(0)
            degrees = degrees.withColumn(
                "total_degree",
                col("inDegree") + col("outDegree")
            )

            # Run PageRank with adjusted parameters
            results = g.pageRank(resetProbability=0.15, maxIter=20)

            # Combine all metrics
            metrics = (results.vertices
                      .join(clusters, "id")
                      .join(degrees, "id", "outer")
                      .join(node_to_id, "id")
                      .fillna(0))

            # Select top nodes based on total degree and PageRank
            top_n = self.node_limit.value
            top_nodes = (metrics
                        .withColumn("score", col("pagerank") * col("total_degree"))
                        .orderBy(desc("score"))
                        .limit(top_n))

            # Collect node and edge data
            top_node_ids = set(top_nodes.select("id").rdd.flatMap(lambda x: x).collect())
            filtered_edges = edges.filter(
                (col("src").isin(top_node_ids)) & 
                (col("dst").isin(top_node_ids))
            ).collect()

            filtered_nodes = top_nodes.collect()

            # Create cluster size mapping
            cluster_sizes = {}
            for node in filtered_nodes:
                cluster = node["cluster"]
                cluster_sizes[cluster] = cluster_sizes.get(cluster, 0) + 1

            # Sort clusters by size and assign colors accordingly
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
            cluster_mapping = {cluster: idx for idx, (cluster, _) in enumerate(sorted_clusters)}

            # Enhanced color palette with greater contrast
            predefined_colors = [
                "#E41A1C",  # Red
                "#377EB8",  # Blue
                "#4DAF4A",  # Green
                "#984EA3",  # Purple
                "#FF7F00",  # Orange
                "#FFFF33",  # Yellow
                "#A65628",  # Brown
                "#F781BF",  # Pink
                "#999999",  # Gray
                "#66C2A5",  # Teal
                "#FC8D62",  # Coral
                "#8DA0CB",  # Light Blue
                "#E78AC3",  # Light Pink
                "#A6D854",  # Light Green
                "#FFD92F"   # Gold
            ]

            # Initialize PyVis with updated physics configuration
            net = Network(
                height="800px",
                width="100%",
                directed=True,
                bgcolor="#ffffff",
                notebook=False
            )

            # Updated physics options for better cluster separation
            options = {
                "physics": {
                    "enabled": True,
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {
                        "gravitationalConstant": -5000,
                        "centralGravity": 0.01,
                        "springLength": 20,
                        "springConstant": 0.8,
                        "damping": 0.4,
                        "avoidOverlap": 1.0
                    },
                    "minVelocity": 0.75,
                    "maxVelocity": 50,
                    "timestep": 0.35,
                    "stabilization": {
                        "enabled": True,
                        "iterations": 10000,
                        "updateInterval": 1000,
                        "fit": True
                    }
                },
                "layout": {
                    "randomSeed": 42,
                    "improvedLayout": True,
                    "clusterThreshold": 0
                },
                "nodes": {
                    "font": {
                        "size": 16,
                        "face": "arial",
                        "color": "#000000"
                    },
                    "fixed": {
                        "x": False,
                        "y": False
                    }
                },
                "edges": {
                    "smooth": False,
                    "color": {
                        "inherit": False
                    }
                },
                "interaction": {
                    "dragNodes": True,
                    "hover": True,
                    "zoomView": True,
                    "navigationButtons": True
                }
            }

            net.set_options(json.dumps(options))

            # Add nodes with modified mass for better cluster separation
            for row in filtered_nodes:
                cluster = row["cluster"]
                cluster_idx = cluster_mapping[cluster]
                degree = float(row["total_degree"])
                pagerank = float(row["pagerank"])

                # Adjust size for better visibility
                size = min(50, max(20, 10 + (degree ** 0.5) * 3))

                color = predefined_colors[cluster_idx % len(predefined_colors)]

                # Calculate mass based on connections to affect layout
                mass = 1 + (degree ** 0.5)  # Non-linear scaling for better distribution

                net.add_node(
                    row["id"],
                    label=row["node"],
                    color=color,
                    size=size,
                    title=f"Cluster: {cluster_idx}\nDegree: {int(degree)}\nPageRank: {pagerank:.4f}",
                    group=cluster_idx,
                    mass=mass,
                    font={'size': 16, 'color': '#000000'}
                )

            # Add edges with cluster-based styling
            edge_count = 0
            id_to_node = {row["id"]: row for row in filtered_nodes}
            for edge in filtered_edges:
                src_id = edge["src"]
                dst_id = edge["dst"]

                # Get cluster information for source and destination
                src_node = id_to_node.get(src_id)
                dst_node = id_to_node.get(dst_id)

                if src_node and dst_node:
                    same_cluster = src_node["cluster"] == dst_node["cluster"]

                    net.add_edge(
                        src_id,
                        dst_id,
                        color={
                            "color": "#666666",
                            "opacity": 0.4 if same_cluster else 0.1
                        },
                        width=2 if same_cluster else 0.5,
                        physics=True  # Enable physics for initial layout
                    )
                    edge_count += 1

            # Generate HTML content
            net_html = net.generate_html()

            # Update the visualization pane
            self.viz_pane.object = net_html
            self.current_html = net_html

            self.spark_status.object = f"Visualization ready ({len(filtered_nodes)} nodes shown)"
            self.spark_status.styles = {'color': 'green'}

        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            traceback.print_exc()
            self.spark_status.object = f"Error: {str(e)}"
            self.spark_status.styles = {'color': 'red'}

    def view(self):
        """Return the component's view."""
        return pn.Column(
            pn.Accordion(
                ("Network Overview", pn.Column(
                    self.controls,
                    self.viz_pane,
                    sizing_mode='stretch_width'
                )),
                active=[0],
                sizing_mode='stretch_width'
            ),
            # Parent class handles flow visualization
            super().view(),
            sizing_mode='stretch_width'
        )

    def update_view(self, results, computation_time, graph_manager, algorithm):
        """Update visualizations."""
        # Let parent class handle flow visualization
        super().update_view(results, computation_time, graph_manager, algorithm)

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'spark') and self.spark_initialized:
            self.spark.stop()
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
        super().cleanup()