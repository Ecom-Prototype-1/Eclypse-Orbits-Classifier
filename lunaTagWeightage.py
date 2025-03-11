import pandas as pd
import random
import secrets
from typing import TypeVar, Generic, Dict, List, Set, Tuple, Callable, Union, Any
from enum import Enum
from collections import defaultdict
import time
import networkx as nx 
import plotly.graph_objects as go
import plotly.express as px

class Graph:
    """
    Graph data structure for the recommendation system
    Implements undirected graph with weighted random walk capabilities
    """
    
    def __init__(self):
        """Initialize an empty graph"""
        self.data = {}  # Dictionary to store nodes and their connections
        self.max_degree = 0
    
    def add_node(self, node):
        """
        Adds a node to the graph
        
        Args:
            node: The node to add
        """
        if node not in self.data:
            self.data[node] = set()
    
    def add_edge(self, node_a, node_b):
        """
        Adds an edge between two nodes in the graph
        Creates nodes if they don't exist
        
        Args:
            node_a: First node
            node_b: Second node
        """
        # Add or update node A's connections
        if node_a not in self.data:
            connections = {node_b}
            self.data[node_a] = connections
        else:
            connections = self.data[node_a]
            connections.add(node_b)
            self.data[node_a] = connections
        
        # Add or update node B's connections
        if node_b not in self.data:
            connections = {node_a}
            self.data[node_b] = connections
        else:
            connections = self.data[node_b]
            connections.add(node_a)
            self.data[node_b] = connections
        
        # Update max degree if needed
        degree_a = len(self.data[node_a])
        degree_b = len(self.data[node_b])
        
        if degree_a > self.max_degree:
            self.max_degree = degree_a
        
        if degree_b > self.max_degree:
            self.max_degree = degree_b
    
    def successors(self, node):
        """
        Gets all connected nodes to the given node
        
        Args:
            node: The node to find successors for
            
        Returns:
            Set of connected nodes
        """
        return set(self.data.get(node, set()))
    
    def get_max_degree(self):
        """
        Gets the maximum degree (number of connections) of any node in the graph
        
        Returns:
            The maximum degree
        """
        return self.max_degree
    
    def degree(self, node):
        """
        Gets the degree (number of connections) of a specific node
        
        Args:
            node: The node to check
            
        Returns:
            The degree of the node
        """
        return len(self.data.get(node, set()))
    
    @staticmethod
    def weighted_sample(elements, weight_fn):
        """
        Performs a weighted sampling from a list of elements
        
        Args:
            elements: The elements to sample from
            weight_fn: Function to determine weight of each element
            
        Returns:
            A selected element based on weights or None if none selected
        """
        # Safe weight function ensures no negative or infinite weights
        def safe_weight_fn(x):
            unsafe_weight = weight_fn(x)
            clamped_weight = max(0, unsafe_weight)
            return clamped_weight if isinstance(clamped_weight, (int, float)) and not (clamped_weight == float('inf')) else 0
        
        # Calculate total weight
        total_weight = sum(safe_weight_fn(elem) for elem in elements)
        
        if total_weight == 0:
            return None
        
        # Generate random value between 0 and total_weight
        random_value = random.random() * total_weight
        
        cumulative_weight = 0
        for elem in elements:
            cumulative_weight += safe_weight_fn(elem)
            if random_value <= cumulative_weight:
                return elem
        
        # Fallback in case of rounding errors
        return elements[-1] if elements else None
    
    def random_walk(self, starting_node, max_hops, weight_fn):
        """
        Performs a random walk on the graph starting from the given node
        
        Args:
            starting_node: The node to start from
            max_hops: Maximum number of hops to make
            weight_fn: Function to determine the weight/probability of moving from one node to another
            
        Returns:
            List of visited nodes in reverse order
        """
        visited = []
        
        if starting_node not in self.data:
            return visited
        
        current_node = starting_node
        hops = max_hops
        
        while hops > 0:
            hops -= 1
            visited.insert(0, current_node)
            
            successors = self.successors(current_node)
            successor_list = list(successors)
            
            if not successor_list:
                break
                
            next_node = self.weighted_sample(
                successor_list,
                lambda next_node: weight_fn(current_node, next_node)
            )
            
            if next_node is None:
                break
            
            current_node = next_node
        
        return visited

class NodeType(Enum):
    TAG = 'tag'
    OBJECT = 'object'

class RecommenderNode:
    """
    Represents a node in the recommendation graph
    Can be either a tag or an object
    """
    def __init__(self, node_type, value):
        self.type = node_type  # NodeType.TAG or NodeType.OBJECT
        self.value = value
    
    def __eq__(self, other):
        if not isinstance(other, RecommenderNode):
            return False
        return self.type == other.type and self.value == other.value
    
    def __hash__(self):
        return hash((self.type, str(self.value) if isinstance(self.value, dict) else self.value))
    
    def __repr__(self):
        return f"RecommenderNode({self.type}, {self.value})"

class Recommender:
    """
    Recommender system based on Pinterest's Pixie algorithm
    Works with objects and their associated tags
    """
    
    def __init__(self):
        """Initialize the recommender with an empty graph"""
        self.graph = Graph()
    
    def add_object(self, obj):
        """
        Adds an object to the recommender system
        
        Args:
            obj: The object to add
        """
        self.graph.add_node(RecommenderNode(NodeType.TAG if isinstance(obj, str) else NodeType.OBJECT, obj))
    
    def add_tag(self, tag):
        """
        Adds a tag to the recommender system
        
        Args:
            tag: The tag to add
        """
        self.graph.add_node(RecommenderNode(NodeType.TAG, tag))
    
    def tag_object(self, obj, tag):
        """
        Associates a tag with an object
        
        Args:
            obj: The object to tag
            tag: The tag to apply
        """
        object_node = RecommenderNode(NodeType.OBJECT, obj)
        tag_node = RecommenderNode(NodeType.TAG, tag)
        
        self.graph.add_edge(object_node, tag_node)
    
    def recommendations_map(
        self,
        from_node,
        depth,
        max_total_steps,
        weight_fn,
        visualize=False  # Visualize flag
    ):
        """
        Creates a map of recommendations with scores based on random walks
        
        Args:
            from_node: Starting node
            depth: Max depth for each walk
            max_total_steps: Total number of steps to take across all walks
            weight_fn: Function to determine edge weights during walk
            visualize: If True, visualize each random walk
            
        Returns:
            Dict of recommended nodes with their scores
        """
        scores = defaultdict(int)
        steps_remaining = max_total_steps
        
        walk_number = 0  # Counter for the walks
        all_walks = []  # Store all walks for later visualization
        
        while steps_remaining > 0:
            steps_to_take = min(depth, steps_remaining)
            
            visited = self.graph.random_walk(from_node, steps_to_take, weight_fn)
            steps_taken = len(visited)
            steps_remaining -= steps_taken
            
            if steps_taken <= 1:  # Only the starting node was visited or nothing
                break
            
            # Skip the first node (it's the starting node)
            for node in visited[1:]:
                scores[node] += 1
            
            if visualize:
                walk_number += 1
                all_walks.append(visited)
        
        # Remove the starting node from results
        if from_node in scores:
            del scores[from_node]
        
        # Visualize all walks at once if requested
        if visualize and all_walks:
            self.visualize_walks(all_walks, from_node)
            
        return dict(scores)
    
    def visualize_walks(self, all_walks, starting_node):
        """
        Visualizes all random walks on the graph using Plotly.
        
        Args:
            all_walks: List of lists, each containing nodes visited during a walk.
            starting_node: The node from which all walks began.
        """
        # Create a NetworkX graph for the structure
        G = nx.Graph()
        
        # Track all edges seen in the walks
        all_edges = set()
        
        # Add nodes and edges from all walks
        for walk in all_walks:
            for i in range(len(walk) - 1):
                G.add_edge(walk[i], walk[i + 1])
                all_edges.add((walk[i], walk[i + 1]))
        
        # Get positions using a force-directed layout
        pos = nx.spring_layout(G, seed=42)
        
        # Prepare node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Create label
            if node.type == NodeType.OBJECT:
                label = f"Product: {node.value.get('name', 'Unknown')}"
                color = 'rgb(66, 133, 244)'  # Blue for products
                size = 15
            else:
                label = f"Tag: {node.value}"
                color = 'rgb(15, 157, 88)'  # Green for tags
                size = 10
                
            node_text.append(label)
            node_colors.append(color)
            node_sizes.append(size)
            
        # Special size and color for starting node
        for i, node in enumerate(G.nodes()):
            if node == starting_node:
                node_colors[i] = 'rgb(244, 160, 0)'  # Orange for starting node
                node_sizes[i] = 20
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=1, color='rgb(50, 50, 50)')
            )
        )
        
        # Create edge traces - one for each walk with a different color
        edge_traces = []
        
        # Define a color palette for walks
        colors = px.colors.qualitative.Plotly
        
        for walk_idx, walk in enumerate(all_walks):
            walk_color = colors[walk_idx % len(colors)]
            
            edge_x = []
            edge_y = []
            
            for i in range(len(walk) - 1):
                from_node = walk[i]
                to_node = walk[i + 1]
                
                # Add edges for this walk
                x0, y0 = pos[from_node]
                x1, y1 = pos[to_node]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color=walk_color),
                hoverinfo='none',
                mode='lines',
                opacity=0.4,
                name=f'Walk {walk_idx+1}'
            )
            
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=f"Random Walks Visualization (Starting from {starting_node.value.get('name', starting_node.value) if starting_node.type == NodeType.OBJECT else starting_node.value})",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Random walks in the recommendation graph",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgb(248, 248, 248)'
            )
        )
        
        fig.show()
    
    def visualize_full_graph(self, max_nodes=100):
        """
        Visualizes the complete recommendation graph using Plotly.
        
        Args:
            max_nodes: Maximum number of nodes to display to prevent overcrowding
        """
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Get degrees for all nodes
        degrees = {}
        for node in self.graph.data:
            degrees[node] = self.graph.degree(node)
        
        # Sort nodes by degree and limit to max_nodes
        top_nodes = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)[:max_nodes]
        
        # Add top nodes to the graph
        for node in top_nodes:
            G.add_node(node)
        
        # Add edges between the top nodes
        for node in top_nodes:
            for neighbor in self.graph.successors(node):
                if neighbor in top_nodes:
                    G.add_edge(node, neighbor)
        
        # Get positions using a force-directed layout
        pos = nx.spring_layout(G, seed=42)
        
        # Prepare node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Create label
            if node.type == NodeType.OBJECT:
                label = f"Product: {node.value.get('name', 'Unknown')}<br>ID: {node.value.get('product_id', 'Unknown')}<br>Category: {node.value.get('category', 'Unknown')}"
                color = 'rgb(66, 133, 244)'  # Blue for products
                size = 10 + min(3 * degrees[node], 20)  # Size based on degree
            else:
                label = f"Tag: {node.value}<br>Connected Products: {degrees[node]}"
                color = 'rgb(15, 157, 88)'  # Green for tags
                size = 8 + min(2 * degrees[node], 15)  # Size based on degree
                
            node_text.append(label)
            node_colors.append(color)
            node_sizes.append(size)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=1, color='rgb(50, 50, 50)')
            )
        )
        
        # Create edge trace
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='rgb(180, 180, 180)'),
            hoverinfo='none',
            mode='lines',
            opacity=0.3,
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Product-Tag Network Visualization",
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text=f"Showing top {len(G.nodes())} nodes by connectivity",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgb(248, 248, 248)'
            )
        )
        
        fig.show()
    
    def recommend_objects(
        self,
        from_obj,
        depth=5,
        max_total_steps=1000,
        weight_fn=None,
        limit=10,
        visualize=False
    ):
        """
        Recommends objects based on the given object
        
        Args:
            from_obj: Object to base recommendations on
            depth: Max depth for each walk
            max_total_steps: Total number of steps to take across all walks
            weight_fn: Optional custom weight function for the random walk
            limit: Maximum number of recommendations to return
            visualize: If True, visualize the recommendation process
            
        Returns:
            List of tuples containing recommended objects and their scores, sorted by score
        """
        # Use default weight function if none provided
        if weight_fn is None:
            weight_fn = lambda from_node, to_node: 1.0
        
        # Create the starting node
        from_node = RecommenderNode(NodeType.OBJECT, from_obj)
        
        # Get recommendations
        recommendations = self.recommendations_map(from_node, depth, max_total_steps, weight_fn, visualize)
        
        # Filter to only include objects and extract their values
        object_recommendations = []
        for node, score in recommendations.items():
            if node.type == NodeType.OBJECT:
                object_recommendations.append((node.value, score))
        
        # Sort by score (descending) and limit results
        return sorted(object_recommendations, key=lambda x: x[1], reverse=True)[:limit]
    
    def recommend_tags(
        self,
        from_obj,
        depth=5,
        max_total_steps=1000,
        weight_fn=None,
        limit=10,
        visualize=False
    ):
        """
        Recommends tags based on the given object
        
        Args:
            from_obj: Object to base recommendations on
            depth: Max depth for each walk
            max_total_steps: Total number of steps to take across all walks
            weight_fn: Optional custom weight function for the random walk
            limit: Maximum number of recommendations to return
            visualize: If True, visualize the recommendation process
            
        Returns:
            List of tuples containing recommended tags and their scores, sorted by score
        """
        # Use default weight function if none provided
        if weight_fn is None:
            weight_fn = lambda from_node, to_node: 1.0
        
        # Create the starting node
        from_node = RecommenderNode(NodeType.OBJECT, from_obj)
        
        # Get recommendations
        recommendations = self.recommendations_map(from_node, depth, max_total_steps, weight_fn, visualize)
        
        # Filter to only include tags and extract their values
        tag_recommendations = []
        for node, score in recommendations.items():
            if node.type == NodeType.TAG:
                tag_recommendations.append((node.value, score))
        
        # Sort by score (descending) and limit results
        return sorted(tag_recommendations, key=lambda x: x[1], reverse=True)[:limit]
    
    def recommend_objects_from_tag(
        self,
        from_tag,
        depth=5,
        max_total_steps=1000,
        weight_fn=None,
        limit=10,
        visualize=False
    ):
        """
        Recommends objects based on the given tag
        
        Args:
            from_tag: Tag to base recommendations on
            depth: Max depth for each walk
            max_total_steps: Total number of steps to take across all walks
            weight_fn: Optional custom weight function for the random walk
            limit: Maximum number of recommendations to return
            visualize: If True, visualize the recommendation process
            
        Returns:
            List of tuples containing recommended objects and their scores, sorted by score
        """
        # Use default weight function if none provided
        if weight_fn is None:
            weight_fn = lambda from_node, to_node: 1.0
        
        # Create the starting node
        from_node = RecommenderNode(NodeType.TAG, from_tag)
        
        # Get recommendations
        recommendations = self.recommendations_map(from_node, depth, max_total_steps, weight_fn, visualize)
        
        # Filter to only include objects and extract their values
        object_recommendations = []
        for node, score in recommendations.items():
            if node.type == NodeType.OBJECT:
                object_recommendations.append((node.value, score))
        
        # Sort by score (descending) and limit results
        return sorted(object_recommendations, key=lambda x: x[1], reverse=True)[:limit]

def load_ecommerce_data(filename="ecommerce_products.csv"):
    """
    Load e-commerce product data from CSV file
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        DataFrame containing product data
    """
    return pd.read_csv(filename)

def build_recommender_from_data(products_df):
    """
    Build a recommender system from product data
    
    Args:
        products_df: DataFrame containing product data
        
    Returns:
        Initialized Recommender object
    """
    recommender = Recommender()
    
    # Add all products to the recommender
    for _, product in products_df.iterrows():
        product_dict = product.to_dict()
        recommender.add_object(product_dict)
        
        # Parse meta tags and add them
        meta_tags = product["meta"].split()
        for tag in meta_tags:
            recommender.add_tag(tag)
            recommender.tag_object(product_dict, tag)
    
    return recommender

def custom_weight_function(from_node, to_node):
    """
    Custom weight function for the random walk
    
    Args:
        from_node: Starting node
        to_node: Potential next node
        
    Returns:
        Weight/probability of moving from from_node to to_node
    """
    # If going from object to tag, weight is 1.0
    if from_node.type == NodeType.OBJECT and to_node.type == NodeType.TAG:
        return 1.0
    
    # If going from tag to object, weight is inversely proportional to the object's popularity
    # This helps discover less popular but potentially relevant items
    if from_node.type == NodeType.TAG and to_node.type == NodeType.OBJECT:
        # Get the degree of the to_node (number of tags)
        deg = len(to_node.value.get("meta", "").split())
        # Higher weight for items with fewer tags (to promote diversity)
        return 1.0 / max(1, deg)
    
    # Default weight
    return 1.0

def format_product_output(product, score=None):
    """
    Format product information for display
    
    Args:
        product: Product dictionary
        score: Recommendation score (optional)
        
    Returns:
        Formatted string with product information
    """
    output = [
        f"ID: {product['product_id']}",
        f"Name: {product['name']}",
        f"Price: ${product['price']:.2f}",
        f"Category: {product['category']} / {product['product_type']}",
        f"Size: {product['size']}",
        f"Brand: {product['brand']}",
        f"Tags: {product['meta']}"
    ]
    
    if score is not None:
        output.append(f"Score: {score}")
    
    return "\n".join(output)

def create_recommendation_dashboard(recommender, products_df):
    """
    Create an interactive dashboard for the recommendation system.
    
    Args:
        recommender: The recommender system
        products_df: DataFrame containing product data
    """
    # Create a selection of products for demonstration
    sample_products = products_df.sample(min(5, len(products_df))).to_dict('records')
    
    # Create a tag summary for visualization
    tag_counts = defaultdict(int)
    for _, product in products_df.iterrows():
        for tag in product["meta"].split():
            tag_counts[tag] += 1
    
    # Visualize tag distribution
    tags = list(tag_counts.keys())
    counts = list(tag_counts.values())
    
    # Sort by popularity
    sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    top_tags = [tags[i] for i in sorted_indices[:20]]  # Top 20 tags
    top_counts = [counts[i] for i in sorted_indices[:20]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_tags,
            y=top_counts,
            marker_color='rgb(55, 83, 109)'
        )
    ])
    fig.update_layout(
        title="Top 20 Tags by Product Count",
        xaxis_title="Tag",
        yaxis_title="Product Count",
        template="plotly_white"
    )
    fig.show()
    
    # For demonstration, pick a random product to show recommendations
    random_product = random.choice(sample_products)
    
    print("\n" + "="*50)
    print("SELECTED PRODUCT FOR RECOMMENDATIONS:")
    print("="*50)
    print(format_product_output(random_product))
    
    # Visualize the full graph
    print("\n" + "="*50)
    print("FULL GRAPH VISUALIZATION:")
    print("="*50)
    recommender.visualize_full_graph(max_nodes=75)
    
    # Get and visualize product recommendations
    print("\n" + "="*50)
    print("PRODUCT RECOMMENDATIONS (with visualization):")
    print("="*50)
    
    recommendations = recommender.recommend_objects(
        random_product,
        depth=3,
        max_total_steps=500,
        weight_fn=custom_weight_function,
        limit=5,
        visualize=True
    )
    
    if recommendations:
        for i, (product, score) in enumerate(recommendations, 1):
            print(f"\n--- Recommendation #{i} ---")
            print(format_product_output(product, score))
    else:
        print("No product recommendations found.")
    
    # Visualize recommendation strength
    if recommendations:
        rec_products = [r[0]["name"] for r in recommendations]
        rec_scores = [r[1] for r in recommendations]
        
        fig = go.Figure(data=[
            go.Bar(
                x=rec_products,
                y=rec_scores,
                marker_color='rgb(26, 118, 255)'
            )
        ])
        fig.update_layout(
            title=f"Recommendation Scores for {random_product['name']}",
            xaxis_title="Recommended Product",
            yaxis_title="Score",
            template="plotly_white"
        )
        fig.show()

def main():
    """Main function to run the recommendation system with enhanced visualization"""
    print("Loading e-commerce product data...")
    try:
        products_df = load_ecommerce_data()
    except FileNotFoundError:
        print("Data file not found. Generating new dataset...")
        from dataset import generate_ecommerce_dataset
        products_df = generate_ecommerce_dataset(100)
        products_df.to_csv("ecommerce_products.csv", index=False)
    
    print(f"Loaded {len(products_df)} products")
    
    print("\nBuilding recommendation graph...")
    start_time = time.time()
    recommender = build_recommender_from_data(products_df)
    build_time = time.time() - start_time
    print(f"Graph built in {build_time:.2f} seconds")
    
    # Create an interactive dashboard with multiple visualizations
    create_recommendation_dashboard(recommender, products_df)
    
    # Select a random product for recommendations
    random_product = products_df.iloc[random.randint(0, len(products_df) - 1)].to_dict()
    
    # Get tag recommendations
    print("\n" + "="*50)
    print("TAG RECOMMENDATIONS:")
    print("="*50)
    tag_recommendations = recommender.recommend_tags(random_product, limit=10, visualize=True)
    
    if tag_recommendations:
        print("Recommended tags:")
        for tag, score in tag_recommendations:
            print(f"- {tag} (score: {score})")
        
        # Visualize tag recommendations
        tags = [t[0] for t in tag_recommendations]
        scores = [t[1] for t in tag_recommendations]
        
        fig = go.Figure(data=[
            go.Bar(
                x=tags,
                y=scores,
                marker_color='rgb(15, 157, 88)'
            )
        ])
        fig.update_layout(
            title=f"Tag Recommendations for {random_product['name']}",
            xaxis_title="Recommended Tag",
            yaxis_title="Score",
            template="plotly_white"
        )
        fig.show()
    else:
        print("No tag recommendations found.")
    
    # Demonstrate tag-based recommendations
    if random_product.get("meta"):
        sample_tag = random_product["meta"].split()[0]
        
        print("\n" + "="*50)
        print(f"PRODUCTS RELATED TO TAG '{sample_tag}':")
        print("="*50)
        tag_based_recommendations = recommender.recommend_objects_from_tag(
            sample_tag, 
            limit=5, 
            visualize=True
        )
        
    if tag_based_recommendations:
        for i, (product, score) in enumerate(tag_based_recommendations, 1):
            print(f"\n--- Related Product #{i} ---")
        print(format_product_output(product, score))
    else:
        print(f"No products found related to tag '{sample_tag}'.")

# if __name__ == "__main__":
main()