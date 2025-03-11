from enum import Enum
import pandas as pd
import random
from collections import defaultdict
from pyvis.network import Network

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
        visited =[]
        
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
        self.graph.add_node(RecommenderNode(NodeType.OBJECT, obj))
    
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
        weight_fn
    ):
        """
        Creates a map of recommendations with scores based on random walks
        
        Args:
            from_node: Starting node
            depth: Max depth for each walk
            max_total_steps: Total number of steps to take across all walks
            weight_fn: Function to determine edge weights during walk
            
        Returns:
            Dict of recommended nodes with their scores
        """
        scores = defaultdict(int)
        steps_remaining = max_total_steps
        
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
        
        # Remove the starting node from results
        if from_node in scores:
            del scores[from_node]
            
        return dict(scores)
    
    def recommend_objects(
        self,
        from_obj,
        depth=5,
        max_total_steps=1000,
        weight_fn=None,
        limit=10
    ):
        """
        Recommends objects based on the given object
        
        Args:
            from_obj: Object to base recommendations on
            depth: Max depth for each walk
            max_total_steps: Total number of steps to take across all walks
            weight_fn: Optional custom weight function for the random walk
            limit: Maximum number of recommendations to return
            
        Returns:
            List of tuples containing recommended objects and their scores, sorted by score
        """
        # Use default weight function if none provided
        if weight_fn is None:
            weight_fn = lambda from_node, to_node: 1.0
        
        # Create the starting node
        from_node = RecommenderNode(NodeType.OBJECT, from_obj)
        
        # Get recommendations
        recommendations = self.recommendations_map(from_node, depth, max_total_steps, weight_fn)
        
        # Filter to only include objects and extract their values
        object_recommendations = []       
        for node, score in recommendations.items():
            if node.type == NodeType.OBJECT:
                object_recommendations.append((node.value, score))
        
        # Sort by score (descending) and limit results
        return sorted(object_recommendations, key=lambda x: x[1], reverse=True)[:limit]

def load_and_split_data(filename="ecommerce_products.csv", split=50):
    """Loads the dataset, splits it, and prepares it for recommendation."""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print("Data file not found. Please provide a valid dataset.")
        return None, None

    # Split data
    df_input = df.head(split).copy()
    df_recommend = df.iloc[split:].copy()

    return df_input, df_recommend

def build_recommender_graph(df_input):
    """Builds the recommender graph from the input DataFrame."""
    recommender = Recommender()
    for _, product in df_input.iterrows():
        product_dict = product.to_dict()
        recommender.add_object(product_dict)
        meta_tags = product["meta"].split()
        for tag in meta_tags:
            recommender.add_tag(tag)
            recommender.tag_object(product_dict, tag)
    return recommender

def generate_recommendations(recommender, df_recommend):
    """Generates recommendations for the remaining dataset."""
    recommendations = []
    for _, product in df_recommend.iterrows():
        product_dict = product.to_dict()
        recs = recommender.recommend_objects(product_dict, limit=5)
        recommendations.append((product_dict, recs))
    return recommendations

def visualize_recommendations(recommendations):
    """Visualizes the recommendation graph using pyvis."""
    net = Network(notebook=True, directed=False,  cdn_resources='in_line')
    added_nodes = set()

    for product, recs in recommendations:
        product_node = RecommenderNode(NodeType.OBJECT, product)
        if product_node not in added_nodes:
            net.add_node(product_node.__hash__(), label=f"Product: {product.get('name', 'Unknown')}", color='blue')
            added_nodes.add(product_node)

        for rec_product, score in recs:
            rec_product_node = RecommenderNode(NodeType.OBJECT, rec_product)
            if rec_product_node not in added_nodes:
                net.add_node(rec_product_node.__hash__(), label=f"Product: {rec_product.get('name', 'Unknown')}", color='green')
                added_nodes.add(rec_product_node)

            net.add_edge(product_node.__hash__(), rec_product_node.__hash__(), title=f"Score: {score}")

    net.show("recommendation_graph.html")

def main():
    df_input, df_recommend = load_and_split_data()
    if df_input is None or df_recommend is None:
        return

    recommender = build_recommender_graph(df_input)
    recommendations = generate_recommendations(recommender, df_recommend)
    visualize_recommendations(recommendations)

if __name__ == "__main__":
    main()