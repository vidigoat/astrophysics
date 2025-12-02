"""Utility functions for bootstrap validation plots."""
import re
import os

def parse_fcit_results(txt_path):
    """
    Parse FCIT results file to extract edges.
    Returns a list of edge strings in original format.
    """
    if not os.path.exists(txt_path):
        return []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    edges = []
    # Find the Graph Edges section
    if "Graph Edges:" in content:
        edges_section = content.split("Graph Edges:")[1]
        # Extract numbered edge lines
        for line in edges_section.split('\n'):
            line = line.strip()
            if not line or line.startswith('='):
                continue
            # Remove leading number and period (e.g., "1. " or "10. ")
            line = re.sub(r'^\d+\.\s*', '', line)
            if any(arrow in line for arrow in ['-->', '<--', 'o-o', 'o->']):
                # Normalize edge format (handle bidirectional)
                edge = normalize_edge_from_fcit(line)
                if edge:
                    edges.append(edge)
    
    return edges

def normalize_edge_from_fcit(edge_str):
    """
    Normalize edge from FCIT results to match bootstrap CSV format.
    For o-o edges, sorts nodes alphabetically (like bootstrap does).
    For --> and o-> edges, keeps original order.
    """
    edge_str = edge_str.strip()
    # Replace <-- with --> (normalize direction)
    if '<--' in edge_str:
        parts = edge_str.split('<--')
        edge_str = f"{parts[1].strip()} --> {parts[0].strip()}"
    
    # Normalize o-o edges by sorting nodes alphabetically (like bootstrap does)
    if ' o-o ' in edge_str:
        parts = edge_str.split(' o-o ')
        nodes = sorted([parts[0].strip(), parts[1].strip()])
        edge_str = f"{nodes[0]} o-o {nodes[1]}"
    elif 'o-o' in edge_str:
        parts = edge_str.split('o-o')
        nodes = sorted([parts[0].strip(), parts[1].strip()])
        edge_str = f"{nodes[0]} o-o {nodes[1]}"
    
    # Ensure consistent spacing
    edge_str = edge_str.replace(' --> ', ' --> ').replace('-->', ' --> ')
    edge_str = edge_str.replace(' o-> ', ' o-> ').replace('o->', ' o-> ')
    # Clean up multiple spaces
    edge_str = ' '.join(edge_str.split())
    return edge_str

def get_edge_display_label(edge):
    """
    Get display label for an edge, preserving the actual edge types:
    --> (directed), o-o (bidirectional), o-> (partially directed)
    """
    # Keep the original edge format for display
    return edge

