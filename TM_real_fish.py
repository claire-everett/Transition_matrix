#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:53:10 2024

@author: claireeverett
"""


import os
import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

# Directories containing the CSV files
fish_1_dir = '/Users/claireeverett/Desktop/real_fish/run_corr/both_1'
fish_2_dir = '/Users/claireeverett/Desktop/real_fish/run_corr/both_2'

# Get the list of CSV files from both directories
fish_1_files = sorted(glob(os.path.join(fish_1_dir, '*.csv')))
fish_2_files = sorted(glob(os.path.join(fish_2_dir, '*.csv')))

# Define orientation thresholds
ori_lower = 70
ori_middle = 110
ori_upper = 150

# Define function to classify states
def classify_state(orientation, combined, ori_lower, ori_middle, ori_upper):
    if orientation > ori_upper:
        if combined == 1:
            return 'flare_face'
        else:
            return 'noflare_face'
    elif ori_middle <= orientation < ori_upper:
        if combined == 1:
            return 'flare_turn'
        else:
            return 'noflare_turn'
    elif ori_middle >= orientation > ori_lower:
        if combined == 1:
            return 'flare_lateral'
        else:
            return 'noflare_lateral'
    else:
        if combined == 1:
            return 'flare_away'
        else:
            return 'noflare_away'

# Initialize a list to hold all the dataframes
all_data = []

# Process each pair of files
for file_1, file_2 in zip(fish_1_files, fish_2_files):
    # Read CSV files
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)

    # Classify states for fish 1 and fish 2
    df1['state'] = df1.apply(lambda row: classify_state(row['orientation'], row['Combined'], ori_lower, ori_middle, ori_upper), axis=1)
    df2['state'] = df2.apply(lambda row: classify_state(row['orientation'], row['Combined'], ori_lower, ori_middle, ori_upper), axis=1)

    # Create combined states
    combined_states = df1['state'] + '_' + df2['state']

    # Create a new dataframe with the combined states
    combined_df = pd.DataFrame({'combined_state': combined_states})

    # Append the combined dataframe to the list
    all_data.append(combined_df)

# Concatenate all dataframes into one
final_df = pd.concat(all_data, ignore_index=True)[:528000]

# Convert combined states to a number system
state_mapping = {state: idx for idx, state in enumerate(final_df['combined_state'].unique())}
final_df['state_id'] = final_df['combined_state'].map(state_mapping)

# Create the transition matrix
transition_matrix = np.zeros((len(state_mapping), len(state_mapping)))

# Populate the transition matrix
for i in range(1, len(final_df)):
    prev_state = final_df.iloc[i - 1]['state_id']
    curr_state = final_df.iloc[i]['state_id']
    transition_matrix[prev_state, curr_state] += 1

# Normalize the transition matrix
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

# Save the transition matrix and the final dataframe if needed
transition_matrix_df = pd.DataFrame(transition_matrix, index=state_mapping.keys(), columns=state_mapping.keys())
final_df.to_csv('final_states.csv', index=False)
transition_matrix_df.to_csv('transition_matrix.csv')

print("Transition matrix and final states dataframe have been saved.")

# Get value counts and filter for counts below 10,000
value_counts = final_df['state_id'].value_counts()
filtered_states = value_counts[value_counts > 10000].index
filtered_state_names = [state for state, idx in state_mapping.items() if idx in filtered_states]

# Filter the transition matrix
filtered_transition_matrix_df = transition_matrix_df.loc[filtered_state_names, filtered_state_names]

np.fill_diagonal(filtered_transition_matrix_df.values, np.nan)

# Save filtered transition matrix
filtered_transition_matrix_df.to_csv('/Users/claireeverett/Desktop/filtered_transition_matrix.csv')

# Plot the filtered transition matrix heatmap
plt.figure(figsize=(24, 20))
sns.heatmap(filtered_transition_matrix_df, annot=False, fmt=".2f", cmap="viridis")
plt.title("Filtered State Transition Matrix Heatmap")
plt.xlabel("To State")
plt.ylabel("From State")
plt.savefig('/Users/claireeverett/Desktop/filtered_test.pdf')
plt.show()

# Create a directed graph from the filtered transition matrix
G = nx.DiGraph()

# Add nodes and edges with weights
for from_state, to_states in filtered_transition_matrix_df.iterrows():
    for to_state, weight in to_states.items():
        if not np.isnan(weight) and weight > 0:
            G.add_edge(from_state, to_state, weight=weight)

# Calculate node sizes based on value counts
node_sizes = value_counts.loc[filtered_states].values
max_size = max(node_sizes)
min_size = min(node_sizes)
scaled_node_sizes = [(size - min_size) / (max_size - min_size) * 9000 + 1000 for size in node_sizes]  # scale between 1000 and 10000

# Draw the graph
plt.figure(figsize=(48, 40))
pos = nx.spectral_layout(G)  # positions for all nodes
edges = G.edges(data=True)

# Filter edges with weight greater than 0.0025
filtered_edges = [(u, v, d) for (u, v, d) in edges if d['weight'] > 0.0025]

# Calculate the edge widths and colors
weights = [d['weight'] for (u, v, d) in filtered_edges]
max_weight = max(weights)
min_weight = min(weights)
scaled_weights = [(weight - min_weight) / (max_weight - min_weight) * 20 + 1 for weight in weights]  # scale between 1 and 10

# Function to map weight to color intensity
def weight_to_color(weight, min_weight, max_weight):
    norm_weight = (weight - min_weight) / (max_weight - min_weight)  # Normalize weight to [0, 1]
    red_intensity = norm_weight  # Scale to [0, 1]
    return (red_intensity, 0, 0, 1)  # RGBA for red with varying intensity

edge_colors = [weight_to_color(d['weight'], min_weight, max_weight) for (u, v, d) in filtered_edges]

# Draw nodes with sizes based on value counts
nx.draw_networkx_nodes(G, pos, node_size=scaled_node_sizes, node_color='skyblue')

# Draw edges with arrows, scaled weights, and varying colors
for (u, v, d), width, color in zip(filtered_edges, scaled_weights, edge_colors):
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)], width=width,
        arrowstyle='-|>', arrowsize=20, edge_color=[color], connectionstyle='arc3,rad=0.1'
    )

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

plt.title("Filtered State Transition Network Graph")
plt.savefig('/Users/claireeverett/Desktop/filtered_network_spectral.pdf')
plt.show()

# Find the most probable sequence of behaviors based on cumulative weight of transitions
def find_most_probable_path(transition_matrix, state_mapping, start_state, steps=10):
    inverse_state_mapping = {v: k for k, v in state_mapping.items()}
    current_state = start_state
    path = [inverse_state_mapping[current_state]]
    for _ in range(steps):
        next_state = np.nanargmax(transition_matrix[current_state])
        path.append(inverse_state_mapping[next_state])
        current_state = next_state
    return path

# Example usage
start_state = final_df.iloc[0]['state_id']  # Starting from the first state in the dataframe
most_probable_path = find_most_probable_path(transition_matrix_df.values, state_mapping, start_state, steps=10)
print("Most probable path:", most_probable_path)

# Create a directed graph from the filtered transition matrix (spring layout)
plt.figure(figsize=(24, 20))
pos = nx.spring_layout(G, seed=42)  # positions for all nodes
edges = G.edges(data=True)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=7000)

# Draw edges with arrows and weights
nx.draw_networkx_edges(
    G, pos, edgelist=edges, width=[d['weight'] * 10 for (u, v, d) in edges],
    arrowstyle='-|>', arrowsize=20, edge_color='blue', connectionstyle='arc3,rad=0.1'
)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

plt.title("Filtered State Transition Network Graph")
plt.savefig('/Users/claireeverett/Desktop/filtered_network_graph.pdf')
plt.show()
