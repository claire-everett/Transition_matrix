# Transition_matrix
transition matrices for real_fish encounterst


Setup and Imports:
- Import necessary libraries.
- Define directories containing the CSV files for both fish.
  
Classification Function:
- Define a function classify_state to categorize the states based on orientation and combined status.
  
Data Processing:
- Read and process each pair of CSV files to classify the states and create combined state dataframes.
Concatenate all dataframes into one final dataframe.

Transition Matrix:
- Create and populate the transition matrix based on the state transitions.
- Normalize the transition matrix.
  
Save Data:
- Save the final dataframe and the transition matrix to CSV files.

Visualization and Graph Creation:
-Filter the transition matrix based on state counts.
- Plot the filtered transition matrix as a heatmap.
- Create and draw a directed graph based on the filtered transition matrix.
- Calculate node sizes based on state counts and adjust edge widths and colors for visualization.
  
Most Probable Path (still working on):
- Define and use a function to find the most probable sequence of state transitions based on the cumulative weights.
