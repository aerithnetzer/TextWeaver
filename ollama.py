import plotly.graph_objs as go

# Define the nodes and links of the graph
nodes = ['A', 'B', 'C', 'D']
links = [('A', 'B'), ('A', 'C'), ('B', 'D')]

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node_size=20,  # size of the nodes
    link_color='blue',  # color of the links
    link_width=10,  # width of the links
    node_label=nodes,  # labels for the nodes
    link_label=links  # labels for the links
)])

# Show the graph
fig.show()