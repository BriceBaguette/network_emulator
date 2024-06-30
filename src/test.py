import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objs as go

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function to create a network graph figure
def create_network_graph():
    # Create a simple networkx graph
    G = nx.karate_club_graph()

    # Get the positions of the nodes using a layout algorithm
    pos = nx.spring_layout(G)

    # Create the edge traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
        )

    # Create the node trace
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        text=[str(node) for node in G.nodes()],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    # Color nodes by their degree
    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    node_trace.marker.color = node_adjacencies

    # Create the figure
    fig = go.Figure(
        data=edge_trace + [node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            annotations=[dict(
                text="Network graph made with Python",
                showarrow=False,
                xref="paper", yref="paper"
            )],
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
    )
    return fig

# Define the layout of the app
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Network Graph Visualization", className="text-center my-4")
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id='network-graph',
                    figure=create_network_graph()
                )
            )
        )
    ],
    fluid=True
)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
