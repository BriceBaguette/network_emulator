"""
This script defines a Dash application for network simulation and analysis.
It includes functions for creating network graphs, defining layout for different pages,
and handling user inputs for simulation parameters.
"""
import ast
import base64
import tempfile
import os
import random
import datetime
import zipfile
import io
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
from dash.long_callback import DiskcacheLongCallbackManager
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px
import diskcache
import pandas as pd
from network_emulator.network_emulator import NetworkEmulator

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config['suppress_callback_exceptions'] = True
net_sim: NetworkEmulator = None

# Sample list of IDs for dropdowns
sample_ids: list[str] = []

# Define the initial layout of the app
app.layout = dbc.Container(
    [
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='stored-file'),
        dcc.Store(id='button-click'),
        dcc.Interval(id='interval-component', interval=1 *
                     1000, n_intervals=0, disabled=True),
        dcc.Interval(id='emulate-interval', interval=100,
                     n_intervals=0, disabled=True),
        html.Div(id='page-content'),
    ],
    fluid=True,
)

# Define the layout for the upload page
upload_layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("File Upload", className="text-center my-4"),
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select a File')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': 'auto'  # Center the upload component
                    },
                    multiple=False
                ),
                style={
                    'display': 'flex',
                    'justifyContent': 'center',
                    'width': '100%',  # Ensure the row takes up the full viewport height
                }
            ),
        ),
        dbc.Row(
            dbc.Col(
                html.Div(id='output-data-upload', className="mt-4")
            )
        )
    ],
    fluid=True,
)

# Define a layout for the loading screen
loading_layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Building Network...", className="text-center my-4"),
            )
        ),
        dbc.Row(
            dbc.Col(
                dbc.Spinner(size="lg", color="primary", children=[
                    html.Div(id='loading-output')
                ], fullscreen=True),
                style={
                    'display': 'flex',
                    'justifyContent': 'center',
                    'width': '100%',  # Ensure the row takes up the full viewport height
                }
            ),
        )
    ],
    fluid=True,
)

menu_layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button("Emulation", color="primary",
                               size="lg", id="start-simulation-btn", className="w-100"),

                    className="d-flex justify-content-center"  # Center the button within the column
                ),
                dbc.Col(
                    dbc.Button("Topology Analysis", color="primary",
                               size="lg", id="ecmp-analysis-btn", className="w-100"),

                    className="d-flex justify-content-center"  # Center the button within the column
                ),
                dbc.Col(
                    dbc.Button("Measurements Analysis", color="primary",
                               size="lg", id="analysis-btn", className="w-100"),
                    className="d-flex justify-content-center"  # Center the button within the column
                )
            ],
            justify="around",  # Space around the columns
            className="my-4",  # Optional: add vertical margin for better spacing
        ),
        dbc.Row(
            dbc.Col(
                html.Div(id='button-output', className="mt-4")
            )
        )
    ],
    fluid=True,
    style={"width": "80%"}  # Set the container width to 80% of the viewport
)


def create_network_graph():
    """
    Creates a network graph using NetworkX and Plotly.

    This function generates a network graph from a topology obtained via `net_sim.get_topology()`.
    It uses a spring layout algorithm to position the nodes with increased spacing.
    The graph is visualized using Plotly with nodes colored by their degree.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object representing the network graph.
    """
    # Create a simple networkx graph
    g = net_sim.get_topology()

    # Get the positions of the nodes using a layout algorithm with increased spacing
    # Adjust the value of k to control node spacing
    pos = nx.spring_layout(g, k=1, dim=2)

    # Create the edge traces
    edge_trace = []
    for edge in g.edges():
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
        x=[pos[node][0] for node in g.nodes()],
        y=[pos[node][1] for node in g.nodes()],
        text=[str(node) for node in g.nodes()],
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
    for node, adjacencies in enumerate(g.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    node_trace.marker.color = node_adjacencies

    # Create the figure
    fig = go.Figure(
        data=edge_trace + [node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
    )
    return fig


# Define the content layouts for each button
def start_simulation_layout() -> dbc.Container:
    """
    Creates the layout for the simulation start page using Dash Bootstrap Components.

    This function sets up the layout for the simulation start page, including input fields for 
    probe rate, number of generations, telemetry period, and hardware failure selection. 
    It also includes a button to start the emulation and displays progress and download links.

    Returns:
        dbc.Container: A Dash Bootstrap Container with the layout for the simulation start page.
    """
    return dbc.Container(
        [
            dcc.Store(id='working-state', data={'working': False}),
            dbc.Row(
                dbc.Col(
                    html.H1("Emulation", className="text-center my-4")
                )
            ),
            # dbc.Row(dbc.Col(dcc.Graph(id='network-graph',
            #      figure=create_network_graph()))),
            dbc.Row(
                dbc.Col(
                    dbc.Form(
                        [
                            dbc.CardGroup(
                                [
                                    dbc.Label("Probe Rate",
                                              html_for="probe-rate"),
                                    dbc.Input(type="number", id="probe-rate",
                                              value=net_sim.generation_rate),
                                ]
                            ),
                            dbc.CardGroup(
                                [
                                    dbc.Label("Number of generation",
                                              html_for="number-of-generation"),
                                    dbc.Input(
                                        type="number",
                                        id="number-of-generation",
                                        value=net_sim.num_generation),
                                ]
                            ),
                            dbc.CardGroup(
                                [
                                    dbc.Label("Telemtry Period in minute",
                                              html_for="telemetry-period"),
                                    dbc.Input(
                                        type='number',
                                        id="telemetry-period",
                                        value=net_sim.duration/60),
                                ]
                            ),
                        ],
                        className="my-4",
                    )
                )
            ),
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='hw-failure-dropdown',
                        options=sample_ids,
                        placeholder="Select a router that encounters hardware failure"
                    ),
                ),
                dbc.Form([
                    dbc.CardGroup(
                        [
                            dbc.Label("Start Probe",
                                      html_for="start-probe"),
                            dbc.Input(type="number", id="start_probe",
                                      value=0),
                        ]
                    ),
                    dbc.CardGroup(
                        [
                            dbc.Label("End Probe",
                                      html_for="end-probe"),
                            dbc.Input(
                                type="number", id="end_probe", value=0),
                        ]
                    ),]),
            ],

            ),
            dbc.Row(dbc.Col(dbc.Button("Emulate Now", color="primary", size="lg", id="emulate-btn"),
                            width="auto"
                            ),
                    justify="center",
                    className="mb-4"),
            html.Div(id='progress-output', className="mt-4"),
            dcc.Download(id='download-link'),
        ]
    )


def ecmp_analysis_layout(dropdown_options) -> dbc.Container:
    """
    Creates the layout for the ECMP analysis page using Dash Bootstrap Components.

    This function sets up the layout for the ECMP analysis page, including dropdowns for 
    selecting IDs, sections for ECMP and latency imbalance analysis, and a submit button 
    for additional analysis. The layout is designed to be fluid and responsive.

    Args:
        dropdown_options (list): A list of options for the dropdown menus.

    Returns:
        dbc.Container: A Dash Bootstrap Container with the layout for the ECMP analysis page.
    """
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    html.H1("Topology Analysis Page",
                            className="text-center my-4")
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.H2("ECMP Analysis", className="text-center my-4")
                )
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Dropdown(
                        id='ecmp-id-dropdown',
                        options=dropdown_options,
                        placeholder="Select an ID"
                    ),
                    width=6,
                    className="mx-auto"
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(id='ecmp-graphs', className="mt-4")
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.H2("Latency Imbalance Analysis",
                            className="text-center my-4")
                )
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Dropdown(
                        id='additional-id-dropdown-1',
                        options=dropdown_options,
                        placeholder="Select an ID"
                    ),
                    width=6,
                    className="mx-auto"
                )
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Dropdown(
                        id='additional-id-dropdown-2',
                        options=dropdown_options,
                        placeholder="Select an ID"
                    ),
                    width=6,
                    className="mx-auto mt-3"
                )
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Button("Submit", id='submit-additional-analysis',
                               color="primary", className="mt-3"),
                    width=6,
                    className="d-flex justify-content-center mx-auto"
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(id='additional-analysis-output', className="mt-4")
                )
            )
        ],
        fluid=True
    )


def measurements_analysis_layout() -> dbc.Container:
    """
    Creates the layout for the measurements analysis page using Dash Bootstrap Components.

    This function sets up the layout for the measurements analysis page, including upload 
    components for source and sink CSV files, and a combined output section. The layout 
    is designed to be fluid and responsive.

    Returns:
        dbc.Container: A Dash Bootstrap Container with the layout for the measurements
        analysis page.
    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Upload(
                                id='upload-source',
                                children=html.Div(
                                    ['Drag and Drop or ', html.A('Select a Source CSV File')]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ),
                            html.Div(id='source-upload-output')
                        ],
                        width=7
                    ),
                    dbc.Col(
                        [
                            dcc.Upload(
                                id='upload-sink',
                                children=html.Div(
                                    ['Drag and Drop or ', html.A('Select a Sink CSV File')]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ),
                            html.Div(id='sink-upload-output')
                        ],
                        width=7
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(id='combined-output',
                             style={'marginTop': '20px'}),
                    width=12
                )
            )
        ],
        fluid=True
    )


@app.callback([Input('probe-rate', 'value'),
               Input('number-of-generation', 'value'),
               Input('telemetry-period', 'value')])
def update_emulation_parameters(probe_rate, num_gen, tel_period):
    """
    Updates the emulation parameters based on the input values from the UI.

    This callback function updates the global `net_sim` object with the new values for 
    probe rate, number of generations, and telemetry period. The telemetry period is 
    converted from minutes to seconds before updating.

    Args:
        probe_rate (int): The new probe rate value.
        num_gen (int): The new number of generations value.
        tel_period (int): The new telemetry period value in minutes.

    Returns:
        dash.no_update: Indicates that no component updates are needed.
    """
    if probe_rate is not None:
        net_sim.generation_rate = probe_rate
    if num_gen is not None:
        net_sim.num_generation = num_gen
    if tel_period is not None:
        net_sim.duration = tel_period * 60
    return dash.no_update
# Define the callback to store the uploaded file contents and navigate to the loading screen


@app.callback(
    [Output('stored-file', 'data'),
     Output('url', 'pathname')],
    [Input('upload-data', 'contents')],
    prevent_initial_call=True
)
def store_file_and_redirect(contents):
    """
    Stores the uploaded file contents and redirects to a loading page.

    This callback function takes the contents of an uploaded file and stores it in a 
    Dash `dcc.Store` component. It then redirects the user to a loading page. If no 
    contents are provided, it redirects to the home page.

    Args:
        contents (str): The contents of the uploaded file.

    Returns:
        tuple: A tuple containing the file contents and the new URL pathname.
    """
    if contents is not None:
        return contents, '/loading'
    return None, '/'

# Define the callback to update the content based on button click


@app.callback(
    Output('button-output', 'children'),
    [Input('start-simulation-btn', 'n_clicks'),
     Input('ecmp-analysis-btn', 'n_clicks'),
     Input('analysis-btn', 'n_clicks')],
    prevent_initial_call=True
)
def update_output(_, __, ___):
    """
    Updates the output based on which button was clicked.

    This callback function determines which button was clicked and updates the 
    'button-output' component accordingly. It handles three buttons: start simulation, 
    ECMP analysis, and general analysis. Depending on the button clicked, it returns 
    the corresponding layout. If no button is clicked or if the net_sim object is None, 
    it returns no update or redirects to the home page.

    Args:
        _ (int): Number of clicks for the start simulation button.
        __ (int): Number of clicks for the ECMP analysis button.
        ___ (int): Number of clicks for the analysis button.

    Returns:
        dash.development.base_component.Component: The updated layout or no update.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if net_sim is None:
        return dcc.Location(id='url', refresh=True, pathname='/')
    if button_id == 'ecmp-analysis-btn':
        dropdown_options = []
        if net_sim.is_running():
            sample_ids = net_sim.get_routers_ids()  # Assuming this retrieves router IDs
            dropdown_options = [{'label': id, 'value': id}
                                for id in sample_ids]
        return ecmp_analysis_layout(dropdown_options)
    elif button_id == 'start-simulation-btn':
        return start_simulation_layout()
    elif button_id == 'analysis-btn':
        return measurements_analysis_layout()
    return dash.no_update

# Define the callback to update the ECMP Analysis content based on dropdown selection
@app.callback(
    Output('ecmp-graphs', 'children'),
    Input('ecmp-id-dropdown', 'value'),
    prevent_initial_call=True
)
def update_ecmp_graphs(selected_id):
    """
    Updates the ECMP graphs based on the selected router ID.

    This function generates two graphs: one showing the number of destinations by 
    the number of paths for the selected router ID, and another showing the latency 
    with the number of hops with the selected router ID as the source.

    Args:
        selected_id (str): The selected router ID.

    Returns:
        html.Div: A Div containing the two generated graphs, or no update if no ID is selected.
    """
    if selected_id:
        ecmp_values, path = net_sim.ecmp_analysis(selected_id)
        
        # Prepare data for the linear regression
        x = np.array(list(path.keys()))
        y = np.array(list(path.values()))
        
        # Calculate linear regression
        slope, intercept = np.polyfit(x, y, 1)
        regression_line = slope * x + intercept
        
        # Create the graphs
        graph1 = dcc.Graph(
            figure={
                'data': [{
                    'x': list(ecmp_values.keys()),
                    'y': list(ecmp_values.values()),
                    'type': 'bar',
                    'name': selected_id
                }],
                'layout': {
                    'title': f'Number of destinations by number of paths for {selected_id}',
                    'xaxis': {'title': 'Number of paths'},
                    'yaxis': {'title': 'Number of destinations'}
                }
            }
        )
        
        graph2 = dcc.Graph(
            figure={
                'data': [
                    {
                        'x': list(path.keys()),
                        'y': list(path.values()),
                        'type': 'scatter',
                        'mode': 'lines+markers',
                        'name': selected_id
                    },
                    {
                        'x': list(path.keys()),
                        'y': regression_line,
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Linear Regression',
                        'line': {'color': 'red'}
                    }
                ],
                'layout': {
                    'title': f'Latency with number of hops with {selected_id} as source',
                    'xaxis': {'title': 'Number of hops'},
                    'yaxis': {'title': 'Latency in ms'}
                }
            }
        )
        
        return html.Div([graph1, graph2])
    
    return dash.no_update


@app.callback(
    Output('additional-analysis-output', 'children'),
    [Input('submit-additional-analysis', 'n_clicks')],
    [State('additional-id-dropdown-1', 'value'),
     State('additional-id-dropdown-2', 'value')],
    prevent_initial_call=True
)
def update_additional_analysis(n_clicks, id1, id2):
    """
    Updates the additional analysis output based on the selected IDs and button click.

    This function performs a latency test between the selected IDs and generates a graph
    and a list of paths with latency imbalances. If only one ID is selected, it tests
    against all other IDs. If no IDs are selected, it performs a test on all paths.

    Args:
        n_clicks (int): Number of clicks on the submit button.
        id1 (str): The first selected ID.
        id2 (str): The second selected ID.

    Returns:
        html.Div: A Div containing the generated graph and list of paths with latency imbalances.
    """
    wrong_paths = []
    graph_value = {}
    if n_clicks and id1 and id2:
        paths, wrong_paths = net_sim.latency_test(id1, id2)
        for path in paths:
            key = path[1]/1000
            if key not in graph_value or graph_value[key] is None:
                graph_value[key] = 1
            else:
                graph_value[key] += 1
    elif n_clicks and id1 and not id2:
        for s_id in sample_ids:
            if s_id != id1:
                _, wrong_path = net_sim.latency_test(id1, s_id)
                wrong_paths.extend(wrong_path)
    else:
        wrong_paths = net_sim.all_latency_test()
    if len(wrong_paths) > 0:
        text_items = [html.Li(f"Path from {wrong_paths[0][0]} to {path[0][-1]} going through "
                              f"{path[0][1:-1]} has a latency imbalance of {path[1]/1000} ms")
                                for path in wrong_paths]
    else:
        text_items = html.Li("No wrong paths found")

    return html.Div([
        dcc.Graph(figure={'data': [{'x': list(graph_value.keys()), 'y': list(graph_value.values()),
                                    'type': 'bar',
                                    'name': f"Latency for number of paths between {id1} and {id2}"}
                                   ],
                          'layout': {
                              'title': f"Latency for number of paths between {id1} and {id2}",
                                     'xaxis': {'title': 'Latency in ms'},
                                     'yaxis': {'title': 'Number of paths'}}})  
        if len(graph_value.items()) > 0
        else html.Div(),
        html.Ul(text_items)])

# Define the callback to build the network


@app.callback(
    Output('interval-component', 'disabled'),
    Input('url', 'pathname'),
    State('stored-file', 'data'),
    prevent_initial_call=True
)
def build_network(pathname, stored_file):
    """
    Builds the network when the pathname is '/loading' and a stored file is provided.

    This function decodes the base64 encoded stored file, writes it to a temporary file,
    and initializes the NetworkEmulator with the temporary file. It then builds and starts
    the network simulation, retrieves the router IDs, and enables the interval component.

    Args:
        pathname (str): The current pathname.
        stored_file (str): The base64 encoded stored file data.

    Returns:
        bool: False to enable the interval component, True otherwise.
    """
    if pathname == '/loading' and stored_file is not None:
        global net_sim
        # Decode the base64 string
        _, content_string = stored_file.split(',')
        decoded = base64.b64decode(content_string)
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(decoded)
            temp_file_path = temp_file.name

        net_sim = NetworkEmulator(node_file=None, link_file=None, single_file=temp_file_path,
                                  generation_rate=20, num_generation=1,
                                  load_folder=None, save_folder=None)
        net_sim.build()
        net_sim.start()
        global sample_ids
        sample_ids = net_sim.get_routers_ids()
        # Clean up the temporary file after use
        os.remove(temp_file_path)

        return False  # Enable the interval component
    return True

# Define the interval callback to handle redirection to the final page


@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=True
)
def redirect_to_final(_):
    """
    Redirects to the final URL based on the network simulation status.

    This function checks if the network simulation is running and redirects to the
    '/menu' page if it is. Otherwise, it redirects to the '/loading' page.

    Args:
        _ (int): The number of intervals passed (not used).

    Returns:
        str: The pathname to redirect to.
    """
    if net_sim and net_sim.is_running():
        return '/menu'
    return '/loading'


@app.callback(
    [
        Output('progress-output', 'children'),
        Output('emulate-btn', 'disabled'),
        Output('emulate-interval', 'disabled'),
    ],
    [
        Input('emulate-btn', 'n_clicks'),
        Input('emulate-interval', 'n_intervals')
    ],
    [State('hw-failure-dropdown', 'value'),
     State('start_probe', 'value'),
     State('end_probe', 'value')],
    prevent_initial_call=True
)
def start_emulation(n_clicks, _, router_id, start, end):
    """
    Starts the network emulation process.

    This function handles the emulation process, including adding hardware issues,
    sending probabilities, exporting data, and updating the progress bar.

    Args:
        n_clicks (int): Number of clicks on the start button.
        _ (any): Placeholder for unused input.
        router_id (str): The ID of the router to add a hardware issue to.
        start (int): The start time of the hardware issue.
        end (int): The end time of the hardware issue.

    Returns:
        tuple: A tuple containing the progress display, a boolean to disable the button, and a
        boolean to show the download button.
    """

    if n_clicks is None or n_clicks == 0:
        return dash.no_update, False, True,

    if net_sim.working:
        return dash.no_update, True, False,

    if net_sim.current_step == 0 and router_id is not None:
        net_sim.add_hw_issue(start=start, end=end, source_id=router_id)

    total_steps = net_sim.num_generation * \
        net_sim.duration * net_sim.generation_rate

    net_sim.working = True


    if net_sim.current_step <= total_steps:
        routers = net_sim.routers
        start_time = []
        end_time = []
        for _, row in net_sim.hw_issue.iterrows():
            start_time.append(row['Start'])
            end_time.append(row['End'])

        start_indices = [index for index, value in enumerate(
            start_time) if value == net_sim.current_step]
        end_indices = [index for index, value in enumerate(
            end_time) if value == net_sim.current_step]
        for s in start_indices:
            net_sim.hw_update_interface(net_sim.hw_issue.iloc[s])
        for e in end_indices:
            net_sim.hw_revert_interface(net_sim.hw_issue.iloc[e])
        for source in routers:
            for destination in routers:
                if source != destination:
                    fl = random.randint(0, 256)
                    net_sim.send_prob(
                        source=source, destination=destination, flow_label=fl)

        if (net_sim.current_step % (net_sim.generation_rate*net_sim.duration) == 0
            and net_sim.current_step != 0):
            timestamp = datetime.datetime.now()
            print(f"Exporting data at {timestamp}")
            for router in routers:
                net_sim.export_source_data(source_router=router, fl=None,
                                           gen_number=net_sim.current_step //
                                           (net_sim.generation_rate*net_sim.duration),
                                           timestamp=timestamp)
                net_sim.export_sink_data(sink_router=router, fl=None,
                                         gen_number=net_sim.current_step //
                                         (net_sim.generation_rate*net_sim.duration),
                                         timestamp=timestamp)
                router.update_bins()

        net_sim.current_step += 1
        net_sim.working = False
        # Calculate progress
        progress_percent = (net_sim.current_step / total_steps) * 100
        progress_bar = dbc.Progress(
            value=progress_percent,
            max=100,
            striped=True,
            animated=True,
            color="primary"
        )
        progress_display = dbc.Container([
            dbc.Row(dbc.Col(f"{net_sim.current_step}/{total_steps} steps completed",
                    className="text-center"), className="mt-4"),
            dbc.Row(dbc.Col(progress_bar), className="mt-4"),
        ], className="my-4")

        return progress_display, True, False,

    net_sim.current_step = 0
    net_sim.working = False
    net_sim.session_id += 1
    # Simulation completed, enable the button and provide download option
    download_button = dbc.Row(
        dbc.Col(
            dbc.Button("Download Output", color="primary",
                       size="lg", id="download-btn"),
            width="auto"
        ),
        justify="center",
        className="mb-4"
    )

    return download_button, False, True,


@app.callback(
    Output('download-link', 'data'),
    Input('download-btn', 'n_clicks'),
    prevent_initial_call=True
)
def download_output(n_click):
    """
    Handles the download of output files when the download button is clicked.

    This function creates a ZIP file containing the sink and source CSV files and
    returns it for download.

    Args:
        n_click (int): Number of clicks on the download button.

    Returns:
        dcc.send_file: A Dash component to send the ZIP file for download.
    """

    if n_click is None or n_click == 0:
        return dash.no_update
        # Example data extraction and function call (replace with actual logic)
    os.makedirs('./src/results/', exist_ok=True)
    sink_file_path = './src/results/sink.csv'
    source_file_path = './src/results/source.csv'
    zip_file_path = './src/results/output_files.zip'

    # Create a ZIP file containing both CSV files
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(sink_file_path, os.path.basename(sink_file_path))
        zipf.write(source_file_path, os.path.basename(source_file_path))

    print(f"Downloaded output files to {zip_file_path}")
    return dcc.send_file(zip_file_path, 'output_files.zip', type='zip')


@app.callback(
    [Output('source-upload-output', 'children'),
     Output('sink-upload-output', 'children'),
     Output('combined-output', 'children')],
    [Input('upload-source', 'filename'),
     Input('upload-sink', 'filename')],
    [State('upload-source', 'contents'),
     State('upload-sink', 'contents')]
)
def upload_measurement_files(source_filename, sink_filename, source_contents, sink_contents):
    """
    Handles the upload of measurement files and performs analysis.

    This function processes the uploaded source and sink CSV files, decodes their contents,
    and performs hardware issue detection analysis. It updates the UI with the
    results of the analysis.

    Args:
        source_filename (str): The filename of the uploaded source file.
        sink_filename (str): The filename of the uploaded sink file.
        source_contents (str): The contents of the uploaded source file in base64 format.
        sink_contents (str): The contents of the uploaded sink file in base64 format.

    Returns:
        tuple: A tuple containing:
            - A Div element with the source file upload message.
            - A Div element with the sink file upload message.
            - A Div element with the analysis layout.
    """
    source_message = ''
    sink_message = ''
    analysis_layout = dbc.Container()

    if source_filename is not None:
        if 'source' in source_filename.lower():
            source_message = f'File "{source_filename}" uploaded successfully.'
            if source_contents:
                _, content_string = source_contents.split(',')
                decoded = base64.b64decode(content_string)
                source_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            source_message = 'Please upload a CSV file containing "source" in its name.'

    if sink_filename is not None:
        if 'sink' in sink_filename.lower():
            sink_message = f'File "{sink_filename}" uploaded successfully.'
            if sink_contents:
                _, content_string = sink_contents.split(',')
                decoded = base64.b64decode(content_string)
                sink_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            sink_message = 'Please upload a CSV file containing "sink" in its name.'

    if (source_message and sink_message and 'uploaded successfully' in source_message
        and 'uploaded successfully' in sink_message):

        list_of_routers = net_sim.hw_issue_detection(
            sink_dataframe=sink_df, source_dataframe=source_df, latency=True)
        routers_display = html.Ul([html.Li(f"Router with id: {router} encountered hardware failure")
                                  for router in list_of_routers])
        analysis_layout = dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        routers_display,
                        className="my-4"
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        html.H1("Measurements Analysis",
                                className="text-center my-4")
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.Dropdown(
                            id='source-id-dropdown',
                            options=sample_ids,
                            placeholder="Select the ID of source router"
                        ),
                        width=6,
                        className="mx-auto"
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.Dropdown(
                            id='sink-id-dropdown',
                            options=sample_ids,
                            placeholder="Select an ID of sink router"
                        ),
                        width=6,
                        className="mx-auto mt-3"
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        dbc.Button("Submit", id='submit-measurement-analysis',
                                   color="primary", className="mt-3"),
                        width=6,
                        className="d-flex justify-content-center mx-auto"
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        html.Div(id='measurement-analysis-output',
                                 className="mt-4")
                    )
                ),

            ],
            fluid=True
        )

    return html.Div(source_message), html.Div(sink_message), html.Div(analysis_layout)


def save_base64_to_csv(base64_str, file_path):
    """
    Decodes a base64-encoded string and saves it as a CSV file.

    Args:
        base64_str (str): The base64-encoded string representing the contents of the CSV file.
        file_path (str): The path where the decoded CSV file will be saved.

    Returns:
        None
    """
    _, content_string = base64_str.split(',')
    decoded = base64.b64decode(content_string)
    with open(file_path, 'wb') as f:
        f.write(decoded)


@app.callback(
    Output('measurement-analysis-output', 'children'),
    [Input('submit-measurement-analysis', 'n_clicks')],
    [State('source-id-dropdown', 'value'),
     State('sink-id-dropdown', 'value'),
     State('upload-source', 'contents'),
     State('upload-sink', 'contents')],
    prevent_initial_call=True
)
def measurement_analysis(n_click, source_id, sink_id, source_file, sink_file):
    """
    Analyzes measurement data and generates a latency histogram.

    This function is triggered by a button click and performs the following steps:
    1. Checks if the button has been clicked and if both source and sink files are uploaded.
    2. Saves the uploaded source and sink files to the 'uploads' directory.
    3. Reads the saved CSV files.
    4. Filters the sink data to find the last entry for the given source and destination.
    5. Extracts histogram template and values from the last entry.
    6. Creates a histogram plot of latency values.

    Args:
        n_click (int): The number of times the button has been clicked.
        source_id (str): The ID of the source router.
        sink_id (str): The ID of the sink router.
        source_file (str): The base64-encoded contents of the uploaded source file.
        sink_file (str): The base64-encoded contents of the uploaded sink file.

    Returns:
        dcc.Graph: A Dash Graph component containing the latency histogram plot.
        or
        html.Div: A Dash Div component with an error message if files are not uploaded.
    """
    if n_click is None or n_click == 0:
        return dash.no_update
    if source_file is None or sink_file is None:
        return html.Div("Please upload both source and sink files")
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    source_file_path = os.path.join('uploads', 'source.csv')
    save_base64_to_csv(source_file, source_file_path)

    # Save the uploaded sink file
    sink_file_path = os.path.join('uploads', 'sink.csv')
    save_base64_to_csv(sink_file, sink_file_path)

    # Read the saved CSV files
    _ = pd.read_csv(source_file_path)
    sink_df = pd.read_csv(sink_file_path)

    # Find the last row with the given source and destination
    source_router = net_sim.get_router_from_id(source_id)
    sink_router = net_sim.get_router_from_id(sink_id)
    filtered_df = sink_df[(sink_df['Source'] == source_router.ip_address) & (
        sink_df['Destination'] == sink_router.ip_address)]
    last_entry = filtered_df.iloc[-1]
    # Extract histogram template and values
    histogram_template = ast.literal_eval(last_entry['Histogram Template'])
    histogram_values = ast.literal_eval(last_entry['Histogram Value'])

    # Create histogram plot
    fig = px.bar(x=histogram_template, y=histogram_values, labels={'x': 'Latency (ms)',
                                                                   'y': 'Count'},
                 title=f'Latency Histogram for Source {source_id} and Destination {sink_id}')
    fig.update_xaxes(type='category')

    return dcc.Graph(figure=fig)

# Define the callback to update the layout based on the URL


@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    prevent_initial_call=True
)
def display_page(pathname):
    """
    Updates the page content based on the URL pathname.

    This function is a callback that updates the content of the 'page-content' div
    based on the current URL pathname. It returns different layouts depending on
    the pathname.

    Args:
        pathname (str): The current URL pathname.

    Returns:
        layout (dash.development.base_component.Component): The layout to be displayed.
            - If pathname is '/loading', returns the loading layout.
            - If pathname is '/menu', returns the menu layout.
            - Otherwise, returns the upload layout.
    """
    if pathname == '/loading':
        return loading_layout
    elif pathname == '/menu':
        return menu_layout
    else:
        return upload_layout


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
