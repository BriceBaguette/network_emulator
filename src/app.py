import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import tempfile
import os
from network_emulator.network_emulator import NetworkEmulator
import networkx as nx
import plotly.graph_objs as go
import random
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
import datetime
import zipfile
import pandas as pd
import plotly.express as px
import io


cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config['suppress_callback_exceptions'] = True
net_sim: NetworkEmulator = None

# Sample list of IDs for dropdowns
sample_ids = []

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
    # Create a simple networkx graph
    G = net_sim.get_topology()

    # Get the positions of the nodes using a layout algorithm with increased spacing
    # Adjust the value of k to control node spacing
    pos = nx.spring_layout(G, k=1, dim=2)

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
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
    )
    return fig


# Define the content layouts for each button
def start_simulation_layout() -> dbc.Container:
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
                                        type="number", id="number-of-generation", value=net_sim.num_generation),
                                ]
                            ),
                            dbc.CardGroup(
                                [
                                    dbc.Label("Telemtry Period in minute",
                                              html_for="telemetry-period"),
                                    dbc.Input(
                                        type='number', id="telemetry-period", value=net_sim.duration/60),
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
    if selected_id:
        ecmp_values, path = net_sim.ecmp_analysis(selected_id)
        # Replace with actual graph generation based on the selected ID
        return html.Div([
            dcc.Graph(figure={'data': [{'x': list(ecmp_values.keys()), 'y': list(ecmp_values.values()), 'type': 'bar', 'name': selected_id}],
                              'layout': {'title': f'Number of destinations by number of paths for {selected_id}',
                                         'xaxis': {'title': 'Number of paths'},
                                         'yaxis': {'title': 'Number of destinations'}}}),
            dcc.Graph(figure={'data': [{'x': list(path.keys()), 'y': list(path.values()), 'type': 'linear', 'name': selected_id}],
                              'layout': {'title': f'Latency with number of hops with {selected_id} as source',
                                         'xaxis': {'title': 'Number of hops'},
                                         'yaxis': {'title': 'Latency in ms'}}}),
        ])
    return dash.no_update

# Define the callback to update the Additional Analysis content based on dropdowns and button click


@app.callback(
    Output('additional-analysis-output', 'children'),
    [Input('submit-additional-analysis', 'n_clicks')],
    [State('additional-id-dropdown-1', 'value'),
     State('additional-id-dropdown-2', 'value')],
    prevent_initial_call=True
)
def update_additional_analysis(n_clicks, id1, id2):
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
        for id in sample_ids:
            if id != id1:
                _, wrong_path = net_sim.latency_test(id1, id)
                wrong_paths.extend(wrong_path)
    else:
        wrong_paths = net_sim.all_latency_test()
    if len(wrong_paths) > 0:
        text_items = [html.Li(f"Path from {wrong_paths[0][0]} to {path[0][-1]} going through {
                              path[0][1:-1]} has a latency imbalance of {path[1]/1000} ms") for path in wrong_paths]
    else:
        text_items = html.Li("No wrong paths found")

    return html.Div([
        dcc.Graph(figure={'data': [{'x': list(graph_value.keys()), 'y': list(graph_value.values()), 'type': 'bar', 'name': f"Latency for number of paths between {id1} and {id2}"}],
                          'layout': {'title': f"Latency for number of paths between {id1} and {id2}",
                                     'xaxis': {'title': 'Latency in ms'},
                                     'yaxis': {'title': 'Number of paths'}}}) if len(graph_value.items()) > 0 else html.Div(),
        html.Ul(text_items)])

# Define the callback to build the network


@app.callback(
    Output('interval-component', 'disabled'),
    Input('url', 'pathname'),
    State('stored-file', 'data'),
    prevent_initial_call=True
)
def build_network(pathname, stored_file):
    if pathname == '/loading' and stored_file is not None:
        global net_sim
        # Decode the base64 string
        content_type, content_string = stored_file.split(',')
        decoded = base64.b64decode(content_string)
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(decoded)
            temp_file_path = temp_file.name

        net_sim = NetworkEmulator(node_file=None, link_file=None, single_file=temp_file_path,
                                  generation_rate=20, num_generation=1, load_folder=None, save_folder=None)
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
def start_emulation(n_clicks, n_intervals, router_id, start, end):
    
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

        if net_sim.current_step % (net_sim.generation_rate*net_sim.duration) == 0 and net_sim.current_step != 0:
            timestamp = datetime.datetime.now()
            print(f"Exporting data at {timestamp}")
            for router in routers:
                net_sim.export_source_data(source_router=router, fl=None, gen_number=net_sim.current_step //
                                           (net_sim.generation_rate*net_sim.duration), timestamp=timestamp)
                net_sim.export_sink_data(sink_router=router, fl=None, gen_number=net_sim.current_step //
                                         (net_sim.generation_rate*net_sim.duration), timestamp=timestamp)
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
    source_message = ''
    sink_message = ''
    analysis_layout = dbc.Container()

    if source_filename is not None:
        if 'source' in source_filename.lower():
            source_message = f'File "{source_filename}" uploaded successfully.'
            if source_contents:
                content_type, content_string = source_contents.split(',')
                decoded = base64.b64decode(content_string)
                source_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            source_message = 'Please upload a CSV file containing "source" in its name.'

    if sink_filename is not None:
        if 'sink' in sink_filename.lower():
            sink_message = f'File "{sink_filename}" uploaded successfully.'
            if sink_contents:
                content_type, content_string = sink_contents.split(',')
                decoded = base64.b64decode(content_string)
                sink_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            sink_message = 'Please upload a CSV file containing "sink" in its name.'

    if source_message and sink_message and 'uploaded successfully' in source_message and 'uploaded successfully' in sink_message:

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
    content_type, content_string = base64_str.split(',')
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
    source_df = pd.read_csv(source_file_path)
    sink_df = pd.read_csv(sink_file_path)

    # Find the last row with the given source and destination
    source_router = net_sim.get_router_from_id(source_id)
    sink_router = net_sim.get_router_from_id(sink_id)
    filtered_df = sink_df[(sink_df['Source'] == source_router.ip_address) & (
        sink_df['Destination'] == sink_router.ip_address)]
    last_entry = filtered_df.iloc[-1]
    # Extract histogram template and values
    histogram_template = eval(last_entry['Histogram Template'])
    histogram_values = eval(last_entry['Histogram Value'])

    # Create histogram plot
    fig = px.bar(x=histogram_template, y=histogram_values, labels={'x': 'Latency (ms)', 'y': 'Count'},
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
    if pathname == '/loading':
        return loading_layout
    elif pathname == '/menu':
        return menu_layout
    else:
        return upload_layout


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
