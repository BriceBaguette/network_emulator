import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import tempfile
import os
from network_emulator.network_emulator import NetworkEmulator

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config['suppress_callback_exceptions'] = True
net_sim: NetworkEmulator = None

# Sample list of IDs for dropdowns
sample_ids = ["1.1.1.1"]

# Define the initial layout of the app
app.layout = dbc.Container(
    [
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='stored-file'),
        dcc.Store(id='button-click'),
        dcc.Interval(id='interval-component', interval=1 * 1000, n_intervals=0, disabled=True),
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
                    dbc.Button("Start Simulation", color="primary", size="lg", id="start-simulation-btn"),
                    width=6,  # 50% of the available space in this row
                    className="d-flex justify-content-center"  # Center the button within the column
                ),
                dbc.Col(
                    dbc.Button("ECMP Analysis", color="primary", size="lg", id="ecmp-analysis-btn"),
                    width=6,  # 50% of the available space in this row
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

# Define the content layouts for each button
start_simulation_layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Simulation Started!", className="text-center my-4")
            )
        )
    ]
)

ecmp_analysis_layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("ECMP Analysis Page", className="text-center my-4")
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
                html.H2("Additional Analysis", className="text-center my-4")
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id='additional-id-dropdown-1',
                    options=[{'label': id, 'value': id} for id in sample_ids],
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
                    options=[{'label': id, 'value': id} for id in sample_ids],
                    placeholder="Select an ID"
                ),
                width=6,
                className="mx-auto mt-3"
            )
        ),
        dbc.Row(
            dbc.Col(
                dbc.Button("Submit", id='submit-additional-analysis', color="primary", className="mt-3"),
                width=6,
                className="d-flex justify-content-center mx-auto"
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(id='additional-analysis-graph', className="mt-4")
            )
        )
    ],
    fluid=True
)

# Define the callbacks

# Callback to update dropdown options and page content based on URL pathname
@app.callback(
    [Output('ecmp-id-dropdown', 'options'),
     Output('additional-id-dropdown-1', 'options'),
     Output('additional-id-dropdown-2', 'options'),
     Output('page-content', 'children')],
    [Input('url', 'pathname')],
    [State('stored-file', 'data')],
    prevent_initial_call=True
)
def update_page_content_and_dropdowns(pathname, stored_file):
    global sample_ids

    if pathname == '/loading' and stored_file:
        global net_sim
        # Initialize NetworkEmulator and build network
        net_sim = NetworkEmulator()
        net_sim.build_network()
        sample_ids = net_sim.get_router_ids()  # Assuming method to get router IDs

        # Define menu layout for navigation after loading
        menu_layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button("Start Simulation", color="primary", size="lg", id="start-simulation-btn"),
                            width=6,  # 50% of the available space in this row
                            className="d-flex justify-content-center"  # Center the button within the column
                        ),
                        dbc.Col(
                            dbc.Button("ECMP Analysis", color="primary", size="lg", id="ecmp-analysis-btn"),
                            width=6,  # 50% of the available space in this row
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

        return ([{'label': id, 'value': id} for id in sample_ids],
                [{'label': id, 'value': id} for id in sample_ids],
                [{'label': id, 'value': id} for id in sample_ids],
                menu_layout)

    return ([], [], [], upload_layout)


# Callback to update ECMP analysis graphs based on dropdown selection
@app.callback(
    Output('ecmp-graphs', 'children'),
    [Input('ecmp-id-dropdown', 'value')],
    prevent_initial_call=True
)
def update_ecmp_graphs(selected_id):
    global net_sim

    if selected_id:
        ecmp_values = net_sim.run_ecmp_analysis(selected_id)
        # Replace with actual graph generation based on the selected ID
        return html.Div([
            dcc.Graph(figure={'data': [{'x': ecmp_values.keys, 'y': ecmp_values.values, 'type': 'line', 'name': selected_id}]}),
            dcc.Graph(figure={'data': [{'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': selected_id}]})
        ])

    return dash.no_update


# Callback to update Additional Analysis graph based on dropdowns and button click
@app.callback(
    Output('additional-analysis-graph', 'children'),
    [Input('submit-additional-analysis', 'n_clicks')],
    [State('additional-id-dropdown-1', 'value'),
     State('additional-id-dropdown-2', 'value')],
    prevent_initial_call=True
)
def update_additional_analysis(n_clicks, id1, id2):
    if n_clicks and id1 and id2:
        # Replace with actual graph generation based on the selected IDs
        return dcc.Graph(figure={'data': [{'x': [1, 2, 3], 'y': [3, 1, 4], 'type': 'line', 'name': f'{id1} vs {id2}'}]})

    return dash.no_update


# Callback to build network and update loading screen
@app.callback(
    Output('interval-component', 'disabled'),
    [Input('url', 'pathname')],
    [State('stored-file', 'data')],
    prevent_initial_call=True
)
def build_network_and_update_loading(pathname, stored_file):
    if pathname == '/loading' and stored_file:
        global net_sim
        # Initialize NetworkEmulator and build network
        net_sim = NetworkEmulator()
        net_sim.build_network()
        return False  # Disable interval component after network build

    return True


# Callback to redirect based on network status
@app.callback(
    Output('url', 'pathname'),
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def redirect_to_menu(n_intervals):
    global net_sim

    if net_sim and net_sim.is_running():
        return '/menu'

    return '/loading'


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
