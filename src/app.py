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

@app.callback(
    Output('ecmp-id-dropdown', 'options'),
    Output('additional-id-dropdown-1', 'options'),
    Output('additional-id-dropdown-2', 'options'),
    Input('ecmp-analysis-btn', 'n_clicks'),
    prevent_initial_call=True
)
def update_dropdown_options(n_clicks):
    # Assuming net_sim and sample_ids are defined and accessible
    if net_sim.is_running():
        print("Network is running")  # Debug statement
        sample_ids = net_sim.get_routers_ids()  # Assuming this retrieves router IDs
        dropdown_options = [{'label': id, 'value': id} for id in sample_ids]
        return dropdown_options, dropdown_options, dropdown_options
    
    # If net_sim is not running or sample_ids is not populated, return empty options
    return [], [], []



# Define the callback to store the uploaded file contents and navigate to the loading screen
@app.callback(
    [Output('stored-file', 'data'),
     Output('url', 'pathname')],
    [Input('upload-data', 'contents')],
    prevent_initial_call=True
)
def store_file_and_redirect(contents):
    if contents is not None:
        print("File uploaded successfully")  # Debug statement
        return contents, '/loading'
    return None, '/'

# Define the callback to update the content based on button click
@app.callback(
    Output('button-output', 'children'),
    [Input('start-simulation-btn', 'n_clicks'),
     Input('ecmp-analysis-btn', 'n_clicks')],
    prevent_initial_call=True
)
def update_output(start_simulation_n, ecmp_analysis_n):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'start-simulation-btn':
        return start_simulation_layout
    elif button_id == 'ecmp-analysis-btn':
        return ecmp_analysis_layout
    return dash.no_update

# Define the callback to update the ECMP Analysis content based on dropdown selection
@app.callback(
    Output('ecmp-graphs', 'children'),
    Input('ecmp-id-dropdown', 'value'),
    prevent_initial_call=True
)
def update_ecmp_graphs(selected_id):
    print(f"Sample IDs {sample_ids}")  # Debug statement
    if selected_id:
        ecmp_values = net_sim.ecmp_analysis(selected_id)
        # Replace with actual graph generation based on the selected ID
        return html.Div([
            dcc.Graph(figure={'data': [{'x': ecmp_values.keys, 'y': ecmp_values.values, 'type': 'line', 'name': selected_id}]}),
            dcc.Graph(figure={'data': [{'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': selected_id}]})
        ])
    return dash.no_update

# Define the callback to update the Additional Analysis content based on dropdowns and button click
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
    if net_sim and net_sim.is_running():  # Assuming there's a method to check the network status
        return '/menu'
    return '/loading'


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
    app.run_server(debug=True)
