import json
import asyncio
import threading

import pyarrow
import websockets
import pyarrow.parquet as pq
import pyarrow as pa
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

class SparkleServer:
    def __init__(self, host, port):
        self.max_datapoints = 1000
        self.host = host
        self.port = port
        self.data_table: pa.Table = None
        self.dimensions = ['loss', 'accuracy']  # Add default dimensions
        self.init_data_structures()

    async def handle_connection(self, websocket, path):
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed.")

    async def handle_message(self, websocket, message):
        try:
            data = json.loads(message)
            entry_type = data.get('entry_type')
            message = data.get('message')

            if entry_type == 'data_report':
                await self.handle_data_report(websocket, message)
            elif entry_type == 'reset_log':
                if data.get('dimensions'):
                    self.dimensions = data.get('dimensions')
                self.init_data_structures()  # Reinitialize the data structures
            elif entry_type == 'update_max_datapoints':
                self.max_datapoints = int(message)
            else:
                print(f"Received unknown entry type: {entry_type}")

        except json.JSONDecodeError:
            print(f"Invalid JSON message received: {message}")

    async def start_server(self):
        server = await websockets.serve(self.handle_connection, self.host, self.port)
        print(f"WebSocket server started on ws://{self.host}:{self.port}")
        await server.wait_closed()

    async def handle_data_report(self, websocket, message):
        new_table = pa.Table.from_pylist([message], self.data_table.schema)
        self.data_table = pa.concat_tables([self.data_table, new_table])

        # clip table to max rows
        max_rows = 1e6
        self.data_table = self.data_table[-int(self.max_datapoints):]

    def init_data_structures(self):
        dimension_fields = [
            pa.field('idx', pa.int64()),
        ]
        for dimension in self.dimensions:
            dimension_fields.append(pa.field(dimension, pa.float64()))

        schema = pa.schema(dimension_fields)
        self.data_table = pa.Table.from_pylist([], schema)

    def get_data(self):
        return self.data_table.to_pylist()

# Define the list of available themes
themes = [
    'bootstrap', 'cerulean', 'cosmo', 'cyborg', 'darkly', 'flatly', 'journal',
    'litera', 'lumen', 'lux', 'materia', 'minty', 'pulse', 'sandstone',
    'simplex', 'sketchy', 'slate', 'solar', 'spacelab', 'superhero', 'united', 'yeti'
]

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=["https://use.fontawesome.com/releases/v5.15.4/css/all.css"]
)

def get_theme_url(theme_name):
    return getattr(dbc.themes, theme_name.upper(), dbc.themes.BOOTSTRAP)

header = html.Div([
    dbc.Row([
        dbc.Col([
            html.H1("ML Training Server Dashboard", className="text-white"),
        ], width=10),
        dbc.Col([
            html.Button([
                html.I(className="fas fa-cog")
            ], id="settings-toggle", n_clicks=0, className="btn btn-primary",
                **{"data-bs-toggle": "collapse", "data-bs-target": "#settings-content"}),
        ], width=2, className="d-flex justify-content-end"),
    ]),
    html.Div([
        dbc.Collapse([
            dcc.Slider(
                id='max-datapoints-slider',
                min=1000,
                max=10000,
                step=1000,
                value=5000,
                marks={i: str(i) for i in range(1000, 10001, 1000)},
            ),
            dcc.Dropdown(
                id='theme-dropdown',
                options=[{'label': theme, 'value': theme} for theme in themes],
                clearable=False,
                style={'color': '#000', 'backgroundColor': '#fff'},
                value='bootstrap'
            ),
        ], id="settings-content", is_open=False),
    ], className="mt-2"),
], className="bg-primary text-white p-3")

# Add a footer
footer = html.Div([
    html.P("Copyright © 2023 Andrew Cranston. All rights reserved.", className="text-center p-3"),
], className="bg-dark text-white")

app.layout = html.Div([
    header,
    html.Div([
        html.Link(rel='stylesheet', href=get_theme_url('bootstrap')),
        dbc.Container(id='graph-container', style={'marginTop': '20px'}),
        dcc.Interval(
            id='graph-update',
            interval=1000,  # Update every 1 second
            n_intervals=0
        ),
    ], id='app-container'),
    footer,
])

@app.callback(
    Output('app-container', 'children'),
    Input('theme-dropdown', 'value')
)
def update_app_theme(theme):
    theme_url = get_theme_url(theme)
    return [
        html.Link(rel='stylesheet', href=theme_url),
        dbc.Container(id='graph-container', style={'marginTop': '20px'}),
        dcc.Interval(
            id='graph-update',
            interval=1000,  # Update every 1 second
            n_intervals=0
        ),
    ]

@app.callback(
    Output("settings-content", "is_open"),
    Input("settings-toggle", "n_clicks"),
    State("settings-content", "is_open")
)
def toggle_settings(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('graph-container', 'children'),
    Input('graph-update', 'n_intervals'),
    Input('max-datapoints-slider', 'value')
)
def update_graphs(n, max_datapoints):
    data = server.get_data()

    # display only the last max_datapoints data points
    data = data[-max_datapoints:]

    idx = [d['idx'] for d in data] if data else []

    graphs = []
    for dimension in server.dimensions:
        dimension_data = [d[dimension] for d in data] if data else []
        figure = {
            'data': [{'x': idx, 'y': dimension_data, 'type': 'line'}],
            'layout': {
                'title': f'{dimension} Graph',
                'xaxis': {'title': 'Epoch'},
                'yaxis': {'title': dimension},
            }
        }
        graphs.append(dbc.Col(dcc.Graph(figure=figure, style={'marginTop': '20px'}), sm=12, md=6))

    return dbc.Row(graphs)

async def send_max_datapoints(value):
    websocket_url = f"ws://{host}:{port}"
    try:
        async with websockets.connect(websocket_url) as websocket:
            message = json.dumps({"entry_type": "update_max_datapoints", "message": value})
            await websocket.send(message)
    except Exception as e:
        print(f"Error connecting to WebSocket server: {e}")

@app.callback(
    Output('max-datapoints-slider', 'value'),
    Input('max-datapoints-slider', 'value')
)
def update_max_datapoints(value):
    asyncio.run(send_max_datapoints(value))
    return value

async def run_websocket_server():
    await server.start_server()

def run_dash_app():
    app.run_server(debug=True, use_reloader=False, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    host = '127.0.0.1'
    port = 8765
    server = SparkleServer(host, port)

    dash_thread = threading.Thread(target=run_dash_app)
    dash_thread.start()

    asyncio.get_event_loop().run_until_complete(server.start_server())
