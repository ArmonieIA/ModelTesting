# resource_dashboard.py

from dash import dcc, html, Input, Output, State, callback
from dash.dependencies import Input, Output
from tensorflow.keras.callbacks import Callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import GPUtil
import dash
import psutil
import platform 
import sys
import subprocess
import json
import time

class CustomTrainingProgressCallback(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.total_batches = self.params['steps']
        self.epoch_time_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1  # Epochs are zero-indexed in Keras

    def on_batch_end(self, batch, logs=None):
        self.current_batch = batch + 1  # Batches are zero-indexed in Keras
        time_per_step = time.time() - self.epoch_time_start
        # Write the progress information to a file
        progress_data = {
            'current_epoch': self.current_epoch,
            'total_epochs': self.epochs,
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'time_per_step': f"{time_per_step:.2f}s/step"
        }
        with open('training_progress.json', 'w') as f:
            json.dump(progress_data, f)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_time_start = time.time()  # Reset timer at the end of each epoch

# Instantiate the callback
progress_callback = CustomTrainingProgressCallback()

def get_cuda_version():
    try:
        # Use the command line to get the CUDA version
        cuda_version = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode().strip()
        return cuda_version
    except Exception as e:
        return f"Could not get CUDA version: {e}"

# In your update_system_info function, replace the GPUtil.getDriverVersion() call with get_cuda_version()

# Initialize the Dash panel with a dark theme
panel = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the panel
panel.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("System Resource Usage and training dashboard", className="text-center text-primary mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.H2("System Resource Usage", className="text-center text-primary mb-4"), width=12),
        dbc.Col([
            html.Div(id='system-info', className="text-center mt-4")  # Container for system info
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='live-update-graph-cpu'),
            html.Div(id='cpu-info', className="text-center"),  # Container for CPU info
            dcc.Interval(
                id='interval-component-cpu',
                interval=1*1000,  # in milliseconds
                n_intervals=0
            )
        ], width=6),
        dbc.Col([
            dcc.Graph(id='live-update-graph-ram'),
            html.Div(id='ram-info', className="text-center"),  # Container for RAM info
            dcc.Interval(
                id='interval-component-ram',
                interval=1*1000,  # in milliseconds
                n_intervals=0
            )
        ], width=6),
        dbc.Col([
            dcc.Graph(id='live-update-graph-gpu'),
            html.Div(id='gpu-info', className="text-center"),  # Container for GPU info
            dcc.Interval(
                id='interval-component-gpu',
                interval=1*1000,  # in milliseconds
                n_intervals=0
            )
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            html.H2("Training Information", className="text-center text-primary mb-4"),
            dcc.Graph(id='training-graph'),  # Graph for training metrics
            dcc.Interval(
                id='interval-component-training',
                interval=5*1000,  # in milliseconds, adjust as needed
                n_intervals=0
            ),
            html.H4("Training State", className="text-center text-primary mb-4"),  # Title for the training state
            # Update the Progress component to include a label and color
            dbc.Progress(id='training-progress-bar', value=0, max=100, striped=True, animated=True, color="primary", label=""),  # Set color to blue and label to empty
            html.Div(id='training-state-info', className="text-center mt-4")  # Container for training state info
        ], width=12)
    ])
], fluid=True)

# Define the callback for updating the CPU graph and info
@panel.callback(
    [Output('live-update-graph-cpu', 'figure'),
     Output('cpu-info', 'children')],
    [Input('interval-component-cpu', 'n_intervals')]
)
def update_graph_cpu(n):
    cpu_percent = psutil.cpu_percent()
    cpu_figure = {
        'data': [{'type': 'indicator',
                  'mode': 'gauge+number',
                  'value': cpu_percent,
                  'title': {'text': "CPU Usage (%)"},
                  'gauge': {'axis': {'range': [0, 100]}}}],
        'layout': {'height': 300}
    }
    cpu_usage = psutil.cpu_percent(percpu=True)
    cpu_info = f"CPU Usage: {sum(cpu_usage):.2f}% / {len(cpu_usage)} Cores"
    return cpu_figure, cpu_info

# Define the callback for updating the RAM graph and info
@panel.callback(
    [Output('live-update-graph-ram', 'figure'),
     Output('ram-info', 'children')],
    [Input('interval-component-ram', 'n_intervals')]
)
def update_graph_ram(n):
    ram = psutil.virtual_memory()
    ram_percent = ram.percent
    ram_total_gb = ram.total / (1024**3)
    ram_used_gb = ram.used / (1024**3)
    ram_figure = {
        'data': [{'type': 'indicator',
                  'mode': 'gauge+number+delta',
                  'value': ram_percent,
                  'delta': {'reference': 50},
                  'title': {'text': "RAM Usage (%)"},
                  'gauge': {'axis': {'range': [0, 100]}}}],
        'layout': {'height': 300}
    }
    ram_info = f"RAM Used: {ram_used_gb:.2f} GB / {ram_total_gb:.2f} GB"
    return ram_figure, ram_info

# Define the callback for updating the GPU graph and info
@panel.callback(
    [Output('live-update-graph-gpu', 'figure'),
     Output('gpu-info', 'children')],
    [Input('interval-component-gpu', 'n_intervals')]
)
def update_graph_gpu(n):
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assuming one GPU
        gpu_load = gpu.load * 100
        gpu_total_gb = gpu.memoryTotal / 1024
        gpu_used_gb = gpu.memoryUsed / 1024
        gpu_figure = {
            'data': [{'type': 'indicator',
                      'mode': 'gauge+number',
                      'value': gpu_load,
                      'title': {'text': "GPU Usage (%)"},
                      'gauge': {'axis': {'range': [0, 100]}}}],
            'layout': {'height': 300}
        }
        gpu_info = f"GPU Used: {gpu_used_gb:.2f} GB / {gpu_total_gb:.2f} GB"
    else:
        gpu_info = "No GPU detected"
        gpu_figure = {
            'data': [{'type': 'indicator',
                      'mode': 'gauge+number',
                      'value': 0,
                      'title': {'text': "GPU Usage (%)"},
                      'gauge': {'axis': {'range': [0, 100]}}}],
            'layout': {'height': 300}
        }
    return gpu_figure, gpu_info

# Define the callback for updating the system information
@panel.callback(
    Output('system-info', 'children'),
    [Input('interval-component-cpu', 'n_intervals')]
)
def update_system_info(n):
    uname = platform.uname()
    system_info = f"""
    Session: {uname.node} | 
    GPU: {', '.join([gpu.name for gpu in GPUtil.getGPUs()])} | 
    CUDA: {get_cuda_version()} | 
    Python: {sys.version.split()[0]} | 
    OS: {uname.system} {uname.release}
    """
    return system_info

@panel.callback(
    [Output('training-graph', 'figure'),
     Output('training-progress-bar', 'value'),
     Output('training-progress-bar', 'label'),  # This will update the label inside the progress bar
     Output('training-state-info', 'children')],  # This will update the training state info
    [Input('interval-component-training', 'n_intervals')]
)
def update_training_graph(n):
    # Read the training progress data from the file
    try:
        with open('training_progress.json', 'r') as f:
            progress_data = json.load(f)
        current_epoch = progress_data['current_epoch']
        total_epochs = progress_data['total_epochs']
        current_batch = progress_data['current_batch']
        total_batches = progress_data['total_batches']
        time_per_step = progress_data['time_per_step']
    except (FileNotFoundError, json.JSONDecodeError):
        # Default values if the file is not found or cannot be read
        current_epoch = 1
        total_epochs = 32
        current_batch = 1
        total_batches = 100
        time_per_step = "0s/step"

    # Here you would retrieve your actual training metrics
    # For this example, we'll just generate some dummy data
    epochs = list(range(1, current_epoch + 1))
    loss = [1/(epoch + 1) for epoch in epochs]  # Dummy loss data
    accuracy = [epoch / max(epochs) for epoch in epochs]  # Dummy accuracy data
    recall = [epoch / max(epochs) for epoch in epochs]  # Dummy recall data
    precision = [epoch / max(epochs) for epoch in epochs]  # Dummy precision data

    # Create the figure with loss, accuracy, recall, and precision
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines+markers', name='Loss'))
    fig.add_trace(go.Scatter(x=epochs, y=accuracy, mode='lines+markers', name='Accuracy'))
    fig.add_trace(go.Scatter(x=epochs, y=recall, mode='lines+markers', name='Recall'))
    fig.add_trace(go.Scatter(x=epochs, y=precision, mode='lines+markers', name='Precision'))

    # Update layout
    fig.update_layout(title='Training Metrics Over Epochs',
                      xaxis_title='Epoch',
                      yaxis_title='Metric Value')


    # Calculate the progress percentage for the progress bar
    progress_percentage = (current_epoch / total_epochs) * 100

    # Update the progress bar label to show percentage completion
    progress_label = f"{progress_percentage:.2f}% Complete"

    # Update the training state info
    training_state_info = f"Epoch {current_epoch}/{total_epochs} - " \
                          f"{current_batch}/{total_batches} batches - " \
                          f"{time_per_step}"

    # Make sure to return four values here, one for each Output in the callback
    return fig, progress_percentage, progress_label, training_state_info

