"""
Performance Dashboard
Subsystem 5: Dashboards & Visualization

Real-time metrics dashboard using Dash/Plotly.
Displays model performance, training metrics, and system performance.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Dashboard libraries
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px


class PerformanceDashboard:
    """Interactive dashboard for model performance metrics."""
    
    def __init__(self, metrics_file=None):
        """
        Initialize dashboard.
        
        Args:
            metrics_file: Path to metrics JSON file
        """
        self.app = dash.Dash(__name__)
        self.metrics_file = metrics_file
        self.metrics_data = {}
        
        if metrics_file and os.path.exists(metrics_file):
            self.load_metrics(metrics_file)
        
        self.setup_layout()
        self.setup_callbacks()
    
    def load_metrics(self, metrics_file):
        """Load metrics from JSON file."""
        with open(metrics_file, 'r') as f:
            self.metrics_data = json.load(f)
        print(f"Metrics loaded from {metrics_file}")
    
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            html.Div([
                html.H1("🚀 Subsystem 5: Model Training & Comparison Dashboard", 
                       style={'textAlign': 'center', 'marginBottom': 30, 'color': '#1f77b4'}),
                html.P("Advanced Computer Vision - Model Performance Analysis",
                      style={'textAlign': 'center', 'color': '#666'})
            ], style={'backgroundColor': '#f8f9fa', 'padding': 20, 'borderRadius': 5}),
            
            # Metrics Cards
            html.Div([
                html.Div([
                    html.H3("Model Summary", style={'color': '#1f77b4'}),
                    html.Hr(),
                    html.Div(id='metrics-summary', style={'fontSize': 14})
                ], className='card', style={'flex': 1, 'margin': 10, 'padding': 20, 
                                            'backgroundColor': 'white', 'borderRadius': 5, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.H3("System Information", style={'color': '#ff7f0e'}),
                    html.Hr(),
                    html.Div(id='system-info', style={'fontSize': 14})
                ], className='card', style={'flex': 1, 'margin': 10, 'padding': 20,
                                            'backgroundColor': 'white', 'borderRadius': 5, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': 30}),
            
            # Charts
            html.Div([
                html.Div([
                    dcc.Graph(id='accuracy-chart')
                ], style={'flex': 1, 'margin': 10}),
                
                html.Div([
                    dcc.Graph(id='f1-chart')
                ], style={'flex': 1, 'margin': 10})
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),
            
            # Additional charts
            html.Div([
                html.Div([
                    dcc.Graph(id='precision-recall-chart')
                ], style={'flex': 1, 'margin': 10}),
                
                html.Div([
                    dcc.Graph(id='comparison-chart')
                ], style={'flex': 1, 'margin': 10})
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),
            
            # Store for auto-refresh
            dcc.Interval(id='interval-component', interval=5000, n_intervals=0),
            
        ], style={'padding': 20, 'backgroundColor': '#fafafa', 'minHeight': '100vh'})
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [Output('metrics-summary', 'children'),
             Output('system-info', 'children'),
             Output('accuracy-chart', 'figure'),
             Output('f1-chart', 'figure'),
             Output('precision-recall-chart', 'figure'),
             Output('comparison-chart', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Metrics summary
            summary_text = self.generate_summary()
            
            # System info
            system_text = self.generate_system_info()
            
            # Charts
            acc_fig = self.create_accuracy_chart()
            f1_fig = self.create_f1_chart()
            pr_fig = self.create_precision_recall_chart()
            comp_fig = self.create_comparison_chart()
            
            return summary_text, system_text, acc_fig, f1_fig, pr_fig, comp_fig
    
    def generate_summary(self):
        """Generate metrics summary."""
        if not self.metrics_data:
            return html.Div("No metrics data available")
        
        items = []
        for model_name, metrics in self.metrics_data.items():
            items.append(html.Div([
                html.Strong(f"{model_name}:"),
                html.Ul([
                    html.Li(f"Accuracy: {metrics.get('accuracy', 0):.4f}"),
                    html.Li(f"Precision: {metrics.get('precision', 0):.4f}"),
                    html.Li(f"Recall: {metrics.get('recall', 0):.4f}"),
                    html.Li(f"F1-Score: {metrics.get('f1_score', 0):.4f}")
                ])
            ], style={'marginBottom': 15}))
        
        return html.Div(items)
    
    def generate_system_info(self):
        """Generate system information."""
        import platform
        import psutil
        
        items = [
            html.Div(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style={'marginBottom': 10}),
            html.Div(f"🖥️ System: {platform.system()} {platform.release()}", style={'marginBottom': 10}),
            html.Div(f"⚙️ Processor: {platform.processor()}", style={'marginBottom': 10}),
            html.Div(f"💾 RAM: {psutil.virtual_memory().percent}% used", style={'marginBottom': 10}),
        ]
        
        # GPU info if available
        try:
            import tensorflow as tf
            gpu_available = tf.config.list_physical_devices('GPU')
            items.append(html.Div(f"🎮 GPU: {len(gpu_available)} device(s) available"))
        except:
            items.append(html.Div("🎮 GPU: Not available"))
        
        return html.Div(items)
    
    def create_accuracy_chart(self):
        """Create accuracy comparison chart."""
        if not self.metrics_data:
            return go.Figure()
        
        models = list(self.metrics_data.keys())
        accuracies = [self.metrics_data[m].get('accuracy', 0) for m in models]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=accuracies, marker_color='#1f77b4', text=accuracies,
                   textposition='auto', name='Accuracy')
        ])
        fig.update_layout(
            title='Model Accuracy Comparison',
            xaxis_title='Model',
            yaxis_title='Accuracy',
            height=400,
            yaxis=dict(range=[0, 1])
        )
        return fig
    
    def create_f1_chart(self):
        """Create F1-score comparison chart."""
        if not self.metrics_data:
            return go.Figure()
        
        models = list(self.metrics_data.keys())
        f1_scores = [self.metrics_data[m].get('f1_score', 0) for m in models]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=f1_scores, marker_color='#2ca02c', text=f1_scores,
                   textposition='auto', name='F1-Score')
        ])
        fig.update_layout(
            title='F1-Score Comparison',
            xaxis_title='Model',
            yaxis_title='F1-Score',
            height=400,
            yaxis=dict(range=[0, 1])
        )
        return fig
    
    def create_precision_recall_chart(self):
        """Create precision-recall chart."""
        if not self.metrics_data:
            return go.Figure()
        
        models = list(self.metrics_data.keys())
        precisions = [self.metrics_data[m].get('precision', 0) for m in models]
        recalls = [self.metrics_data[m].get('recall', 0) for m in models]
        
        fig = go.Figure(data=[
            go.Scatter(x=recalls, y=precisions, mode='markers+text', text=models,
                      marker=dict(size=15, color='#ff7f0e'), textposition='top center')
        ])
        fig.update_layout(
            title='Precision vs Recall',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=400,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        return fig
    
    def create_comparison_chart(self):
        """Create multi-metric comparison chart."""
        if not self.metrics_data:
            return go.Figure()
        
        models = list(self.metrics_data.keys())
        
        fig = go.Figure()
        
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for metric, color in zip(metrics_names, colors):
            values = [self.metrics_data[m].get(metric, 0) for m in models]
            fig.add_trace(go.Bar(x=models, y=values, name=metric.replace('_', ' ').title(),
                                marker_color=color))
        
        fig.update_layout(
            title='Comprehensive Model Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=400,
            yaxis=dict(range=[0, 1])
        )
        return fig
    
    def run(self, host='127.0.0.1', port=8050, debug=False):
        """Run the dashboard server."""
        print(f"\n{'='*80}")
        print(f"Starting Performance Dashboard...")
        print(f"Access the dashboard at: http://{host}:{port}")
        print(f"{'='*80}\n")
        self.app.run_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    # Example usage
    dashboard = PerformanceDashboard()
    dashboard.run(debug=True)
