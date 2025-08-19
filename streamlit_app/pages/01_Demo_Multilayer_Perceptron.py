import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import io
import sys
from pathlib import Path

# Import neural network modules from the core directory
root_path = Path.cwd().parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from NeuralKit import *

# Set page config
st.set_page_config(
    page_title="Multilayer Perceptron Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #FF6B6B;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .sub-header {
            font-size: 1.5rem;
            color: #4ECDC4;
            margin-bottom: 1rem;
        }
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .stSelectbox label, .stSlider label {
            color: #FF6B6B !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Try to import your actual neural network modules
def load_nn_modules():
    """Load neural network modules"""
    try:
        # Add the parent directory to sys.path to import core modules
        root_path = Path.cwd().parent
        if str(root_path) not in sys.path:
            sys.path.insert(0, str(root_path))
        
        # Import your actual modules
        from NeuralKit.network import NeuralNetwork
        from NeuralKit.layers import Linear, ReLULayer, SigmoidLayer, TanhLayer, SoftmaxCrossEntropyLoss
        from NeuralKit.optimizers import Adam, SGD, Momentum, RMSProp
        from NeuralKit.loss import mse_loss, mse_loss_derivative, cross_entropy_loss
        from NeuralKit.utils import batch_iterator
        from NeuralKit.data_generator import Spiral, Circle, Zone, GeneratePolynomialData
        
        return {
            'NeuralNetwork': NeuralNetwork,
            'Linear': Linear,
            'ReLULayer': ReLULayer,
            'SigmoidLayer': SigmoidLayer,
            'TanhLayer': TanhLayer,
            'SoftmaxCrossEntropyLoss': SoftmaxCrossEntropyLoss,
            'Adam': Adam,
            'SGD': SGD,
            'Momentum': Momentum,
            'RMSProp': RMSProp,
            'mse_loss': mse_loss,
            'mse_loss_derivative': mse_loss_derivative,
            'cross_entropy_loss': cross_entropy_loss,
            'batch_iterator': batch_iterator,
            'Spiral': Spiral,
            'Circle': Circle,
            'Zone': Zone,
            'GeneratePolynomialData': GeneratePolynomialData
        }
    except ImportError as e:
        st.error(f"Could not import neural network modules: {e}")
        st.info("Make sure your 'core' module is in the parent directory")
        return None

# Data generation functions using your actual classes
class DataGenerator:
    def __init__(self, nn_modules):
        self.nn_modules = nn_modules
    
    def spiral_data(self, n_points=200, n_classes=3):
        """Generate spiral classification data using your Spiral class"""
        if self.nn_modules and 'Spiral' in self.nn_modules:
            spiral = self.nn_modules['Spiral'](n_points, n_classes, 2)
            return spiral.generate()
        else:
            # Fallback implementation
            return self._fallback_spiral(n_points, n_classes)
    
    def circle_data(self, n_points=200, n_classes=3):
        """Generate circular classification data using your Circle class"""
        if self.nn_modules and 'Circle' in self.nn_modules:
            circle = self.nn_modules['Circle'](n_points, n_classes, 2)
            return circle.generate()
        else:
            return self._fallback_circle(n_points, n_classes)
    
    def polynomial_data(self, n_points=200):
        """Generate polynomial regression data"""
        if self.nn_modules and 'GeneratePolynomialData' in self.nn_modules:
            poly_gen = self.nn_modules['GeneratePolynomialData'](n_points, [1, -2, 3], 2)
            data = poly_gen.generate()
            return data['x'].reshape(-1, 1), data['y']
        else:
            x = np.linspace(-3, 3, n_points)
            y = 0.5 * x**3 - 2 * x**2 + x + np.random.normal(0, 1, n_points)
            return x.reshape(-1, 1), y
    
    def xor_data(self):
        """Generate XOR problem data"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 1, 0], dtype=np.uint8)
        return X, y
    
    def _fallback_spiral(self, n_points, n_classes):
        """Fallback spiral data generation"""
        X = np.zeros((n_points * n_classes, 2))
        y = np.zeros(n_points * n_classes, dtype='uint8')
        
        for j in range(n_classes):
            ix = range(n_points * j, n_points * (j + 1))
            r = np.linspace(0.0, 1, n_points)
            t = np.linspace(j * 4, (j + 1) * 4, n_points) + np.random.randn(n_points) * 0.2
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j
        return X, y
    
    def _fallback_circle(self, n_points, n_classes):
        """Fallback circle data generation"""
        X = np.zeros((n_points * n_classes, 2))
        y = np.zeros(n_points * n_classes, dtype='uint8')
        
        for j in range(n_classes):
            ix = range(n_points * j, n_points * (j + 1))
            r = (j + 1) * 2 + np.random.randn(n_points) * 0.5
            t = np.linspace(0, 2 * np.pi, n_points) + np.random.randn(n_points) * 0.2
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j
        return X, y

def plot_data_2d(X, y, title="Data Visualization"):
    """Plot 2D data with different colors for each class"""
    fig = px.scatter(
        x=X[:, 0], y=X[:, 1], 
        color=y.astype(str),
        title=title,
        labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Class'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    return fig

def plot_training_history(losses, accuracies=None):
    """Plot training history"""
    epochs = range(1, len(losses) + 1)
    
    if accuracies is not None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Loss', 'Training Accuracy')
        )
        fig.add_trace(
            go.Scatter(x=list(epochs), y=losses, name='Loss', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(epochs), y=accuracies, name='Accuracy', line=dict(color='blue')),
            row=1, col=2
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(epochs), y=losses, name='Loss', line=dict(color='red')))
        fig.update_layout(title='Training Loss', xaxis_title='Epoch', yaxis_title='Loss')
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    return fig

def train_model(model, optimizer, X, y, epochs, batch_size, problem_type):
    """Train the model with real implementation"""
    losses = []
    accuracies = [] if "Classification" in problem_type else None
    
    # Prepare progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0
        

        batch_iter = batch_iterator(X, y, batch_size, shuffle=True)
        
        for X_batch, y_batch in batch_iter:
            model.zero_grad()
            
            # Forward pass
            if "Classification" in problem_type:
                logits = model.forward(X_batch)
                
                # Multi-class classification with softmax cross entropy
                if nn_modules and 'SoftmaxCrossEntropyLoss' in nn_modules:
                    loss_fn = nn_modules['SoftmaxCrossEntropyLoss']()
                    loss = loss_fn.forward(logits, y_batch)
                    grad = loss_fn.backward()
                else:
                    loss = nn_modules['cross_entropy_loss'](logits, y_batch)
                    grad = 2 * (logits - y_batch) / len(y_batch)  # Simplified
                    
                    # Accuracy calculation
                    predictions = np.argmax(logits, axis=1)
                    epoch_correct += np.sum(predictions == y_batch)
            else:
                # Regression
                predictions = model.forward(X_batch)
                loss = nn_modules['mse_loss'](predictions, y_batch.reshape(-1, 1))
                grad = nn_modules['mse_loss_derivative'](predictions, y_batch.reshape(-1, 1))
            
            epoch_loss += loss
            epoch_total += len(y_batch)
            batch_count += 1
            
            # Backward pass
            model.backward(grad)
            model.update_params(optimizer)
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / max(batch_count, 1)
        losses.append(avg_loss)
        
        if accuracies is not None:
            acc = epoch_correct / max(epoch_total, 1)
            accuracies.append(acc)
        
        # Update progress
        progress_bar.progress((epoch + 1) / epochs)
        if epoch == epochs - 1:
            progress_bar.empty()

        # Allow for early stopping or UI updates
        if epoch % 10 == 0:
            time.sleep(0.01)  # Small delay to allow UI updates
    
    return losses, accuracies

def main():
    # # Load neural network modules
    nn_modules = load_nn_modules()
    if nn_modules is None:
        st.warning("Using fallback implementations. Neural network training may not work properly.")
        return
    
    # Initialize data generator
    # data_gen = DataGenerator(nn_modules)
    
    # Main title
    st.markdown('<h1 class="main-header">MLP Demo</h1>', unsafe_allow_html=True)
    st.markdown("<br></br>", unsafe_allow_html=True)
    st.markdown("""
            <div style="text-align: center;">
                <h5>Description: Built from scratch with customizable architecture and real-time visualization</h5>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("", unsafe_allow_html=True)

    col1, space, col2, space, col3 = st.columns([3, 0.5, 3, 0.5, 3])

    with col1:
        # configuration
        st.markdown('<h3 class="sub-header"> Configuration</h3>', unsafe_allow_html=True)
        
        # Problem type selection
        problem_type = st.selectbox(
            "Select type of problem",
            ["Classification - Spiral", "Classification - Circles", "Regression - Polynomial"]
        )
        # Data generation
        st.markdown('<h3 class="sub-header"> Data parameters</h3>', unsafe_allow_html=True) 
        
        if "Classification" in problem_type:
            if "XOR" in problem_type:
                n_samples = 4
                st.info("XOR problem has fixed 4 samples")
            else:
                n_samples = st.slider("Number of samples per class", 1, 500, 200)
                n_classes = st.slider("Number of classes", 2, 5, 3)
        else:
            n_samples = st.slider("Number of samples", 100, 1000, 200)
    with col2:
        # Network architecture
        st.markdown("<h3>Network architecture</h3>", unsafe_allow_html=True) 
        
        if "XOR" in problem_type:
            hidden_sizes = [4]  # Fixed for XOR
            st.info("XOR uses fixed architecture: 2 → 4 → 1")
        else:
            n_hidden_layers = st.slider("Number of hidden layers", 1, 5, 2)
            hidden_sizes = []
            for i in range(n_hidden_layers):
                size = st.slider(f"Hidden layer {i+1} size", 8, 512, 64)
                hidden_sizes.append(size)
        
        activation = st.selectbox("Activation function", ["ReLU", "Sigmoid", "Tanh"])
    with col3:
        # Training parameters
        st.markdown("<h3>Training parameters</h3>", unsafe_allow_html=True)
        
        optimizer_type = st.selectbox("Optimizer", ["Adam", "SGD", "Momentum", "RMSProp"])
        learning_rate = st.slider("Learning rate", 0.0001, 0.1, 0.001, format="%.4f")
        epochs = st.slider("Epochs", 10, 1000, 100)
        batch_size = st.slider("Batch size", 8, 256, 32) if not "XOR" in problem_type else 4
    
    
    # Generate data
    col1, col2, col3 = st.columns([1.5,1,1.5])
    with col2:
        st.markdown("", unsafe_allow_html=True)
        if st.button("# Click here to generate new data", type="primary", width="stretch"):
            if 'X' in st.session_state:
                del st.session_state.X
                del st.session_state.y
        
        if 'X' not in st.session_state:
            with st.spinner("Generating data..."):
                if problem_type == "Classification - Spiral":
                    X, y = SpiralDataset(n_samples, n_classes).generate()
                elif problem_type == "Classification - Circles":
                    X, y = CircleDataset(n_samples, n_classes).generate()
                else:  # Regression
                    X, y = PolynomialDataset(n_samples).generate()
                
                st.session_state.X = X
                st.session_state.y = y
    
    X, y = st.session_state.X, st.session_state.y
    
    st.markdown("<br></br>", unsafe_allow_html=True)
    
    # Main content area
    col1, space, col2 = st.columns([6, 1, 4])
    
    with col1:
        st.markdown('<h3 class="sub-header">Data Visualization</h3>', unsafe_allow_html=True)
        
        if X.shape[1] == 2:  # 2D data
            fig_data = plot_data_2d(X, y, f"{problem_type} Data")
            st.plotly_chart(fig_data, use_container_width=True)
        else:  # 1D regression data
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Data'))
            fig.update_layout(
                title="Regression Data",
                xaxis_title="X",
                yaxis_title="y",
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="sub-header">Network Architecture</h3>', unsafe_allow_html=True)
        st.markdown('<br></br>', unsafe_allow_html=True)
        # Display network architecture
        arch_text = f"Input ({X.shape[1]})"
        for i, size in enumerate(hidden_sizes):
            arch_text += f" → Hidden {i+1} ({size})"
        
        if "Classification" in problem_type:
            if "XOR" in problem_type:
                output_size = 1
            else:
                output_size = len(np.unique(y))
        else:
            output_size = 1
        
        arch_text += f" → Output ({output_size})"
        
        st.code(arch_text)
        
        # Training parameters summary
        st.markdown("**Training Configuration:**")
        st.write(f"- Optimizer: {optimizer_type}")
        st.write(f"- Learning Rate: {learning_rate}")
        st.write(f"- Epochs: {epochs}")
        st.write(f"- Batch Size: {batch_size}")
        st.write(f"- Activation: {activation}")
    
    col1, col2, col3 = st.columns([1.5,1,1.5])
    with col2:
        st.markdown("", unsafe_allow_html=True)
        # Training section
        if st.button("Start Training", type="primary", width="stretch"):
            with st.spinner("Training neural network..."):
                # Create model
                model = nn_modules['NeuralNetwork']()
                
                # Add layers
                prev_size = X.shape[1]
                for hidden_size in hidden_sizes:
                    model.add_layer(nn_modules['Linear'](prev_size, hidden_size, init_type='he'))
                    
                    if activation == "ReLU":
                        model.add_layer(nn_modules['ReLULayer']())
                    elif activation == "Sigmoid":
                        model.add_layer(nn_modules['SigmoidLayer']())
                    else:  # Tanh
                        model.add_layer(nn_modules['TanhLayer']())
                        
                    prev_size = hidden_size
                
                # Output layer
                model.add_layer(nn_modules['Linear'](prev_size, output_size))
                
                if "XOR" in problem_type:
                    model.add_layer(nn_modules['SigmoidLayer']())
                
                # Create optimizer
                if optimizer_type == "Adam":
                    optimizer = nn_modules['Adam'](learning_rate)
                elif optimizer_type == "SGD":
                    optimizer = nn_modules['SGD'](learning_rate)
                elif optimizer_type == "Momentum":
                    optimizer = nn_modules['Momentum'](learning_rate)
                else:  # RMSProp
                    optimizer = nn_modules['RMSProp'](learning_rate)
                
                # Train model
                losses, accuracies = train_model(
                    model, optimizer, X, y, epochs, batch_size, problem_type, nn_modules
                )
                
                # Store results in session state
                st.session_state.training_complete = True
                st.session_state.losses = losses
                st.session_state.accuracies = accuracies
                st.session_state.model = model
                st.session_state.output_size = output_size
                
                msg = st.empty()
                msg.success("Training complete!")
                time.sleep(2)   # Hiện trong 1 giây
                msg.empty()     # Xóa message
                

if __name__ == "__main__":
    main()