import time
import sys
import numpy as np
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go   
import plotly.express as px
from plotly.subplots import make_subplots

root_path = Path.cwd().parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from NeuralKit import *
from theme import *

st.set_page_config(
    page_title= "Demo Multilayer Perceptron",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_css(), unsafe_allow_html=True)

def Optimizer(optimizer_type, learning_rate):
    # Create an optimizer based on the selected type
    if optimizer_type == "Adam":
        return Adam(learning_rate=learning_rate)
    elif optimizer_type == "SGD":
        return SGD(learning_rate=learning_rate)
    elif optimizer_type == "Momentum":
        return Momentum(learning_rate=learning_rate)
    elif optimizer_type == "RMSProp":
        return RMSProp(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def create_model(input_size, hidden_sizes, output_size, activation):
    # Create a neural network model with specified architecture.
    if activation == "ReLU":
        activation = ReLULayer()
    elif activation == "Sigmoid":
        activation = SigmoidLayer()
    elif activation == "Tanh":
        activation = TanhLayer()
    else:
        raise ValueError(f"Unknown activation function: {activation}")
    
    # Initialize the model
    model = NeuralNetwork()

    for i, hidden_size in enumerate(hidden_sizes):
        if i == 0:
            model.add_layer(Linear(input_size, hidden_size))
            model.add_layer(activation)
        else:
            model.add_layer(Linear(hidden_sizes[i-1], hidden_size))
            model.add_layer(activation)

    model.add_layer(Linear(hidden_sizes[-1], output_size))  # Output layer

    return model

def train_model(X, y, optimizer, model, epochs: int, batch_size: int):
    losses = []
    accuracies = []

    # UI
    progress_bar = st.progress(0)
    status_text = st.empty()

    model.train()
    is_classification = (st.session_state.get("data_type") == "Classification")
    ce_loss = SoftmaxCrossEntropyLoss() if is_classification else None

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0

        for X_batch, y_batch in batch_iterator(X, y, batch_size, shuffle=True):
            model.zero_grad()

            if is_classification:
                # y phải là 1-D int, zero-based
                if not np.issubdtype(y_batch.dtype, np.integer):
                    y_batch = y_batch.astype(np.int64)
                if y_batch.ndim != 1:
                    y_batch = y_batch.reshape(-1)

                # forward → logits (N, C)
                logits = model.forward(X_batch)
                assert logits.ndim == 2, f"logits must be 2D, got {logits.shape}"
                N, C = logits.shape
                max_label = int(np.max(y_batch))
                if max_label >= C:
                    raise ValueError(
                        f"Target label {max_label} out of range for logits with C={C}. "
                        f"→ Hãy đảm bảo output layer có {max_label+1} units."
                    )

                # loss & grad (đã ổn định số bên trong lớp)
                loss_value = ce_loss.forward(logits, y_batch)
                grad_logits = ce_loss.backward()

                # accuracy
                preds = np.argmax(logits, axis=1)
                epoch_correct += np.sum(preds == y_batch)
                epoch_total += len(y_batch)

                # backward
                model.backward(grad_logits)

            else:
                # Regression: đảm bảo (N, 1)
                preds = model.forward(X_batch)
                if preds.ndim == 1: 
                    preds = preds.reshape(-1, 1)
                y_b = y_batch
                if y_b.ndim == 1:
                    y_b = y_b.reshape(-1, 1)

                if preds.shape != y_b.shape:
                    raise ValueError(f"Prediction shape {preds.shape} != target shape {y_b.shape}. "
                                     f"→ Với regression, output layer nên là 1 unit và y có shape (N,1).")

                loss_value = mse_loss(preds, y_b)
                grad_preds = mse_loss_derivative(preds, y_b)

                # backward
                model.backward(grad_preds)

            # tích lũy
            epoch_loss += loss_value
            batch_count += 1

            # cập nhật tham số
            model.update_params(optimizer)   

        # log theo epoch
        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        losses.append(avg_epoch_loss)

        if is_classification:
            epoch_acc = epoch_correct / max(epoch_total, 1)
            accuracies.append(epoch_acc)
            # status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
        else:
            accuracies = None  # không có acc cho regression
            # status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")

        progress_bar.progress((epoch + 1) / epochs)

    # Set 2s to clear progress    
    time.sleep(2)
    progress_bar.empty()

    return losses, accuracies

def plot_data_2d(X, y, title="Data Visualization"):
    # Plot 2D data with different colors for each class
    fig = px.scatter(
        x= X[:, 0], y= X[:, 1], 
        color= y.astype(str),
        title= title,
        labels= {
            'x': 'Feature 1', 
            'y': 'Feature 2', 
            'color': 'Class'
        },
        color_discrete_sequence= px.colors.qualitative.Set1
    )
    fig.update_layout(
        plot_bgcolor= 'rgba(0,0,0,0)',
        paper_bgcolor= 'rgba(0,0,0,0)',
        font= dict(size=12)
    )
    return fig

def render_training_results(
    X, y, model, losses, accuracies=None, *,
    epochs: int = None,
    problem_type: str = "Classification",     # "Classification" | "Regression"
    center_charts: bool = True                # đưa biểu đồ vào cột giữa
):
    """
    Hiển thị Training Results + Visualization (Decision Boundary hoặc Regression Curve)
    - X, y: numpy arrays
    - model: có .forward(), .train(), .eval(), và các layer Linear với .weights/.bias (nếu muốn đếm params chuẩn)
    - losses: list[float] theo epoch
    - accuracies: list[float] theo epoch (hoặc None cho Regression)
    - epochs: tổng số epoch để hiển thị khi không có acc
    - problem_type: "Classification" / "Regression"
    - center_charts: True -> đặt biểu đồ vào col giữa (layout [1,2,1])
    """

    # ====== Helpers nội bộ ======
    def _count_params_from_model(m):
        total = 0
        for lyr in getattr(m, "layers", []):
            if hasattr(lyr, "weights") and lyr.weights is not None:
                total += np.prod(lyr.weights.shape)
            if hasattr(lyr, "bias") and lyr.bias is not None:
                total += np.prod(lyr.bias.shape)
        return int(total) if total > 0 else None

    def _plot_history(loss_list, acc_list):
        has_acc = acc_list is not None and len(acc_list) > 0
        if has_acc:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                y=loss_list, x=list(range(1, len(loss_list)+1)),
                mode="lines+markers", name="Loss", hovertemplate="Epoch %{x}<br>Loss=%{y:.4f}<extra></extra>"
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                y=acc_list, x=list(range(1, len(acc_list)+1)),
                mode="lines+markers", name="Accuracy", hovertemplate="Epoch %{x}<br>Acc=%{y:.4f}<extra></extra>"
            ), secondary_y=True)
            fig.update_yaxes(title_text="Loss", secondary_y=False)
            fig.update_yaxes(title_text="Accuracy", secondary_y=True, range=[0, 1])
        else:
            fig = go.Figure(go.Scatter(
                y=loss_list, x=list(range(1, len(loss_list)+1)),
                mode="lines+markers", name="Loss", hovertemplate="Epoch %{x}<br>Loss=%{y:.4f}<extra></extra>"
            ))
            fig.update_yaxes(title_text="Loss")

        fig.update_layout(
            title=dict(text="Training History", x=0.5, xanchor="center"),
            xaxis_title="Epoch",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hoverlabel=dict(bgcolor="white")
        )
        return fig

    def _plot_decision_boundary(X2d, y_cls):
        # Lưới quyết định
        h = 0.02
        x_min, x_max = X2d[:, 0].min() - 1.0, X2d[:, 0].max() + 1.0
        y_min, y_max = X2d[:, 1].min() - 1.0, X2d[:, 1].max() + 1.0
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]

        model.eval()
        Z_pred = model.forward(grid)

        # Nhận dạng binary vs multi-class
        if Z_pred.ndim == 1 or (Z_pred.ndim == 2 and Z_pred.shape[1] == 1):
            # binary 1-logit -> sigmoid threshold 0.5
            probs = 1.0 / (1.0 + np.exp(-Z_pred.reshape(-1)))
            Z = (probs >= 0.5).astype(int)
        else:
            Z = np.argmax(Z_pred, axis=1)

        Z = Z.reshape(xx.shape)

        # Vẽ
        fig = go.Figure()
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            showscale=False,
            opacity=0.35,
            line=dict(width=0),
            colorscale="Viridis",
            name="Decision Regions"
        ))

        colors = px.colors.qualitative.Set1
        classes = np.unique(y_cls)
        for i, c in enumerate(classes):
            mask = (y_cls == c)
            fig.add_trace(go.Scatter(
                x=X2d[mask, 0], y=X2d[mask, 1],
                mode="markers", name=f"Class {c}",
                marker=dict(color=colors[i % len(colors)], size=7, line=dict(width=1.5, color="white"))
            ))

        fig.update_layout(
            title=dict(text="Decision Boundary", x=0.5, xanchor="center"),
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hoverlabel=dict(bgcolor="white"),
            xaxis=dict(title="Feature 1", showgrid=True, gridcolor="rgba(0,0,0,0.08)",
                       zeroline=False, showline=True, linecolor="rgba(0,0,0,0.25)",
                       mirror=True, ticks="outside", ticklen=6,
                       scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Feature 2", showgrid=True, gridcolor="rgba(0,0,0,0.08)",
                       zeroline=False, showline=True, linecolor="rgba(0,0,0,0.25)",
                       mirror=True, ticks="outside", ticklen=6),
        )
        return fig

    def _plot_regression(X1d, y_real):
        # chuẩn hóa X thành (N,1)
        X1 = X1d.reshape(-1, 1) if X1d.ndim == 1 else X1d
        # đường mượt
        x_smooth = np.linspace(X1.min(), X1.max(), 300).reshape(-1, 1)

        model.eval()
        y_pred_smooth = model.forward(x_smooth).reshape(-1)
        y_pred_train = model.forward(X1).reshape(-1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X1.reshape(-1), y=y_real.reshape(-1),
            mode="markers", name="Training Data",
            marker=dict(size=7, opacity=0.75)
        ))
        fig.add_trace(go.Scatter(
            x=x_smooth.reshape(-1), y=y_pred_smooth,
            mode="lines", name="Model Predictions",
            line=dict(width=3)
        ))
        fig.add_trace(go.Scatter(
            x=X1.reshape(-1), y=y_pred_train,
            mode="markers", name="Train Preds",
            marker=dict(size=6, symbol="x")
        ))
        fig.update_layout(
            title=dict(text="Regression Model Predictions", x=0.5, xanchor="center"),
            xaxis_title="X", yaxis_title="y",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hoverlabel=dict(bgcolor="white"),
        )
        return fig

    # ====== UI: Header ======
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Training Results</h3>', unsafe_allow_html=True)

    # ====== Metrics ======
    c1, c2, c3 = st.columns(3)
    with c1:
        final_loss = float(losses[-1]) if (losses and len(losses) > 0) else float("nan")
        st.metric(label="Final Loss", value=f"{final_loss:.4f}" if np.isfinite(final_loss) else "N/A")

    with c2:    
        if problem_type == "Classification" and accuracies and len(accuracies) > 0:
            st.metric(label="Final Accuracy", value=f"{accuracies[-1]:.4f}")
        else:
            st.metric(label="Epochs", value=epochs if epochs is not None else len(losses))

    with c3:
        param_count = _count_params_from_model(model)
        st.metric(label="Parameters", value=f"{param_count:,}" if param_count is not None else "N/A")

    # ====== Training history (loss/acc) ======
    fig_history = _plot_history(losses, accuracies)
    
    c1, c2, c3 = st.columns([3, 0.25, 3])
    with c1: 
        st.plotly_chart(fig_history, use_container_width=True)

    with c3:
        if problem_type == "Classification":
            fig_db = _plot_decision_boundary(X, y)
            st.plotly_chart(fig_db, use_container_width=True)

        elif problem_type == "Regression":
            fig_reg = _plot_regression(X, y)
            st.plotly_chart(fig_reg, use_container_width=True)

def main():
    st.markdown("""
            <div style="text-align: center;">
                <h1>Demo Multilayer Perceptron</h1>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("<br></br>", unsafe_allow_html=True)
    
    st.markdown("""
            <div style="text-align: center;">
                <h5>Description: Built from scratch with customizable architecture and real-time visualization</h5>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("", unsafe_allow_html=True)

    st.markdown("<h3>Type of Data: </h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        data_type = st.selectbox(
            "Select Data Type",
            options=["Classification", "Regression"],
            key="data_type"
        )

    with col2:
        if st.session_state.get("data_type") == "Classification":
            problem_type = st.selectbox(
                "Select Problem Type",
                options=["Spiral", "Circle", "Line", "Zone"],
                key="problem_type"
            )
        else:
            problem_type = st.selectbox(
                "Select Problem Type",
                options=["Polynomial"],
                key="problem_type"
            )
    
    col1, space, col2, space, col3 = st.columns([3, 0.5, 3, 0.5, 3])

    with col1:
        # Network architecture
        st.markdown("<h3>Network architecture</h3>", unsafe_allow_html=True) 
        

        n_hidden_layers = st.slider("Number of hidden layers", 1, 5, 2)
        hidden_sizes = []
        for i in range(n_hidden_layers):
            size = st.slider(f"Hidden layer {i+1} size", 8, 512, 64)
            hidden_sizes.append(size)
        
        activation = st.selectbox("Activation function", ["ReLU", "Sigmoid", "Tanh"], key="activation")
        
    with col2:
        # Training parameters
        st.markdown("<h3>Training parameters</h3>", unsafe_allow_html=True)
        learning_rate = st.slider("Learning rate", 0.0001, 0.1, 0.001, 0.0001, format="%.4f", key="learning_rate")
        epochs = st.slider("Epochs", 10, 1000, 100, key="epochs")
        batch_size = st.slider("Batch size", 8, 256, 32, key="batch_size") 
        optimizer_type = st.selectbox(
            "Optimizer", 
            ["Adam", "SGD", "Momentum", "RMSProp"],
            key="optimizer_type"
        )
        
    with col3:
        # Data generation
        st.markdown('<h3>Data parameters</h3>', unsafe_allow_html=True) 

        # Classification 
        if st.session_state.get("data_type") == "Classification":
            num_classes = st.number_input(
                "Number of classes",
                min_value=2,
                max_value=5,
                value=2,
                key="num_classes",
            )
            num_samples = st.number_input(
                "Number of samples",
                min_value=20,
                max_value=512,
                value=100,
                key="num_samples"
            )

        # Regression    
        if st.session_state.get("data_type") == "Regression":
            num_samples = st.number_input(
                "Number of samples",
                min_value=20,
                max_value=512,
                value=100,
                key="num_samples",
            )

    # Generate data
    if st.session_state.get("data_type") == "Classification":
        if st.session_state.get("problem_type") == "Spiral":
            st.session_state.X, st.session_state.y = SpiralDataset(
                n_classes= st.session_state.get("num_classes"), 
                points_per_class= st.session_state.get("num_samples")).generate()
            
        elif st.session_state.get("problem_type") == "Circle":
            st.session_state.X, st.session_state.y = CircleDataset(
                n_classes= st.session_state.get("num_classes"), 
                points_per_class= st.session_state.get("num_samples")).generate()
            
        elif st.session_state.get("problem_type") == "Line":
            st.session_state.X, st.session_state.y = LineDataset(
                n_classes= st.session_state.get("num_classes"), 
                points_per_class= st.session_state.get("num_samples")).generate()
            
        elif st.session_state.get("problem_type") == "Zone":
            st.session_state.X, st.session_state.y = ZoneDataset(
                n_classes= st.session_state.get("num_classes"), 
                points_per_class= st.session_state.get("num_samples")).generate()
    else:
        st.session_state.X, st.session_state.y = PolynomialDataset(
            n_points= st.session_state.get("num_samples")).generate()

    st.markdown('<br></br>', unsafe_allow_html=True)

    # Plot data
    col1, space, col2 = st.columns([6, 1, 4])
    
    with col1:
        st.markdown('<h3>Data Visualization</h3>', unsafe_allow_html=True)
        # Fix lag ngay đây -> thêm 1 điều kiện khi thay đỏi n_classes, poin_per_class
        # dùng on_change trả về state
        if st.session_state.X.shape[1] == 2:  # 2D data
            fig_data = plot_data_2d(
                st.session_state.X, 
                st.session_state.y, 
                f"{problem_type} Data")
            
            st.plotly_chart(fig_data, use_container_width=True)

        else:  # 1D regression data
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x= st.session_state.X.flatten(), 
                y= st.session_state.y, 
                mode='markers', name='Data'))
            
            fig.update_layout(
                title="Regression Data",
                xaxis_title="X",
                yaxis_title="y",
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3>Network Architecture</h3>', unsafe_allow_html=True)
        st.markdown('<br></br>', unsafe_allow_html=True)

        # Display network architecture
        arch_text = f"Input ({st.session_state.X.shape[1]})"
        for i, size in enumerate(hidden_sizes):
            arch_text += f" → Hidden {i+1} ({size})"
        
        if st.session_state.get("data_type") == "Classification":
            output_size = len(np.unique(st.session_state.y))
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

    st.markdown("", unsafe_allow_html=True)

    state_mode = {
        "model": None,
        "losses": [],
        "accuracies": [],
        "training_complete": False,
    }

    for k, v in state_mode.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Training section
    space, col1, space = st.columns([1.5,1,1.5])
    with col1:
        if st.button("Start Training", type="primary", width="stretch"):
            with st.spinner("Training in progress..."):
                model = create_model(
                    input_size= st.session_state.X.shape[1],
                    hidden_sizes= hidden_sizes,
                    output_size= output_size,
                    activation= activation
                )
                optimizer_type = Optimizer(
                    optimizer_type= st.session_state.get("optimizer_type"),
                    learning_rate= st.session_state.get("learning_rate")
                )
                losses, accuracies = train_model(
                    X= st.session_state.X, 
                    y= st.session_state.y,
                    optimizer= optimizer_type,
                    model= model,
                    epochs= st.session_state.get("epochs"),
                    batch_size= st.session_state.get("batch_size")
                )

                # Lưu kết quả vào session_state
                st.session_state.model = model
                st.session_state.losses = losses or []

                # Regression có thể trả None
                st.session_state.accuracies = (accuracies or []) if st.session_state.get("data_type") == "Classification" else []
                st.session_state.training_complete = True

            messages = st.empty()
            messages.success("Training completed!")
            time.sleep(2)
            messages.empty()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.training_complete and st.session_state.model is not None and len(st.session_state.losses) > 0:
        render_training_results(
            X = st.session_state.X,
            y = st.session_state.y,
            model = st.session_state.model,
            losses = st.session_state.losses,
            accuracies = (st.session_state.accuracies if st.session_state.get("data_type") == "Classification" else None),
            epochs = st.session_state.get("epochs"),
            problem_type = st.session_state.get("data_type"),   # "Classification" | "Regression"
            center_charts = True
        )
    else:
        st.info("Click **Start Training** to begin.")

if __name__ == "__main__":
    main()
    


