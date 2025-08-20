import time
import sys
import numpy as np
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go   
import plotly.express as px
from plotly.subplots import make_subplots
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist
import matplotlib.pyplot as plt

root_path = Path.cwd().parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from NeuralKit import *
from theme import *

st.set_page_config(
    page_title= "Convolutional Neural Network Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_css(), unsafe_allow_html=True)

class DatasetLoader:
    def __init__(self):
        self.datasets = {
            "MNIST": self.load_mnist,
            "CIFAR-10": self.load_cifar10,
            "Fashion-MNIST": self.load_fashion_mnist
        }

    def load_mnist(self, subset_size=None):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.astype(np.float32) / 255.0
        X_test  = X_test.astype(np.float32) / 255.0
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test  = X_test.reshape(-1, 1, 28, 28)
        if subset_size:
            idx = np.random.choice(len(X_train), subset_size, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
            idx = np.random.choice(len(X_test), max(1, subset_size//5), replace=False)
            X_test, y_test = X_test[idx], y_test[idx]
        return (X_train, y_train), (X_test, y_test), 10, (1, 28, 28)
    
    def load_cifar10(self, subset_size=None):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = (X_train.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        X_test  = (X_test.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        y_train = y_train.flatten(); y_test = y_test.flatten()
        if subset_size:
            idx = np.random.choice(len(X_train), subset_size, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
            idx = np.random.choice(len(X_test), max(1, subset_size//5), replace=False)
            X_test, y_test = X_test[idx], y_test[idx]
        return (X_train, y_train), (X_test, y_test), 10, (3, 32, 32)
    
    def load_fashion_mnist(self, subset_size=None):
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train = X_train.astype(np.float32) / 255.0
        X_test  = X_test.astype(np.float32) / 255.0
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test  = X_test.reshape(-1, 1, 28, 28)
        if subset_size:
            idx = np.random.choice(len(X_train), subset_size, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
            idx = np.random.choice(len(X_test), max(1, subset_size//5), replace=False)
            X_test, y_test = X_test[idx], y_test[idx]
        return (X_train, y_train), (X_test, y_test), 10, (1, 28, 28)

def visualize_sample_images(X, y, n_samples=8, dataset_name="Dataset"):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6)); axes = axes.ravel()
    names = get_class_names(dataset_name); idxs = np.random.choice(len(X), n_samples, replace=False)
    for i, idx in enumerate(idxs):
        img, label = X[idx], y[idx]
        if img.shape[0] == 1: img_disp, cmap = img[0], 'gray'
        else: img_disp, cmap = img.transpose(1, 2, 0), None
        axes[i].imshow(img_disp, cmap=cmap); axes[i].set_title(f'Class: {names[label] if names else label}'); axes[i].axis('off')
    plt.tight_layout(); 
    return fig

def get_class_names(dataset_name):
    if dataset_name == "CIFAR-10":
        return ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    if dataset_name == "Fashion-MNIST":
        return ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    if dataset_name == "MNIST":
        return [str(i) for i in range(10)]
    return None

def plot_training_history(losses, train_accs=None, val_accs=None):
    epochs = list(range(1, len(losses)+1))
    if train_accs is not None and val_accs is not None:
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Training Loss', 'Accuracy'))
        fig.add_trace(go.Scatter(x=epochs, y=losses, name='Loss'), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=train_accs, name='Train Acc'), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=val_accs, name='Val Acc'), row=1, col=2)
    else:
        fig = go.Figure(go.Scatter(x=epochs, y=losses, name='Loss'))
        fig.update_layout(title='Training Loss', xaxis_title='Epoch', yaxis_title='Loss')
    fig.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
    return fig

def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    return (input_size + 2*padding - kernel_size)//stride + 1

def get_model_architecture_text(conv_cfgs, fc_cfgs, input_shape):
    lines, cur = [], input_shape
    lines.append(f"Input: {cur}")
    for i, c in enumerate(conv_cfgs):
        lines.append(f"Conv2D-{i+1}: {cur[0]}→{c['out_channels']} (k={c['kernel_size']}, pad={c['padding']})")
        h = calculate_conv_output_size(cur[1], c['kernel_size'], 1, c['padding'])
        w = calculate_conv_output_size(cur[2], c['kernel_size'], 1, c['padding'])
        cur = (c['out_channels'], h, w); lines.append(f"ReLU: {cur}")
        if c.get('use_pooling', True):
            p = c.get('pool_size', 2); lines.append(f"MaxPool2D: {cur}→({cur[0]}, {cur[1]//p}, {cur[2]//p})")
            cur = (cur[0], cur[1]//p, cur[2]//p)
        if c.get('use_batchnorm', False): lines.append(f"BatchNorm2D: {cur}")
        if c.get('dropout_rate', 0)>0: lines.append(f"Dropout({c['dropout_rate']}): {cur}")
    flat = cur[0]*cur[1]*cur[2]; lines.append(f"Flatten: {cur}→({flat},)")
    prev = flat
    for i, f in enumerate(fc_cfgs[:-1]): lines += [f"Linear-{i+1}: {prev}→{f}", f"ReLU: ({f},)"]; prev = f
    lines.append(f"Output: {prev}→{fc_cfgs[-1]}"); return lines

# ===================== Training =====================
def train_cnn_model(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, epochs, batch_size):
    train_losses, train_accs, val_accs = [], [], []
    progress_bar, status_text, loss_chart = st.progress(0), st.empty(), st.empty()
    model.train()
    for epoch in range(epochs):
        ep_loss = 0.0; correct = 0; total = 0; batches = 0
        for Xb, yb in batch_iterator(X_train, y_train, batch_size, shuffle=True):
            model.zero_grad()
            logits = model.forward(Xb)
            loss = loss_fn.forward(logits, yb)
            grad = loss_fn.backward()
            model.backward(grad)
            model.update_params(optimizer)
            ep_loss += loss; pred = np.argmax(logits, 1); correct += np.sum(pred == yb); total += len(yb); batches += 1
        avg_loss = ep_loss / max(batches, 1); train_acc = correct / max(total, 1)
        train_losses.append(avg_loss); train_accs.append(train_acc)

        # validation
        model.eval(); v_correct = 0; v_total = 0
        for Xb, yb in batch_iterator(X_val, y_val, batch_size, shuffle=False):
            logits = model.forward(Xb); v_correct += np.sum(np.argmax(logits,1) == yb); v_total += len(yb)
        val_acc = v_correct / max(v_total, 1); val_accs.append(val_acc); model.train()

        progress_bar.progress((epoch+1)/epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} | Loss {avg_loss:.4f} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")
        if epoch % 5 == 0 or epoch == epochs-1:
            loss_chart.plotly_chart(plot_training_history(train_losses, train_accs, val_accs), use_container_width=True)
        time.sleep(0.01)
    progress_bar.empty(); status_text.empty(); return train_losses, train_accs, val_accs

# ===================== App =====================
def main():
    data_loader = DatasetLoader()

    st.markdown('<h1 class="main-header">CNN Demo</h1>', unsafe_allow_html=True)
    st.markdown("<div style='text-align:center'><h5>Convolutional Neural Network built from scratch with real-time training visualization</h5></div>", unsafe_allow_html=True)
    st.markdown("")

    col1, sp1, col2, sp2, col3 = st.columns([3, .5, 3, .5, 3])
    with col1:
        st.markdown('<h3 class="sub-header">Dataset Configuration</h3>', unsafe_allow_html=True)
        dataset_name = st.selectbox("Select Dataset", ["MNIST", "CIFAR-10", "Fashion-MNIST"])
        subset_size = st.slider("Training samples (for faster training)", 100, 10000, 2000, step=100)
        train_val_split = st.slider("Train/Validation split", 0.1, 0.3, 0.2)
    with col2:
        st.markdown('<h3 class="sub-header">CNN Architecture</h3>', unsafe_allow_html=True)
        n_conv_layers = st.slider("Number of conv layers", 1, 4, 2)
        conv_configs = []
        for i in range(n_conv_layers):
            with st.expander(f"Conv Layer {i+1}", expanded=(i==0)):
                in_ch = 3 if (i==0 and dataset_name=="CIFAR-10") else (1 if i==0 else conv_configs[i-1]['out_channels'])
                out_ch = st.slider("Output channels", 8, 128, 16*(2**i), key=f"conv_{i}_out")
                ksize  = st.selectbox("Kernel size", [3,5], key=f"conv_{i}_k")
                pad    = st.selectbox("Padding", [0,1,2], index=1, key=f"conv_{i}_pad")
                use_pool = st.checkbox("Use MaxPool", value=True, key=f"conv_{i}_pool")
                pool_sz  = 2 if use_pool else 1
                use_bn   = st.checkbox("Use BatchNorm", value=False, key=f"conv_{i}_bn")
                drop     = st.slider("Dropout rate", 0.0, 0.5, 0.0, key=f"conv_{i}_drop")
                conv_configs.append({
                    'in_channels': in_ch, 'out_channels': out_ch,
                    'kernel_size': ksize, 'padding': pad,
                    'use_pooling': use_pool, 'pool_size': pool_sz,
                    'use_batchnorm': use_bn, 'dropout_rate': drop
                })
        n_fc_layers = st.slider("Number of FC layers", 1, 3, 2)
        fc_configs = []
        for i in range(n_fc_layers-1):
            fc_configs.append(st.slider(f"FC layer {i+1} size", 32, 512, 128, key=f"fc_{i}"))
        fc_configs.append(10)  # 10 classes
    with col3:
        st.markdown('<h3 class="sub-header">Training Parameters</h3>', unsafe_allow_html=True)
        opt_name = st.selectbox("Optimizer", ["Adam", "SGD", "Momentum", "RMSProp"])
        lr = st.slider("Learning rate", 0.0001, 0.01, 0.001, format="%.4f")
        epochs = st.slider("Epochs", 1, 50, 5)
        batch_size = st.slider("Batch size", 16, 128, 32)
        st.markdown("### Model Summary")
        st.info(f"Dataset: {dataset_name}\nTraining samples: {subset_size}\nConv layers: {n_conv_layers}\nFC layers: {n_fc_layers}")

    # Load dataset
    c1, c2, c3 = st.columns([1.5, 1, 1.5])
    with c2:
        if st.button("Load Dataset", type="primary", use_container_width=True):
            if 'dataset_loaded' in st.session_state: del st.session_state['dataset_loaded']
        if 'dataset_loaded' not in st.session_state:
            with st.spinner(f"Loading {dataset_name} dataset..."):
                (Xtr, ytr), (Xte, yte), n_classes, input_shape = data_loader.datasets[dataset_name](subset_size)
                Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=train_val_split)
                st.session_state.X_train, st.session_state.y_train = Xtr, ytr
                st.session_state.X_val,   st.session_state.y_val   = Xval, yval
                st.session_state.X_test,  st.session_state.y_test  = Xte, yte
                st.session_state.input_shape = input_shape
                st.session_state.dataset_name = dataset_name
                st.session_state.dataset_loaded = True

    if 'dataset_loaded' in st.session_state:
        X_train, y_train = st.session_state.X_train, st.session_state.y_train
        X_val, y_val     = st.session_state.X_val,   st.session_state.y_val
        X_test, y_test   = st.session_state.X_test,  st.session_state.y_test
        input_shape      = st.session_state.input_shape
        dataset_name     = st.session_state.dataset_name

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<h3 class="sub-header">Dataset Overview</h3>', unsafe_allow_html=True)
            st.write(f"**Training**: {len(X_train)} | **Validation**: {len(X_val)} | **Test**: {len(X_test)}")
            st.write(f"**Input shape**: {input_shape} | **Classes**: {len(np.unique(y_train))}")
            u, cnt = np.unique(y_train, return_counts=True)
            st.plotly_chart(px.bar(x=[str(i) for i in u], y=cnt, title="Class Distribution",
                                   labels={'x':'Class','y':'Count'}, template="plotly_white"),
                            use_container_width=True)
        with c2:
            st.markdown('<h3 class="sub-header">Sample Images</h3>', unsafe_allow_html=True)
            st.pyplot(visualize_sample_images(X_train, y_train, 8, dataset_name))

        st.markdown('<h3 class="sub-header">Model Architecture</h3>', unsafe_allow_html=True)
        arch_text = get_model_architecture_text(conv_configs, fc_configs, input_shape)
        a1, a2 = st.columns(2)
        with a1: st.code("\n".join(arch_text[:len(arch_text)//2]))
        with a2: st.code("\n".join(arch_text[len(arch_text)//2:]))

        b1, b2, b3 = st.columns([1.5, 1, 1.5])
        with b2:
            if st.button("Start Training", type="primary", use_container_width=True):
                with st.spinner("Building and training CNN..."):
                    # Build model
                    model = CNN()
                    for cfg in conv_configs:
                        model.add_layer(Conv2D(cfg['in_channels'], cfg['out_channels'], cfg['kernel_size'], padding=cfg['padding']))
                        model.add_layer(ReLULayer())
                        if cfg['use_pooling']:   model.add_layer(MaxPool2D(pool_size=cfg['pool_size']))
                        if cfg['use_batchnorm']: model.add_layer(BatchNorm2D(cfg['out_channels']))
                        if cfg['dropout_rate']>0: model.add_layer(Dropout(cfg['dropout_rate']))
                    model.add_layer(Flatten())

                    # compute flatten size
                    cur = input_shape
                    for cfg in conv_configs:
                        h = calculate_conv_output_size(cur[1], cfg['kernel_size'], 1, cfg['padding'])
                        w = calculate_conv_output_size(cur[2], cfg['kernel_size'], 1, cfg['padding'])
                        cur = (cfg['out_channels'], h, w)
                        if cfg['use_pooling']: cur = (cur[0], cur[1]//cfg['pool_size'], cur[2]//cfg['pool_size'])
                    flat = cur[0]*cur[1]*cur[2]

                    prev = flat
                    for f in fc_configs[:-1]:
                        model.add_layer(Linear(prev, f)); model.add_layer(ReLULayer()); prev = f
                    model.add_layer(Linear(prev, fc_configs[-1]))  # logits

                    # Optimizer
                    if opt_name == "Adam": optimizer = Adam(lr)
                    elif opt_name == "SGD": optimizer = SGD(lr)
                    elif opt_name == "Momentum": optimizer = Momentum(lr)
                    else: optimizer = RMSProp(lr)

                    loss_fn = SoftmaxCrossEntropyLoss()
                    st.success("Model created successfully! Starting training...")

                st.markdown('<h3 class="sub-header">Training Progress</h3>', unsafe_allow_html=True)
                train_losses, train_accs, val_accs = train_cnn_model(
                    model, optimizer, loss_fn, X_train, y_train, X_val, y_val, epochs, batch_size
                )

                # Final charts
                st.plotly_chart(plot_training_history(train_losses, train_accs, val_accs), use_container_width=True)

                # Evaluate on test set
                model.eval(); t_correct = 0; t_total = 0
                for Xb, yb in batch_iterator(X_test, y_test, batch_size, shuffle=False):
                    logits = model.forward(Xb); t_correct += np.sum(np.argmax(logits, 1) == yb); t_total += len(yb)
                test_acc = t_correct / max(t_total, 1)
                st.success(f"Test Accuracy: **{test_acc:.3f}**")

if __name__ == "__main__":
    main()