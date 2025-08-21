import time
import sys
import numpy as np
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go   
import plotly.express as px
from plotly.subplots import make_subplots
from tensorflow.keras.datasets import mnist, fashion_mnist
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
            # "CIFAR-10": self.load_cifar10,
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
    
    # def load_cifar10(self, subset_size=None):
    #     (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #     X_train = (X_train.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    #     X_test  = (X_test.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    #     y_train = y_train.flatten(); y_test = y_test.flatten()
    #     if subset_size:
    #         idx = np.random.choice(len(X_train), subset_size, replace=False)
    #         X_train, y_train = X_train[idx], y_train[idx]
    #         idx = np.random.choice(len(X_test), max(1, subset_size//5), replace=False)
    #         X_test, y_test = X_test[idx], y_test[idx]
    #     return (X_train, y_train), (X_test, y_test), 10, (3, 32, 32)
    
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

def visualize_sample_images(
    X, y, n_samples: int = 16, dataset_name: str = "Dataset",
    model=None, show_predict: bool = False
):
    """
    Hiển thị n ảnh mẫu theo lưới (4 cột). Tự xử lý shape:
    - Ảnh (C,H,W) → hiển thị (H,W,C)
    - Ảnh xám (1,H,W) → hiển thị (H,W) kèm cmap='gray'
    - Khi show_predict=True: tự thêm batch dim và suy luận 1 ảnh/lần.
    """
    class_names = get_class_names(dataset_name)

    # ----- chọn mẫu an toàn -----
    n_total = len(X)
    n_samples = int(n_samples)
    if n_samples <= 0:
        n_samples = 1
    replace = n_samples > n_total
    idxs = np.random.choice(n_total, size=n_samples, replace=replace)

    # ----- layout -----
    ncols = 4
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])  # khi chỉ có 1 subplot

    def _to_hwc(img):
        """Trả về (img_display, cmap). Nhận (H,W), (H,W,C) hoặc (C,H,W)."""
        if img.ndim == 2:                     # (H, W)
            return img, "gray"
        if img.ndim == 3:
            # Channel-first phổ biến: (1,H,W) hoặc (3,H,W)
            if img.shape[0] in (1, 3):
                c, h, w = img.shape
                if c == 1:
                    return img[0], "gray"
                return np.transpose(img, (1, 2, 0)), None  # (H,W,C)
            # Channel-last: (H,W,C)
            return img, (None if img.shape[-1] != 1 else "gray")
        # Trường hợp khác: cố gắng squeeze
        img2 = np.squeeze(img)
        return _to_hwc(img2)

    def _scalar_label(lbl):
        arr = np.array(lbl).reshape(-1)
        return int(arr[0])

    for i, idx in enumerate(idxs):
        ax = axes[i]
        img = X[idx]
        true_lbl = _scalar_label(y[idx])

        # ----- predict (tuỳ chọn) -----
        pred_lbl = None
        if show_predict and model is not None:
            model.eval()
            x_in = img[None, ...]                 # (1, C, H, W) hoặc (1, H, W, C)
            logits = np.asarray(model.forward(x_in))
            # Chuẩn hoá về (1, C)
            if logits.ndim == 1:                 # (C,)
                logits = logits[None, :]
            elif logits.ndim > 2:                # fallback: flatten chiều còn lại
                logits = logits.reshape(logits.shape[0], -1)
            pred_lbl = int(np.argmax(logits, axis=1)[0])

        # ----- hiển thị -----
        img_disp, cmap = _to_hwc(img)
        ax.imshow(img_disp, cmap=cmap)
        true_text = class_names[true_lbl] if class_names else true_lbl
        if pred_lbl is not None:
            pred_text = class_names[pred_lbl] if class_names else pred_lbl
            ax.set_title(f"Pred: {pred_text} | True: {true_text}", fontsize=10)
        else:
            ax.set_title(f"Class: {true_text}", fontsize=10)
        ax.axis("off")

    # Ẩn subplot thừa
    for j in range(n_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
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
        lines.append(f"Conv2D {i+1}: {cur[0]}→{c['out_channels']} (k={c['kernel_size']}, pad={c['padding']})")
        h = calculate_conv_output_size(cur[1], c['kernel_size'], 1, c['padding'])
        w = calculate_conv_output_size(cur[2], c['kernel_size'], 1, c['padding'])
        cur = (c['out_channels'], h, w); lines.append(f"ReLU: {cur}")
        if c.get('use_max_pooling', True):
            p = c.get('pooling_size', 2); lines.append(f"MaxPool2D: {cur}→({cur[0]}, {cur[1]//p}, {cur[2]//p})")
            cur = (cur[0], cur[1]//p, cur[2]//p)
        if c.get('use_batch_norm', False): lines.append(f"BatchNorm2D: {cur}")
        if c.get('dropout_rate', 0)>0: lines.append(f"Dropout({c['dropout_rate']}): {cur}")
    flat = cur[0]*cur[1]*cur[2]; lines.append(f"Flatten: {cur}→({flat},)")
    prev = flat
    for i, f in enumerate(fc_cfgs[:-1]): lines += [f"Linear {i+1}: {prev}→{f}", f"ReLU: ({f},)"]; prev = f
    lines.append(f"Output: {prev}→{fc_cfgs[-1]}"); 
    return lines

# ===================== Training =====================
def train_cnn_model(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, epochs, batch_size):
    train_losses, train_accs, val_accs = [], [], []
    progress_bar, status_text, loss_chart = st.progress(0), st.empty(), st.empty()
    rate = 0.0
    batch_r = batch_iterator(X_train, y_train, batch_size, shuffle=True)
    total_len_batch = epochs * batch_r.len_batch() 
    model.train()
    for epoch in range(epochs):
        ep_loss = 0.0; correct = 0; total = 0; batches = 0
        for Xb, yb in batch_r:
            model.zero_grad()
            logits = model.forward(Xb)
            loss = loss_fn.forward(logits, yb)
            grad = loss_fn.backward()
            model.backward(grad)
            model.update_params(optimizer)
            ep_loss += loss; pred = np.argmax(logits, 1); correct += np.sum(pred == yb); total += len(yb); batches += 1
            rate += 1
            progress_bar.progress(rate / total_len_batch)
        avg_loss = ep_loss / max(batches, 1); train_acc = correct / max(total, 1)
        train_losses.append(avg_loss); train_accs.append(train_acc)

        # validation
        model.eval(); v_correct = 0; v_total = 0
        for Xb, yb in batch_iterator(X_val, y_val, batch_size, shuffle=False):
            logits = model.forward(Xb); v_correct += np.sum(np.argmax(logits,1) == yb); v_total += len(yb)
        val_acc = v_correct / max(v_total, 1); val_accs.append(val_acc); model.train()

            
        status_text.text(f"Epoch {epoch+1}/{epochs} | Loss {avg_loss:.4f} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}") 
        time.sleep(0.01)
    progress_bar.empty(); status_text.empty(); return train_losses, train_accs, val_accs

# ===================== App =====================
st.session_state.setdefault("run_id", 0)
def main():
    data_loader = DatasetLoader()

    st.markdown("""
            <div style="text-align: center;">
                <h1>CNN Demo</h1>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("<br></br>", unsafe_allow_html=True)

    st.markdown("""
            <div style="text-align: center;">
                <h5>Description: Convolutional Neural Network built from scratch with real-time training visualization</h5>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("", unsafe_allow_html=True)


    col1, sp1, col2, sp2, col3 = st.columns([3, .5, 3, .5, 3])
    with col1:
        st.markdown('<h3>Dataset Configuration</h3>', unsafe_allow_html=True)
        dataset_name = st.selectbox("Select Dataset", ["MNIST", "Fashion-MNIST"])
        subset_size = st.slider("Training samples", 100, 50000, 5000, step=100)
        train_val_split = st.slider("Train split", 0.0, 1.0, 0.8)
    with col2:
        st.markdown('<h3>CNN Architecture</h3>', unsafe_allow_html=True)
        n_conv_layers = st.slider("Number of conv layers", 1, 4, 2)
        conv_configs = []
        for i in range(n_conv_layers):
            with st.expander(f"Conv Layer {i+1}"):
                # Input channel 
                if i == 0: 
                    in_channel = 1
                else:
                    in_channel = conv_configs[i-1]['out_channels']

                # Kernel size
                kernel_size = st.selectbox("Kernel size", [3, 5], key= f"conv_{i}_k")

                # Padding 
                padding = st.selectbox("Padding", [0, 1, 2], key= f"conv_{i}_pad")

                # Max pooling 
                use_max_pooling = st.checkbox("Use max pooling", value= True, key= f"conv_{i}_pool")
                pooling_size = 2 if use_max_pooling else 1

                # Batch normalization
                use_batch_norm = st.checkbox("Use batch norm", value= True, key= f"conv_{i}_bn")

                # Dropout
                dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.0, key= f"conv_{i}_d")
                
                # Output channel layers slider 
                output_channels = st.slider("Output channels", 8, 128, 16, key= f"conv_{i}_out")

                # Add attribute in convolutional configuration
                conv_configs.append({
                    "in_channels": in_channel,
                    "out_channels": output_channels,
                    "kernel_size": kernel_size, 
                    "padding": padding, 
                    "use_max_pooling": use_max_pooling,
                    "pooling_size": pooling_size,
                    "use_batch_norm": use_batch_norm,
                    "dropout_rate": dropout_rate, 
                })

        # FC in last layers
        num_fc_layers = st.slider("Number of Fully Connected layers", 1, 5, 2)
        # Save layers atributte in FC configuation
        fc_configs = []
        for i in range(num_fc_layers-1):
            fc_configs.append(st.slider(f"FC layer {i+1} size", 32, 512, 32))
        # last layers append 10 classes 
        fc_configs.append(10)  # 10 classes

    with col3:
        st.markdown('<h3>Training Parameters</h3>', unsafe_allow_html=True)
        lr = st.slider("Learning rate", 0.0001, 0.01, 0.001, format="%.3f")
        epochs = st.slider("Epochs", 1, 10, 5)
        batch_size = st.slider("Batch size", 16, 2048, 1024)
        opt_name = st.selectbox("Optimizer", ["Adam", "SGD", "Momentum", "RMSProp"])

    # Load dataset
    c1, c2, c3 = st.columns([1.5, 1, 1.5])
    with c2:
        if st.button("Load Dataset", type="primary", use_container_width=True):
            if 'dataset_loaded' in st.session_state:
                del st.session_state['dataset_loaded']
        if 'dataset_loaded' not in st.session_state:
            with st.spinner(f"Loading {dataset_name} dataset..."):
                (X_tr, y_tr), (X_te, y_te), n_classes, input_shape = data_loader.datasets[dataset_name](subset_size)
                X_tr, Xval, y_tr, yval = train_test_split(X_tr, y_tr, train_size=train_val_split)
                st.session_state.X_train, st.session_state.y_train = X_tr, y_tr
                st.session_state.X_val,   st.session_state.y_val   = Xval, yval
                st.session_state.X_test,  st.session_state.y_test  = X_te, y_te
                st.session_state.input_shape = input_shape
                st.session_state.dataset_name = dataset_name
                st.session_state.dataset_loaded = True

    if 'dataset_loaded' in st.session_state:
        X_train, y_train = st.session_state.X_train, st.session_state.y_train
        X_val, y_val     = st.session_state.X_val,   st.session_state.y_val
        X_test, y_test   = st.session_state.X_test,  st.session_state.y_test
        input_shape      = st.session_state.input_shape
        dataset_name     = st.session_state.dataset_name

        st.markdown('<h3>Model Architecture</h3>', unsafe_allow_html=True)
        arch_text = get_model_architecture_text(conv_configs, fc_configs, input_shape)
        with st.expander("Show model architecture"):
            for block in arch_text:
                st.code(block)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, sp, c2 = st.columns([6, 0.5, 4])
        with c1:
            st.markdown('<h3>Dataset Overview</h3>', unsafe_allow_html=True)

            # Plot the bar chart to check distribution
            # Counting unique value
            unique, cnt = np.unique(y_train, return_counts= True)
            st.plotly_chart(
                px.bar(
                    x= [str(i) for i in unique],
                    y= cnt,
                    title= "Class distribution",
                    labels= {
                        'x': 'Class',
                        'y': 'Count',
                    }
                ), use_container_width= True, key= "bar_chart"
            )

            st.write(f"**Training**: {len(X_train)} | **Validation**: {len(X_val)} | **Test**: {len(X_test)}")
            st.write(f"**Input shape**: {input_shape} | **Classes**: {len(np.unique(y_train))}")

        with c2:
            st.markdown('<h3>Sample Images</h3>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html= True)

            st.pyplot(visualize_sample_images(X_train, y_train, 16, dataset_name))

        b1, b2, b3 = st.columns([1.5, 1, 1.5])
        with b2:
            if st.button("Start Training", type="primary", use_container_width=True):
                st.session_state.run_id += 1

                with st.spinner("Building and training CNN..."):
                    # Build model
                    model = CNN()
                    for cfg in conv_configs:
                        model.add_layer(
                            Conv2D(
                                cfg['in_channels'],
                                cfg['out_channels'], 
                                cfg['kernel_size'], 
                                padding=cfg['padding']
                        ))
                        model.add_layer(ReLULayer())
                        if cfg['use_max_pooling']:   
                            model.add_layer(MaxPool2D(pool_size=cfg['pooling_size']))
                        if cfg['use_batch_norm']: 
                            model.add_layer(BatchNorm2D(cfg['out_channels']))
                        if cfg['dropout_rate']>0: 
                            model.add_layer(Dropout(cfg['dropout_rate']))
                    model.add_layer(Flatten())

                    # compute flatten size
                    cur = input_shape
                    for cfg in conv_configs:
                        h = calculate_conv_output_size(
                            cur[1], 
                            cfg['kernel_size'], 
                            1, 
                            cfg['padding']
                        )
                        w = calculate_conv_output_size(
                            cur[2], 
                            cfg['kernel_size'], 
                            1, 
                            cfg['padding']
                        )
                        cur = (cfg['out_channels'], h, w)
                        if cfg['use_max_pooling']: cur = (cur[0], cur[1]//cfg['pooling_size'], cur[2]//cfg['pooling_size'])
                    flat = cur[0]*cur[1]*cur[2]

                    prev = flat
                    for f in fc_configs[:-1]:
                        model.add_layer(Linear(prev, f)); model.add_layer(ReLULayer()); prev = f
                    model.add_layer(Linear(prev, fc_configs[-1]))  # logits

                    if "model" not in st.session_state:
                        st.session_state.model = model

                    # optimizer
                    if opt_name == "Adam": 
                        optimizer = Adam(lr)
                    elif opt_name == "SGD": 
                        optimizer = SGD(lr)
                    elif opt_name == "Momentum": 
                        optimizer = Momentum(lr)
                    else: 
                        optimizer = RMSProp(lr)

                    if "optimizer" not in st.session_state:
                        st.session_state.optimizer = optimizer

                    ms = st.empty()
                    loss_fn = SoftmaxCrossEntropyLoss()
                    ms.success("Model created successfully! Starting training...")
                    ms.empty()

                    if "loss_fn" not in st.session_state:
                        st.session_state.loss_fn = loss_fn

        model = st.session_state.get('model', None)
        optimizer = st.session_state.get('optimizer', None)
        loss_fn = st.session_state.get('loss_fn', None)
        
        if model is not None:
            st.markdown('<h3>Training Progress</h3>', unsafe_allow_html=True)
            train_losses, train_accs, val_accs = train_cnn_model(
                model, optimizer, loss_fn, X_train, y_train, X_val, y_val, epochs, batch_size
            )


            # Final charts
            st.plotly_chart(plot_training_history(train_losses, train_accs, val_accs), use_container_width=True, key="loss_final")

            # Evaluate on test set
            model.eval(); t_correct = 0; t_total = 0
            for Xb, yb in batch_iterator(X_test, y_test, batch_size, shuffle=False):
                logits = model.forward(Xb); t_correct += np.sum(np.argmax(logits, 1) == yb); t_total += len(yb)
            test_acc = t_correct / max(t_total, 1)
            st.success(f"Test Accuracy: **{test_acc:.3f}**")
            st.pyplot(visualize_sample_images(
                X= X_test, y= y_test, n_samples= 16, dataset_name= dataset_name, model= model, show_predict= True,
            ))

if __name__ == "__main__":
    main()