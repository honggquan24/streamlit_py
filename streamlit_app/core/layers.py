import numpy as np
from .activations import *
from .loss import *

class Layer:
    # Base class for layers

    def forward(self, x):
        raise NotImplementedError  # Must override in subclass

    def backward(self, grad_output):
        raise NotImplementedError  # Must override in subclass

    def update_params(self, optimizer):
        pass  # Only param layers (e.g. Dense) need to override

    def zero_grad(self):
        pass  # Only layers with grads need to override

class Linear(Layer):
    # Dense / Fully connected layer

    def __init__(self, input_size, output_size, init_type= 'xavier', bias= True):
        self.input_size =input_size
        self.output_size = output_size
        self.bias_flag = bias

        # Weight initialization
        if init_type.lower() == 'xavier':
            bound = np.sqrt(6 / (self.input_size + self.output_size))
        elif init_type.lower() == 'he':
            bound = np.sqrt(2 / self.input_size)
        else:  # default: small random uniform
            bound = 0.01
        self.weights = np.random.uniform(-bound, bound, (self.input_size, self.output_size))

        if bias:
            self.bias = np.zeros(output_size)
        else:
            self.bias = None

        # Gradients
        self.weights_grad = np.zeros_like(self.weights)
        if bias:
            self.bias_grad = np.zeros_like(self.bias)

        # Cache for backward pass
        self.input_cache = None
    
    def forward(self, x):
        # Foward pass
        self.input_cache = x.copy() 

        output = np.dot(x, self.weights)
        if self.bias_flag:
            output += self.bias
        return output
    
    def backward(self, grad_output):
        # Gradient w.r.t. weights: dL / dW = X^T * grad_output
        self.weights_grad += np.dot(self.input_cache.T, grad_output)

        # Gradient w.r.t bias: dl / db = sum(grad_output)
        if self.bias_flag:
            self.bias_grad += np.sum(grad_output, axis= 0)

        # Gradient w.r.t input: dl / dX = grad_output * W^T
        grad_input = np.dot(grad_output, self.weights.T)

        return grad_input
    
    def update_params(self, optimizer):
        # Update parameters by optimizer
        optimizer.update(self.weights, self.weights_grad)
        if self.bias_flag:
            optimizer.update(self.bias, self.bias_grad)

    def zero_grad(self):
        self.weights_grad = np.zeros_like(self.weights_grad)
        if self.bias_flag:
            self.bias_grad = np.zeros_like(self.bias_grad)

class Activation(Layer):
    def __init__(self, activation_function, derivative_function):
        self.activation_function = activation_function
        self.derivative_function = derivative_function
        self.input_cache = None

    def forward(self, x):
        self.input_cache = x.copy()
        return self.activation_function(x)
    
    def backward(self, grad_output):
        return grad_output * self.derivative_function(self.input_cache)
    
class ReLULayer(Activation):
    def __init__(self):
        super().__init__(relu, relu_derivative)

class SigmoidLayer(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_derivative)

class TanhLayer(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_derivative)

class LeakyReLULayer(Activation):
    def __init__(self, alpha= 0.01):
        super().__init__(
            lambda x: leaky_relu(x, alpha),
            lambda x: leaky_relu_derivative(x, alpha)
        )

class SoftmaxLayer(Activation):
    def __init__(self):
        self.output_cache = None

    def forward(self, x):
        output = softmax(x)
        self.output_cache = output.copy()
        return output

    def backward(self, grad_output):
        # Standard softmax gradient for general case
        # Note: When used with CrossEntropy, this may be redundant
        # as CrossEntropy backward already includes softmax derivative
        # s = softmax(z) 
        # ds_i / dz_j: s_i * (1 - s_i) if i = j 
        #            :-s_i * s_j       if i >< j 
        # Or write in Jacobian 
        # J_ij = s_i * (δ_ij - s_j) = diag(s) - s sᵀ
        # dL / dz = Jᵀ · dL / dŷ

        s = self.output_cache
        len_sample = self.output_cache.shape[0]

        # dL / dz has same shape as softmax output
        grad_input = np.zeros_like(s)
        for i in range(len_sample):
            s_i = s[i].reshape(-1, 1) # Column vector shape: (C, 1)
            # Jacobian matrix: diag(s) - s sᵀ       
            jacobian = np.diagflat(s_i) - np.dot(s_i, s_i.T)
            # Jacobian with gradient output 
            grad_input[i] = np.dot(jacobian, grad_output[i])
        return grad_input

class Dropout(Layer):
    def __init__(self, dropout_rate= 0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True

    def forward(self, x):
        if self.training and self.dropout_rate > 0:
            # Create dropout mask 
            self.mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float32)
            return x * self.mask / (1 - self.dropout_rate)
        return x.copy()
    
    def backward(self, grad_output):
        if self.training and self.dropout_rate > 0:
            return grad_output * self.mask / (1 - self.dropout_rate)
        return grad_output
    
    def set_training(self, training):
        self.training = training

import numpy as np

class BatchNorm1D(Layer):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True,
                 gamma=1.0, beta=0.0):
        self.num_features = num_features
        self.affine = affine

        # Exponentially weighted average
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters
        if self.affine:
            self.gamma = gamma * np.ones(num_features)  # scale
            self.beta = beta * np.ones(num_features)    # shift
        else:
            # Non-learnable parameters for non-affine mode
            self.gamma = np.ones(num_features)
            self.beta = np.zeros(num_features)

        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Gradients (only meaningful if affine=True)
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad = np.zeros_like(self.beta)

        # Cache for backward pass 
        self.input_cache = None
        self.normalized_cache = None
        self.var_cache = None
        self.mean_cache = None
        self.training = True

    def forward(self, x):
        if self.training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0, ddof=0)  # Use population variance

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)

            # Cache for backward pass 
            self.input_cache = x.copy()
            self.normalized_cache = x_norm.copy()  # Fixed variable name
            self.var_cache = batch_var.copy()
            self.mean_cache = batch_mean.copy()
        else: 
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        return self.gamma * x_norm + self.beta

    def backward(self, grad_output):
        if not self.training:
            # For inference mode, gradient flows through normalization only
            return grad_output * self.gamma / np.sqrt(self.running_var + self.eps)
        
        batch_size = self.input_cache.shape[0]
        x_centered = self.input_cache - self.mean_cache
        std_inv = 1.0 / np.sqrt(self.var_cache + self.eps)

        # Gradients for gamma and beta (only update if affine=True)
        if self.affine:
            self.gamma_grad += np.sum(grad_output * self.normalized_cache, axis=0)
            self.beta_grad += np.sum(grad_output, axis=0)

        # Gradients w.r.t normalized input 
        grad_norm = grad_output * self.gamma

        # Gradient w.r.t variance
        grad_var = -np.sum(grad_norm * x_centered, axis=0) * (std_inv ** 3) / 2

        # Gradient w.r.t mean
        grad_mean = (-np.sum(grad_norm * std_inv, axis=0) + 
                    grad_var * (-2.0 * np.sum(x_centered, axis=0) / batch_size))

        # Gradients w.r.t input 
        grad_input = (grad_norm * std_inv + 
                     2.0 * grad_var * x_centered / batch_size + 
                     grad_mean / batch_size)

        return grad_input
    
    def update_params(self, optimizer):
        if self.affine:
            optimizer.update(self.gamma, self.gamma_grad)
            optimizer.update(self.beta, self.beta_grad)

    def zero_grad(self):
        self.gamma_grad = np.zeros_like(self.gamma_grad)
        self.beta_grad = np.zeros_like(self.beta_grad)

    def set_training(self, training):
        self.training = training

class SoftmaxCrossEntropyLoss:
    
    def __init__(self):
        self.cache = None  

    def forward(self, logits, targets):
        N = logits.shape[0]  # Number of samples

        # Numerically stable softmax 
        # Shift logits by max for stability: exp(x - max) avoids overflow
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (N, C)

        # Compute cross-entropy loss 
        # Clip probabilities to prevent log(0)
        probs_clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)

        if targets.ndim == 1:  # Class indices
            # Extract log probability of true class
            log_probs = -np.log(probs_clipped[np.arange(N), targets])  # (N,)
        else:  # One-hot encoded
            # -sum(y * log(p)) for each sample
            log_probs = -np.sum(targets * np.log(probs_clipped), axis=1)  # (N,)

        loss = np.mean(log_probs)  # Average over batch

        #  Cache for backward pass 
        self.cache = (probs, targets)
        return loss

    def backward(self):
        assert self.cache is not None, "Must call forward() before backward()"
        probs, targets = self.cache
        N = probs.shape[0]

        # Gradient: (probs - targets) / N
        # But targets may be class indices or one-hot
        if targets.ndim == 1:
            # Convert class indices to one-hot for gradient computation
            grad = probs.copy()  # (N, C)
            grad[np.arange(N), targets] -= 1  # p_i - 1 for correct class
        else:
            grad = probs - targets  # (N, C)

        # Normalize by batch size because loss was averaged
        return grad / N

    def __call__(self, logits, targets):
        return self.forward(logits, targets)