import torch
import torch.nn as nn
import torch.nn.functional as F

import custom_kernels  # <-- Custom CUDA kernels for matmul, matmul2, softmax, rope, etc.


# ==================================
# 1. Custom Autograd Function for MatMul
# ==================================
class MatMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        input:  (batch_size, seq_len, in_dim) or (M, K)
        weight: (in_dim, out_dim)             or (K, N)
        returns (batch_size, seq_len, out_dim) or (M, N)
        """
        ctx.save_for_backward(input, weight)
        output = custom_kernels.matmul_forward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight = custom_kernels.matmul_backward(grad_output, input, weight)
        return grad_input, grad_weight

def custom_matmul(input, weight):
    """
    Helper function for custom matmul usage in your model.
    """
    return MatMulFunction.apply(input, weight)


# ==================================
# 2. Custom Autograd Function for MatMul2 (batch matmul)
# ==================================
class MatMul2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        """
        Suppose:
          a: (batch, M, K)
          b: (batch, K, N)
        returns:
          (batch, M, N)
        """
        ctx.save_for_backward(a, b)
        out = custom_kernels.matmul2_forward(a, b)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (batch, M, N)
        We need dA: (batch, M, K) and dB: (batch, K, N)
        """
        (a, b) = ctx.saved_tensors
        grad_a, grad_b = custom_kernels.matmul2_backward(grad_output, a, b)
        return grad_a, grad_b

def custom_matmul2(a, b):
    """
    Helper function for custom batch-matmul usage in your model.
    """
    return MatMul2Function.apply(a, b)


# ==================================
# 3. Custom Autograd Function for Softmax
# ==================================
class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        input: (batch_size, seq_len, dim) or (batch_size, dim)
        returns the same shape, softmax along last dimension
        """
        output = custom_kernels.softmax_forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        grad_input = custom_kernels.softmax_backward(grad_output, output)
        return grad_input

def custom_softmax(input):
    return SoftmaxFunction.apply(input)


# ==================================
# 4. Custom Autograd Function for RoPE
# ==================================
class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sin_table, cos_table):
        """
        x:          (batch_size, seq_len, dim)
        sin_table:  (batch_size, seq_len, dim//2)
        cos_table:  (batch_size, seq_len, dim//2)
        """
        ctx.save_for_backward(sin_table, cos_table)
        out = custom_kernels.rope_forward(x, sin_table, cos_table)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        sin_table, cos_table = ctx.saved_tensors
        grad_input = custom_kernels.rope_backward(grad_output, sin_table, cos_table)
        # No gradients w.r.t. sin_table, cos_table
        return grad_input, None, None

# ==================================
# 5. Rotary Position Embedding (Module)
# ==================================
class RoPE(nn.Module):
    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

        # Generate position encoding (seq_len, 1)
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)

        # Compute frequency scaling factors for half dimension
        half_dim = dim // 2
        div_term = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / half_dim)
        )

        # sin_table, cos_table with shape (1, seq_len, dim//2)
        sin_table = torch.sin(position * div_term).unsqueeze(0)
        cos_table = torch.cos(position * div_term).unsqueeze(0)

        self.register_buffer("sin_table", sin_table.cuda())
        self.register_buffer("cos_table", cos_table.cuda())

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, dim)
        We'll expand sin_table, cos_table from (1, seq_len, dim//2)
        to (batch_size, seq_len, dim//2)
        """
        b, s, d = x.shape
        sin_table_expanded = self.sin_table.expand(b, -1, -1)
        cos_table_expanded = self.cos_table.expand(b, -1, -1)

        return RoPEFunction.apply(x, sin_table_expanded, cos_table_expanded)


# ==================================
# 6. Custom PyTorch Model
# ==================================
class CustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super().__init__()
        self.weight0 = nn.Parameter(torch.randn(input_dim, hidden_dim, device="cuda"))
        self.weight1 = nn.Parameter(torch.randn(input_dim, hidden_dim, device="cuda"))
        self.weight2 = nn.Parameter(torch.randn(hidden_dim, output_dim, device="cuda"))

        self.rope = RoPE(hidden_dim, seq_len)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        # custom_matmul 0
        x0 = custom_matmul(x, self.weight0)  # (b, s, hidden_dim)
        x0 = self.rope(x0)                  # apply RoPE

        # custom_matmul 1
        x1 = custom_matmul(x, self.weight1)  # (b, s, hidden_dim)
        x1 = self.rope(x1)                   # apply RoPE

        # custom_matmul2 => (b, s, s)
        x1 = x1.transpose(1, 2)
        x01 = custom_matmul2(x0, x1)
        x01 = custom_softmax(x01)  # softmax along last dim => shape still (b, s, s)

        # custom_matmul 2
        x2 = custom_matmul(x, self.weight2)  # (b, s, output_dim)

        # Another custom_matmul2 => (b, s, output_dim)
        # Because x01: (b, s, s), x2: (b, s, output_dim)
        output = custom_matmul2(x01, x2)  # => (b, s, output_dim)

        # If your final shape for the loss is (b, output_dim),
        # you might want to reduce the seq_len dimension, or
        # just pick x[:, 0, :] or do a mean-pool across seq_len, etc.
        # For example, let's pick x[:, 0, :]:
        output = output[:, 0, :]  # shape (b, output_dim)

        return output


# ==================================
# 7. Training Script
# ==================================
if __name__ == "__main__":
    # Model configuration
    batch_size = 32
    seq_len = 128
    input_dim = 128
    hidden_dim = 64
    output_dim = 10

    # Initialize model
    model = CustomModel(input_dim, hidden_dim, output_dim, seq_len).cuda()

    # Generate random input & labels
    inputs = torch.randn(batch_size, seq_len, input_dim, device="cuda", requires_grad=True)
    labels = torch.randint(0, output_dim, (batch_size,), device="cuda")

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(10):
        outputs = model(inputs)  # (batch_size, output_dim)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
