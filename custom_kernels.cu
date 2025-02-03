#include <torch/extension.h>
#include <vector>

// ==================================
// === Matrix Multiplication (MatMul)
// ==================================
//
// Single-matrix or 3D input for A:
//    A: (b, s, in_dim)  or (M, K)
//    B: (in_dim, out_dim) or (K, N)
// Flatten as needed and do a standard matmul.
//

__global__ void matmul_forward_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // M = b*s, K = in_dim, N = out_dim
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // [0..M)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // [0..N)

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_backward_kernel(
    const float* dC, const float* A, const float* B,
    float* dA, float* dB,
    int M, int N, int K)
{
    // M = b*s, N = out_dim, K = in_dim
    //   dA = dC * B^T  => shape (M, K)
    //   dB = A^T * dC  => shape (K, N)

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Compute dA[row, col] ---
    if (row < M && col < K)
    {
        float val = 0.0f;
        for (int i = 0; i < N; i++) {
            val += dC[row * N + i] * B[i * K + col];
        }
        dA[row * K + col] = val;
    }

    // --- Compute dB[row2, col2] ---
    // Re-use the same thread index for the other part
    int row2 = row;  // blockIdx.y * blockDim.y + threadIdx.y
    int col2 = col;  // blockIdx.x * blockDim.x + threadIdx.x
    if (row2 < K && col2 < N)
    {
        float val = 0.0f;
        for (int i = 0; i < M; i++) {
            val += A[i * K + row2] * dC[i * N + col2];
        }
        dB[row2 * N + col2] = val;
    }
}

// ==================================
// === Matrix Multiplication (MatMul2) - batch
// ==================================
//
// Here we want: a: (b, M, K), b: (b, K, N) => c: (b, M, N)
// We'll do a naive kernel: For each (batch, row, col), sum over K.
//

__global__ void matmul2_forward_kernel(
    const float* A, const float* B, float* C,
    int BATCH, int M, int N, int K)
{
    // blockIdx.z => batch in [0..BATCH)
    // blockIdx.y => row in [0..M)
    // blockIdx.x => col in [0..N)
    int batch_id = blockIdx.z;
    int row      = blockIdx.y;
    int col      = blockIdx.x;

    if (batch_id < BATCH && row < M && col < N)
    {
        // A -> index: batch_id*M*K + row*K + k
        // B -> index: batch_id*K*N + k*N + col
        // C -> index: batch_id*M*N + row*N + col
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < K; k_idx++) {
            float a_val = A[batch_id * (M*K) + row * K + k_idx];
            float b_val = B[batch_id * (K*N) + k_idx * N + col];
            sum += a_val * b_val;
        }
        C[batch_id * (M*N) + row * N + col] = sum;
    }
}

__global__ void matmul2_backward_kernel(
    const float* dC, const float* A, const float* B,
    float* dA, float* dB,
    int BATCH, int M, int N, int K)
{
    // We want to compute:
    //   dA[b, i, k] = sum_j( dC[b, i, j] * B[b, k, j] )
    //   dB[b, k, j] = sum_i( dC[b, i, j] * A[b, i, k] )

    int batch_id = blockIdx.z;
    int row      = blockIdx.y; // we can interpret as 'i' for dA or 'k' for dB
    int col      = blockIdx.x; // we can interpret as 'k' for dA or 'j' for dB

    if (batch_id < BATCH)
    {
        // --- dA part: shape (BATCH, M, K) ---
        // row in [0..M), col in [0..K)
        // We'll do that in one portion of the grid
        if (row < M && col < K) {
            float val = 0.0f;
            for (int j = 0; j < N; j++) {
                float dc_val = dC[batch_id * (M*N) + row*N + j]; // dC[b, i=row, j]
                float b_val  = B[batch_id * (K*N) + col*N + j];  // B[b, k=col, j]
                val += dc_val * b_val;
            }
            dA[batch_id * (M*K) + row*K + col] = val;
        }

        // --- dB part: shape (BATCH, K, N) ---
        // row in [0..K), col in [0..N)
        // We'll reuse the same thread indices, but interpret them differently
        if (row < K && col < N) {
            float val = 0.0f;
            for (int i = 0; i < M; i++) {
                float dc_val = dC[batch_id * (M*N) + i*N + col]; // dC[b, i, j=col]
                float a_val  = A[batch_id * (M*K) + i*K + row];  // A[b, i, k=row]
                val += dc_val * a_val;
            }
            dB[batch_id * (K*N) + row*N + col] = val;
        }
    }
}

// ==================================
// === Softmax Forward & Backward
// ==================================
//
// Flatten (b, s, d) => (b*s, d) or use (batch, dim).
// Then do row-wise softmax.
//

__global__ void softmax_forward_kernel(
    const float* input, float* output, int total_rows, int dim)
{
    // total_rows = b*s (or batch_size), each block = 1 row
    int row = blockIdx.x; 
    int col = threadIdx.x; 
    extern __shared__ float shared_exp[];

    // 1) find max
    __shared__ float row_max;
    if (col == 0) {
        float m = -1e30f;
        for (int i = 0; i < dim; i++) {
            float val = input[row * dim + i];
            if (val > m) m = val;
        }
        row_max = m;
    }
    __syncthreads();

    // 2) compute exp
    float val = input[row * dim + col];
    float e   = expf(val - row_max);
    shared_exp[col] = e;
    __syncthreads();

    // 3) sum of exp
    float sum_exp = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_exp += shared_exp[i];
    }

    // 4) write
    output[row * dim + col] = e / sum_exp;
}

__global__ void softmax_backward_kernel(
    const float* grad_output,
    const float* softmax_output,
    float* grad_input,
    int total_rows,
    int dim)
{
    int row = blockIdx.x;
    int col = threadIdx.x;

    // reduce to find sum_j (y_j * grad_j)
    __shared__ float sum_y_grad;
    if (col == 0) {
        float acc = 0.0f;
        for (int i = 0; i < dim; i++) {
            float y  = softmax_output[row * dim + i];
            float gy = grad_output[row * dim + i];
            acc += y * gy;
        }
        sum_y_grad = acc;
    }
    __syncthreads();

    float y  = softmax_output[row * dim + col];
    float gy = grad_output[row * dim + col];

    float gx = y * (gy - sum_y_grad);
    grad_input[row * dim + col] = gx;
}

// ==================================
// === Rotary Position Embedding (RoPE)
// ==================================

__global__ void rope_forward_kernel(
    const float* input, float* output,
    const float* sin_table, const float* cos_table,
    int seq_len, int dim)
{
    int batch_id    = blockIdx.x;
    int seq_id      = blockIdx.y;
    int half_dim_id = threadIdx.x;  // from 0..(dim/2 - 1)

    // index offset
    int idx = (batch_id * seq_len + seq_id) * dim + half_dim_id;
    if (half_dim_id < dim/2) {
        float x1 = input[idx];
        float x2 = input[idx + dim/2];

        float s_val = sin_table[seq_id * (dim/2) + half_dim_id];
        float c_val = cos_table[seq_id * (dim/2) + half_dim_id];

        output[idx]           = x1 * c_val - x2 * s_val;
        output[idx + dim/2]   = x1 * s_val + x2 * c_val;
    }
}

__global__ void rope_backward_kernel(
    const float* grad_output, float* grad_input,
    const float* sin_table, const float* cos_table,
    int seq_len, int dim)
{
    int batch_id    = blockIdx.x;
    int seq_id      = blockIdx.y;
    int half_dim_id = threadIdx.x;

    int idx = (batch_id * seq_len + seq_id) * dim + half_dim_id;
    if (half_dim_id < dim/2)
    {
        float g1 = grad_output[idx];
        float g2 = grad_output[idx + dim/2];

        float s_val = sin_table[seq_id * (dim/2) + half_dim_id];
        float c_val = cos_table[seq_id * (dim/2) + half_dim_id];

        // derivative of:
        // out_x1 = x1*c - x2*s
        // out_x2 = x1*s + x2*c
        // => dL/dx1 = g1*c + g2*s
        // => dL/dx2 = -g1*s + g2*c
        grad_input[idx]          = g1 * c_val + g2 * s_val;
        grad_input[idx + dim/2]  = -g1 * s_val + g2 * c_val;
    }
}

// ==================================
// === PyTorch Bindings
// ==================================

// ---------- matmul ----------
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "matmul_forward expects A to be 2D or 3D.");
    TORCH_CHECK(B.dim() == 2, "matmul_forward expects B to be 2D.");

    if (A.dim() == 3) {
        int b = A.size(0);
        int s = A.size(1);
        int in_dim  = A.size(2);
        int out_dim = B.size(1);

        auto A_2d = A.reshape({b*s, in_dim});
        auto C_2d = torch::zeros({b*s, out_dim}, A.options());

        int M = b*s;
        int K = in_dim;
        int N = out_dim;

        dim3 block(16, 16);
        dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);

        matmul_forward_kernel<<<grid, block>>>(
            A_2d.data_ptr<float>(),
            B.data_ptr<float>(),
            C_2d.data_ptr<float>(),
            M, N, K
        );

        return C_2d.reshape({b, s, out_dim});
    } else {
        // 2D
        int M = A.size(0);
        int K = A.size(1);
        int N = B.size(1);

        auto C = torch::zeros({M, N}, A.options());

        dim3 block(16, 16);
        dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);

        matmul_forward_kernel<<<grid, block>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K
        );

        return C;
    }
}

std::vector<torch::Tensor> matmul_backward(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor B)
{
    TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "matmul_backward expects A to be 2D or 3D.");
    TORCH_CHECK(B.dim() == 2, "matmul_backward expects B to be 2D.");

    if (A.dim() == 3) {
        int b = A.size(0);
        int s = A.size(1);
        int in_dim  = A.size(2);
        int out_dim = B.size(1);

        auto A_2d  = A.reshape({b*s, in_dim});
        auto dC_2d = dC.reshape({b*s, out_dim});

        auto dA_2d = torch::zeros_like(A_2d);
        auto dB_2d = torch::zeros_like(B);

        int M = b*s;  // row
        int K = in_dim;
        int N = out_dim;

        dim3 block(16,16);
        dim3 grid((std::max(K,N) + block.x - 1)/block.x,
                  (M + block.y - 1)/block.y);

        matmul_backward_kernel<<<grid, block>>>(
            dC_2d.data_ptr<float>(),
            A_2d.data_ptr<float>(),
            B.data_ptr<float>(),
            dA_2d.data_ptr<float>(),
            dB_2d.data_ptr<float>(),
            M, N, K
        );

        auto dA = dA_2d.reshape({b, s, in_dim});
        return {dA, dB_2d};
    } else {
        // 2D
        int M = A.size(0);
        int K = A.size(1);
        int N = B.size(1);

        auto dA = torch::zeros_like(A);
        auto dB = torch::zeros_like(B);

        dim3 block(16,16);
        dim3 grid((std::max(K,N) + block.x - 1)/block.x,
                  (M + block.y - 1)/block.y);

        matmul_backward_kernel<<<grid, block>>>(
            dC.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            dA.data_ptr<float>(),
            dB.data_ptr<float>(),
            M, N, K
        );

        return {dA, dB};
    }
}

// ---------- matmul2 (batch matmul) ----------
torch::Tensor matmul2_forward(torch::Tensor A, torch::Tensor B) {
    // A: (b, M, K)
    // B: (b, K, N)
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3,
        "matmul2_forward expects A, B both 3D with shape (b, M, K), (b, K, N).");

    int BATCH = A.size(0);
    int M     = A.size(1);
    int K     = A.size(2);
    TORCH_CHECK(B.size(0) == BATCH, "Batch dim mismatch.");
    TORCH_CHECK(B.size(1) == K,     "Inner dim mismatch for matmul2.");

    int N = B.size(2);

    auto C = torch::zeros({BATCH, M, N}, A.options());

    // We'll launch a 3D grid: (N, M, BATCH)
    dim3 grid(N, M, BATCH);
    dim3 block(1, 1);  
    // This is extremely naive (one thread does one C-element).
    // If M*N is large, you might do something more elaborate.

    matmul2_forward_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, M, N, K
    );

    return C;
}

std::vector<torch::Tensor> matmul2_backward(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor B)
{
    // dC: (b, M, N)
    // A: (b, M, K)
    // B: (b, K, N)
    int BATCH = A.size(0);
    int M     = A.size(1);
    int K     = A.size(2);
    int N     = B.size(2);

    auto dA = torch::zeros_like(A);
    auto dB = torch::zeros_like(B);

    // We'll launch a 3D grid: (max(K,N), max(M,K), BATCH), but let's keep it simpler:
    // or do a large grid of size (max(M,N), max(M,N), BATCH). We'll do something naive:
    int maxDimY = std::max(M, K);
    int maxDimX = std::max(K, N);

    dim3 grid(maxDimX, maxDimY, BATCH);
    dim3 block(1, 1);

    matmul2_backward_kernel<<<grid, block>>>(
        dC.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        dA.data_ptr<float>(),
        dB.data_ptr<float>(),
        BATCH, M, N, K
    );

    return {dA, dB};
}

// ---------- softmax ----------
torch::Tensor softmax_forward(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2 || input.dim() == 3,
                "softmax_forward expects 2D or 3D input.");

    if (input.dim() == 3) {
        int b = input.size(0);
        int s = input.size(1);
        int d = input.size(2);

        auto in_2d  = input.reshape({b*s, d});
        auto out_2d = torch::zeros_like(in_2d);

        int total_rows = b*s;
        int dim        = d;

        softmax_forward_kernel<<<total_rows, dim, dim*sizeof(float)>>>(
            in_2d.data_ptr<float>(),
            out_2d.data_ptr<float>(),
            total_rows,
            dim
        );

        return out_2d.reshape({b, s, d});
    } else {
        // 2D
        int batch_size = input.size(0);
        int dim = input.size(1);
        auto output = torch::zeros_like(input);

        softmax_forward_kernel<<<batch_size, dim, dim*sizeof(float)>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            dim
        );
        return output;
    }
}

torch::Tensor softmax_backward(torch::Tensor grad_output, torch::Tensor softmax_output) {
    TORCH_CHECK(grad_output.dim() == 2 || grad_output.dim() == 3,
                "softmax_backward expects 2D or 3D grad_output.");
    TORCH_CHECK(softmax_output.dim() == grad_output.dim(),
                "softmax_backward input dims must match.");

    if (grad_output.dim() == 3) {
        int b = grad_output.size(0);
        int s = grad_output.size(1);
        int d = grad_output.size(2);

        auto grad_out_2d = grad_output.reshape({b*s, d});
        auto sm_out_2d   = softmax_output.reshape({b*s, d});
        auto grad_in_2d  = torch::zeros_like(grad_out_2d);

        int total_rows = b*s;

        softmax_backward_kernel<<<total_rows, d>>>(
            grad_out_2d.data_ptr<float>(),
            sm_out_2d.data_ptr<float>(),
            grad_in_2d.data_ptr<float>(),
            total_rows,
            d
        );

        return grad_in_2d.reshape({b, s, d});
    } else {
        // 2D
        int batch_size = grad_output.size(0);
        int dim = grad_output.size(1);

        auto grad_input = torch::zeros_like(grad_output);
        softmax_backward_kernel<<<batch_size, dim>>>(
            grad_output.data_ptr<float>(),
            softmax_output.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            batch_size,
            dim
        );
        return grad_input;
    }
}

// ---------- rope ----------
torch::Tensor rope_forward(
    torch::Tensor input,
    torch::Tensor sin_table,
    torch::Tensor cos_table)
{
    // (b, seq, dim)
    auto output = torch::zeros_like(input);

    int b   = input.size(0);
    int s   = input.size(1);
    int dim = input.size(2);

    dim3 blocks(b, s);
    int threads = dim / 2;

    rope_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        cos_table.data_ptr<float>(),
        s,
        dim
    );
    return output;
}

torch::Tensor rope_backward(
    torch::Tensor grad_output,
    torch::Tensor sin_table,
    torch::Tensor cos_table)
{
    auto grad_input = torch::zeros_like(grad_output);

    int b   = grad_output.size(0);
    int s   = grad_output.size(1);
    int dim = grad_output.size(2);

    dim3 blocks(b, s);
    int threads = dim / 2;

    rope_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        cos_table.data_ptr<float>(),
        s,
        dim
    );
    return grad_input;
}

// ==================================
// === PyBind Module
// ==================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Standard matmul
    m.def("matmul_forward", &matmul_forward, "Custom matmul forward");
    m.def("matmul_backward", &matmul_backward, "Custom matmul backward");

    // Batch matmul2
    m.def("matmul2_forward", &matmul2_forward, "Custom batch-matmul forward");
    m.def("matmul2_backward", &matmul2_backward, "Custom batch-matmul backward");

    // Softmax
    m.def("softmax_forward", &softmax_forward, "Custom softmax forward");
    m.def("softmax_backward", &softmax_backward, "Custom softmax backward");

    // RoPE
    m.def("rope_forward", &rope_forward, "Custom RoPE forward");
    m.def("rope_backward", &rope_backward, "Custom RoPE backward");
}
