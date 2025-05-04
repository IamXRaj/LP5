#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matrixMul(const float *a, const float *b, float *c, int rows_a, int cols_a, int cols_b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows_a && col < cols_b) {
        float sum = 0.0f;
        for (int k = 0; k < cols_a; ++k) {
            sum += a[row * cols_a + k] * b[k * cols_b + col];
        }
        c[row * cols_b + col] = sum;
    }
}

#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

void printVector(const float *vec, int n, const char *name) {
    printf("%s: ", name);
    for (int i = 0; i < n && i < 10; ++i) {
        printf("%.2f ", vec[i]);
    }
    if (n > 10) printf("...");
    printf("\n");
}

void printMatrix(const float *mat, int rows, int cols, const char *name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows && i < 4; ++i) {
        for (int j = 0; j < cols && j < 4; ++j) {
            printf("%.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
    if (rows > 4 || cols > 4) printf("...\n");
}

int main() {
    const int N = 10000;
    size_t size = N * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_a[i] = rand() % 100 / 10.0f;
        h_b[i] = rand() % 100 / 10.0f;
    }

    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    printVector(h_a, N, "Vector A");
    printVector(h_b, N, "Vector B");
    printVector(h_c, N, "Vector C (A + B)");

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);

    const int rows_a = 512, cols_a = 256, cols_b = 512;
    size_t size_a = rows_a * cols_a * sizeof(float);
    size_t size_b = cols_a * cols_b * sizeof(float);
    size_t size_c = rows_a * cols_b * sizeof(float);

    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);

    for (int i = 0; i < rows_a * cols_a; ++i) {
        h_a[i] = rand() % 100 / 10.0f;
    }
    for (int i = 0; i < cols_a * cols_b; ++i) {
        h_b[i] = rand() % 100 / 10.0f;
    }

    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size_a));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size_b));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size_c));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

    dim3 threadsPerBlockMat(16, 16);
    dim3 blocksPerGridMat((cols_b + threadsPerBlockMat.x - 1) / threadsPerBlockMat.x,
                          (rows_a + threadsPerBlockMat.y - 1) / threadsPerBlockMat.y);
    matrixMul<<<blocksPerGridMat, threadsPerBlockMat>>>(d_a, d_b, d_c, rows_a, cols_a, cols_b);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost));

    printMatrix(h_a, rows_a, cols_a, "Matrix A");
    printMatrix(h_b, cols_a, cols_b, "Matrix B");
    printMatrix(h_c, rows_a, cols_b, "Matrix C (A * B)");

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}