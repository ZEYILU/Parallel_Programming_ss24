#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Function prototypes
void read_matrix(const char *filename, double **A, double **b, int *n);
void write_solution(const char *filename, double *x, int n);

// CUDA kernel function for Gauss-Jacobi iteration
__global__ void gaussJacobiKernel(double *A, double *b, double *x, double *x_new, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        x_new[idx] = b[idx];
        for (int j = 0; j < N; j++) {
            if (j != idx) {
                x_new[idx] -= A[idx * N + j] * x[j];
            }
        }
        x_new[idx] /= A[idx * N + idx];
    }
}

// Host function to manage memory and launch the kernel
void gaussJacobiCUDA(double *A, double *b, double *x, int N, int max_iter, double tol, int numBlocks, int threadsPerBlock) {
    double *d_A, *d_b, *d_x, *d_x_new;

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * N * sizeof(double));
    cudaMalloc((void**)&d_b, N * sizeof(double));
    cudaMalloc((void**)&d_x, N * sizeof(double));
    cudaMalloc((void**)&d_x_new, N * sizeof(double));
    
    // Copy data to device
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
    
    double *x_new = (double *)malloc(N * sizeof(double));
    double max_diff, diff;
    int iter = 0;

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Start timing
    cudaEventRecord(start);
    
    do {
        max_diff = 0.0;
        
        // Launch kernel
        gaussJacobiKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_b, d_x, d_x_new, N);
        cudaDeviceSynchronize();
        
        // Copy new solution to host
        cudaMemcpy(x_new, d_x_new, N * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Calculate maximum difference
        for (int i = 0; i < N; ++i) {
            diff = fabs(x_new[i] - x[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
            x[i] = x_new[i];
        }
        
        // Copy updated solution to device
        cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
        
        iter++;
    } while (iter < max_iter && max_diff > tol);
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Running with %d Blocks and %d threads per Block\n", numBlocks, threadsPerBlock);
    printf("Execution time: %.6f seconds\n", milliseconds / 1000.0);
    
    // Cleanup
    free(x_new);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_x_new);
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <max_iter> <num_blocks> <threads_per_block>\n";
        return EXIT_FAILURE;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int max_iter = atoi(argv[3]);
    int numBlocks = atoi(argv[4]);
    int threadsPerBlock = atoi(argv[5]);
    double tol = 1e-10;

    double *A, *b, *x;
    int N;

    // Read matrix and vector from input file
    read_matrix(input_file, &A, &b, &N);

    // Allocate memory for the solution vector
    x = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; ++i) {
        x[i] = 0.0; // Initial guess
    }

    // Run the Gauss-Jacobi method using CUDA
    gaussJacobiCUDA(A, b, x, N, max_iter, tol, numBlocks, threadsPerBlock);

    // Write the solution to the output file
    write_solution(output_file, x, N);

    // Free allocated memory
    free(A);
    free(b);
    free(x);

    return 0;
}

// Function to read matrix and vector from a file
void read_matrix(const char *filename, double **A, double **b, int *n) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    fscanf(file, "%d", n);
    *A = (double *)malloc((*n) * (*n) * sizeof(double));
    *b = (double *)malloc((*n) * sizeof(double));
    for (int i = 0; i < (*n) * (*n); ++i) {
        fscanf(file, "%lf", &(*A)[i]);
    }
    for (int i = 0; i < *n; ++i) {
        fscanf(file, "%lf", &(*b)[i]);
    }
    fclose(file);
}

// Function to write the solution vector to a file
void write_solution(const char *filename, double *x, int n) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; ++i) {
        fprintf(file, "%.10lf\n", x[i]);
    }
    fclose(file);
}
