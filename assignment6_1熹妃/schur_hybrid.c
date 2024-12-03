#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to perform matrix-vector multiplication using OpenMP
void matrix_vector_mult(double* matrix, double* vector, double* result, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

// Function to perform matrix-matrix multiplication using OpenMP
void matrix_matrix_mult(double* A, double* B, double* C, int m, int n, int p) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

// Function to invert a matrix using LU decomposition
int lu_decomposition(double* A, int* P, int N) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        P[i] = i;
    }
    for (i = 0; i < N; i++) {
        double maxA = 0.0;
        int imax = i;
        for (k = i; k < N; k++) {
            double absA = fabs(A[k * N + i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }
        if (maxA < 1e-12) return 0; // Singular matrix
        if (imax != i) {
            // Pivoting
            int j = P[i];
            P[i] = P[imax];
            P[imax] = j;
            for (k = 0; k < N; k++) {
                double temp = A[i * N + k];
                A[i * N + k] = A[imax * N + k];
                A[imax * N + k] = temp;
            }
        }
        for (j = i + 1; j < N; j++) {
            A[j * N + i] /= A[i * N + i];
            for (k = i + 1; k < N; k++) {
                A[j * N + k] -= A[j * N + i] * A[i * N + k];
            }
        }
    }
    return 1;
}

void lu_solve(double* A, double* B, double* x, int* P, int N) {
    int i, k;
    for (i = 0; i < N; i++) {
        x[i] = B[P[i]];
        for (k = 0; k < i; k++) {
            x[i] -= A[i * N + k] * x[k];
        }
    }
    for (i = N - 1; i >= 0; i--) {
        for (k = i + 1; k < N; k++) {
            x[i] -= A[i * N + k] * x[k];
        }
        x[i] /= A[i * N + i];
    }
}

// Function to invert a matrix using LU decomposition
void invert_matrix(double* A, double* A_inv, int N) {
    int* P = (int*)malloc(N * sizeof(int));
    double* B = (double*)malloc(N * sizeof(double));
    double* x = (double*)malloc(N * sizeof(double));
    
    if (lu_decomposition(A, P, N)) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B[j] = (i == j) ? 1.0 : 0.0;
            }
            lu_solve(A, B, x, P, N);
            for (int j = 0; j < N; j++) {
                A_inv[j * N + i] = x[j];
            }
        }
    } else {
        fprintf(stderr, "Matrix inversion failed due to singular matrix.\n");
        exit(EXIT_FAILURE);
    }
    
    free(P);
    free(B);
    free(x);
}

// Function to read matrices and vectors from a file (only by root process)
void read_data(const char* filename, double* A11, double* A22, double* A33, double* A13, double* A23, double* A31, double* A32, double* f1, double* f2, double* f3, int N, int n) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        exit(1);
    }
    
    fscanf(file, "%d %d", &N, &n);

    for (int i = 0; i < N * N; i++) fscanf(file, "%lf", &A11[i]);
    for (int i = 0; i < N * N; i++) fscanf(file, "%lf", &A22[i]);
    for (int i = 0; i < n * n; i++) fscanf(file, "%lf", &A33[i]);
    for (int i = 0; i < N * n; i++) fscanf(file, "%lf", &A13[i]);
    for (int i = 0; i < N * n; i++) fscanf(file, "%lf", &A23[i]);
    for (int i = 0; i < n * N; i++) fscanf(file, "%lf", &A31[i]);
    for (int i = 0; i < n * N; i++) fscanf(file, "%lf", &A32[i]);

    for (int i = 0; i < N; i++) fscanf(file, "%lf", &f1[i]);
    for (int i = 0; i < N; i++) fscanf(file, "%lf", &f2[i]);
    for (int i = 0; i < n; i++) fscanf(file, "%lf", &f3[i]);

    fclose(file);

}

// Function to write the vector to a file
void write_output(const char* filename, double* u1, double* u2, double* u3, int N, int n) {
    FILE* file = fopen(filename, "w");
    fprintf(file, "u1: [");
    for (int i = 0; i < N; i++) {
        fprintf(file, "%lf", u1[i]);
        if (i < N - 1) fprintf(file, ", ");
    }
    fprintf(file, "]\n");
    fprintf(file, "u2: [");
    for (int i = 0; i < N; i++) {
        fprintf(file, "%lf", u2[i]);
        if (i < N - 1) fprintf(file, ", ");
    }
    fprintf(file, "]\n");
    fprintf(file, "u3: [");
    for (int i = 0; i < n; i++) {
        fprintf(file, "%lf", u3[i]);
        if (i < n - 1) fprintf(file, ", ");
    }
    fprintf(file, "]\n");
    fclose(file);
}

// Gauss-Seidel solver for solving S * u3 = RHS
void solve_schur_complement(double* S, double* RHS, double* u3, int N) {
    int max_iterations = 1000;
    double tolerance = 1e-6;
    double* old_u3 = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        u3[i] = 0.0;
    }

    for (int iter = 0; iter < max_iterations; iter++) {
        double max_diff = 0.0;
        for (int i = 0; i < N; i++) {
            old_u3[i] = u3[i];
            double sum = RHS[i];
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    sum -= S[i * N + j] * u3[j];
                }
            }
            u3[i] = sum / S[i * N + i];
            double diff = fabs(u3[i] - old_u3[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
        if (max_diff < tolerance) {
            break;
        }
    }

    free(old_u3);
}

void compute_schur_complement(double* matrices, double* vectors, int N, int n, double* u3) {
    // Assuming matrices are: A11, A22, A33, A13, A23, A31, A32
    // Vectors are: f1, f2, f3
    double *A11 = matrices;
    double *A22 = matrices + N * N;
    double *A33 = matrices + 2 * N * N;
    double *A13 = matrices + 2 * N * N + N * n;
    double *A23 = matrices + 2 * N * N + 2 * N * n;
    double *A31 = matrices + 2 * N * N + 3 * N * n;
    double *A32 = matrices + 2 * N * N + 3 * N * n + n * N;

    double *f1 = vectors;
    double *f2 = vectors + N;
    double *f3 = vectors + 2 * N;

    double *A11_inv = (double*)malloc(N * N * sizeof(double));
    double *A22_inv = (double*)malloc(N * N * sizeof(double));
    double *A31_A11_inv = (double*)malloc(n * N * sizeof(double));
    double *A32_A22_inv = (double*)malloc(n * N * sizeof(double));
    double *A31_A11_inv_A13 = (double*)malloc(n * n * sizeof(double));
    double *A32_A22_inv_A23 = (double*)malloc(n * n * sizeof(double));
    double *temp1 = (double*)malloc(n * sizeof(double));
    double *temp2 = (double*)malloc(n * sizeof(double));
    double *RHS = (double*)malloc(n * sizeof(double));
    double *S = (double*)malloc(n * n * sizeof(double));

    // Compute inverse of A11 and A22
    invert_matrix(A11, A11_inv, N);
    invert_matrix(A22, A22_inv, N);

    // Compute A31 * inv(A11) and A32 * inv(A22)
    matrix_matrix_mult(A31, A11_inv, A31_A11_inv, n, N, N);
    matrix_matrix_mult(A32, A22_inv, A32_A22_inv, n, N, N);

    // Compute A31 * inv(A11) * A13 and A32 * inv(A22) * A23
    matrix_matrix_mult(A31_A11_inv, A13, A31_A11_inv_A13, n, N, n);
    matrix_matrix_mult(A32_A22_inv, A23, A32_A22_inv_A23, n, N, n);

    // Compute Schur complement S = A33 - A31 * inv(A11) * A13 - A32 * inv(A22) * A23
    for (int i = 0; i < n * n; i++) {
        S[i] = A33[i] - A31_A11_inv_A13[i] - A32_A22_inv_A23[i];
    }

    // Compute RHS = f3 - A31 * inv(A11) * f1 - A32 * inv(A22) * f2
    matrix_vector_mult(A31_A11_inv, f1, temp1, n, N);
    matrix_vector_mult(A32_A22_inv, f2, temp2, n, N);

    for (int i = 0; i < n; i++) {
        RHS[i] = f3[i] - temp1[i] - temp2[i];
    }

    // Solve S * u3 = RHS
    solve_schur_complement(S, RHS, u3, n);

    free(A11_inv);
    free(A22_inv);
    free(A31_A11_inv);
    free(A32_A22_inv);
    free(A31_A11_inv_A13);
    free(A32_A22_inv_A23);
    free(temp1);
    free(temp2);
    free(RHS);
    free(S);
}

void solve_with_mpi(double* matrices, double* vectors, int N, int n) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double* u3 = (double*)malloc(n * sizeof(double)); // Result vector for u3
    compute_schur_complement(matrices, vectors, N, n, u3);

    if (world_rank == 0) {
        write_vector("output_u3.txt", u3, n);
    }

    // Each MPI process computes its part of u1 and u2 using the same matrices and u3
    double *A11 = matrices;
    double *A22 = matrices + N * N;
    double *A13 = matrices + 2 * N * N + N * n;
    double *A23 = matrices + 2 * N * N + 2 * N * n;
    double *f1 = vectors;
    double *f2 = vectors + N;

    double *u1 = (double*)malloc(N * sizeof(double));
    double *u2 = (double*)malloc(N * sizeof(double));

    // Using OpenMP within each MPI process
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double sum1 = f1[i];
        for (int j = 0; j < n; j++) {
            sum1 -= A13[i * n + j] * u3[j];
        }
        u1[i] = sum1 / A11[i * N + i];

        double sum2 = f2[i];
        for (int j = 0; j < n; j++) {
            sum2 -= A23[i * n + j] * u3[j];
        }
        u2[i] = sum2 / A22[i * N + i];
    }

    if (world_rank == 0) {
        write_vector("output_u1.txt", u1, N);
        write_vector("output_u2.txt", u2, N);
    }

    free(u3);
    free(u1);
    free(u2);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double *matrices, *vectors;
    int N, n;

    if (world_rank == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        read_data(argv[1], &matrices, &vectors, &N, &n);
    }

    // Broadcast N and n from the root process to all other processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for matrices and vectors in all processes
    if (world_rank != 0) {
        matrices = (double*)malloc((2 * N * N + 4 * N * n + n * n) * sizeof(double)); // 7 matrices
        vectors = (double*)malloc((2 * N + n) * sizeof(double));  // 3 vectors
    }

    // Broadcast matrices and vectors to all processes
    MPI_Bcast(matrices, (2 * N * N + 4 * N * n + n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(vectors, (2 * N + n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    solve_with_mpi(matrices, vectors, N, n);

    free(matrices);
    free(vectors);

    MPI_Finalize();
    return 0;
}

