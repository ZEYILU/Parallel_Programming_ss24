#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// void read_matrix(const char* filename, double** A11, double** A22, double** A33, double** A13, double** A23, double** A31, double** A32, double** f1, double** f2, double** f3, int* N, int* n);
// void schur_complement_solve(double* A11, double* A22, double* A33, double* A13, double* A23, double* A31, double* A32, double* f1, double* f2, double* f3, double* u1, double* u2, double* u3, int N, int n);
// void write_output(const char* filename, double* u1, double* u2, double* u3, int N, int n);

// 矩阵-向量乘法，使用OpenMP并行化
void matrix_vector_mult(double* matrix, double* vector, double* result, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

// 矩阵-矩阵乘法，使用OpenMP并行化
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

// LU分解
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
        if (maxA < 1e-12) return 0; // 奇异矩阵
        if (imax != i) {
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

// LU求解
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

// 计算Schur补矩阵并解决小系统
void schur_solve(double* A11, double* A22, double* A13, double* A23, double* A31, double* A32, double* A33, double* f1, double* f2, double* f3, double* u3, int N, int n) {
    double* A11_inv = (double*)malloc(N * N * sizeof(double));
    double* A22_inv = (double*)malloc(N * N * sizeof(double));
    int* P11 = (int*)malloc(N * sizeof(int));
    int* P22 = (int*)malloc(N * sizeof(int));
    lu_decomposition(A11, P11, N);
    lu_decomposition(A22, P22, N);

    double* S = (double*)malloc(n * n * sizeof(double));
    double* temp1 = (double*)malloc(N * n * sizeof(double));
    double* temp2 = (double*)malloc(N * n * sizeof(double));

    matrix_matrix_mult(A31, A11_inv, temp1, n, N, N);
    matrix_matrix_mult(temp1, A13, temp2, n, N, n);

    matrix_matrix_mult(A32, A22_inv, temp1, n, N, N);
    matrix_matrix_mult(temp1, A23, temp2, n, N, n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            S[i * n + j] = A33[i * n + j] - temp2[i * n + j];
        }
    }

    // Solve S * u3 = f3 - A31 * inv(A11) * f1 - A32 * inv(A22) * f2
    double* temp3 = (double*)malloc(n * sizeof(double));
    matrix_vector_mult(A31, f1, temp3, n, N);
    matrix_vector_mult(A32, f2, temp3, n, N);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        u3[i] = f3[i] - temp3[i];
    }

    lu_solve(S, u3, u3, P11, n);

    free(A11_inv);
    free(A22_inv);
    free(P11);
    free(P22);
    free(S);
    free(temp1);
    free(temp2);
    free(temp3);
}

void load_input(const char* filename, double* A11, double* A22, double* A33, double* A13, double* A23, double* A31, double* A32, double* f1, double* f2, double* f3, int N, int n) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Cannot open file %s\n", filename);
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
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1000;
    int n = 100;

    double* A11 = (double*)malloc(N * N * sizeof(double));
    double* A22 = (double*)malloc(N * N * sizeof(double));
    double* A13 = (double*)malloc(N * n * sizeof(double));
    double* A23 = (double*)malloc(N * n * sizeof(double));
    double* A31 = (double*)malloc(n * N * sizeof(double));
    double* A32 = (double*)malloc(n * N * sizeof(double));
    double* A33 = (double*)malloc(n * n * sizeof(double));
    double* f1 = (double*)malloc(N * sizeof(double));
    double* f2 = (double*)malloc(N * sizeof(double));
    double* f3 = (double*)malloc(n * sizeof(double));
    double* u1 = (double*)malloc(N * sizeof(double));
    double* u2 = (double*)malloc(N * sizeof(double));
    double* u3 = (double*)malloc(n * sizeof(double));


    if (rank == 0) {
    load_input(argv[1], A11, A22, A33, A13, A23, A31, A32, f1, f2, f3, N, n);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);


    MPI_Bcast(A11, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(A22, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(A33, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(A13, N * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(A23, N * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(A31, n * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(A32, n * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(f1, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(f2, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(f3, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    schur_solve(A11, A22, A13, A23, A31, A32, A33, f1, f2, f3, u3, N, n);


    int chunk_size = N / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? N : start + chunk_size;

    #pragma omp parallel for
    for (int i = start; i < end; i++) {
        u1[i] = f1[i];
        for (int j = 0; j < n; j++) {
            u1[i] -= A13[i * n + j] * u3[j];
        }
        for (int k = 0; k < N; k++) {
            u1[i] /= A11[i * N + k];
        }
    }

    #pragma omp parallel for
    for (int i = start; i < end; i++) {
        u2[i] = f2[i];
        for (int j = 0; j < n; j++) {
            u2[i] -= A23[i * n + j] * u3[j];
        }
        for (int k = 0; k < N; k++) {
            u2[i] /= A22[i * N + k];
        }
    }


    double end_time = MPI_Wtime();
    // 收集和输出结果，省略
    if (rank == 0) {
    printf("Time taken: %lf seconds\n", end_time - start_time);
    write_output(argv[2], u1, u2, u3, N, n);
    free(u1);
    free(u2);
    free(u3);
    }

    free(A11);
    free(A22);
    free(A13);
    free(A23);
    free(A31);
    free(A32);
    free(A33);
    free(f1);
    free(f2);
    free(f3);

    MPI_Finalize();
    return 0;
}

