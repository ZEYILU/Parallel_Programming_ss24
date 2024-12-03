#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 1000
#define TOL 1e-6

void gauss_jacobi(int n, double **A, double *b, double *x, int rank, int size) {
    double *x_new = (double *)malloc(n * sizeof(double));
    int iter, i, j;

    for (i = 0; i < n; i++) {
        x[i] = 0.0;  // 初始猜测
    }

    for (iter = 0; iter < MAX_ITER; iter++) {
        for (i = rank; i < n; i += size) {
            double sum = 0.0;
            for (j = 0; j < n; j++) {
                if (j != i) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
        }

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x_new, n/size, MPI_DOUBLE, MPI_COMM_WORLD);

        // Check convergence
        double norm = 0.0;
        for (i = 0; i < n; i++) {
            norm += fabs(x_new[i] - x[i]);
            x[i] = x_new[i];
        }

        double global_norm;
        MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (global_norm < TOL) {
            break;
        }
    }

    free(x_new);
}

void read_input(const char *filename, int *n, double ***A, double **b) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d", n);

    *A = (double **)malloc(*n * sizeof(double *));
    for (int i = 0; i < *n; i++) {
        (*A)[i] = (double *)malloc(*n * sizeof(double));
    }

    *b = (double *)malloc(*n * sizeof(double));

    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *n; j++) {
            fscanf(file, "%lf", &(*A)[i][j]);
        }
    }

    for (int i = 0; i < *n; i++) {
        fscanf(file, "%lf", &(*b)[i]);
    }

    fclose(file);
}

void write_output(const char *filename, int n, double *x) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        fprintf(file, "x[%d] = %f\n", i, x[i]);
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n;
    double **A;
    double *b;
    double *x;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        read_input(argv[1], &n, &A, &b);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        A = (double **)malloc(n * sizeof(double *));
        for (int i = 0; i < n; i++) {
            A[i] = (double *)malloc(n * sizeof(double));
        }
        b = (double *)malloc(n * sizeof(double));
    }

    for (int i = 0; i < n; i++) {
        MPI_Bcast(A[i], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    x = (double *)malloc(n * sizeof(double));

    gauss_jacobi(n, A, b, x, rank, size);

    if (rank == 0) {
        write_output(argv[2], n, x);
    }

    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
    free(b);
    free(x);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
