#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 10000
#define TOL 1e-5

void read_matrix(const char *filename, double ***A, double **b, int *n);
void write_solution(const char *filename, double *x, int n);

int main(int argc, char *argv[]) {
    int rank, size, n, iter;
    double **A, *b, *x, *x_old, *local_x;
    double local_diff, global_diff;
    
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        read_matrix(argv[1], &A, &b, &n);
        x = (double*) malloc(n * sizeof(double));
        x_old = (double*) malloc(n * sizeof(double));
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        b = (double*) malloc(n * sizeof(double));
        x = (double*) malloc(n * sizeof(double));
        x_old = (double*) malloc(n * sizeof(double));
    }

    local_x = (double*) malloc(n * sizeof(double));

    MPI_Bcast(b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        A = (double**) malloc(n * sizeof(double*));
        for (int i = 0; i < n; i++)
            A[i] = (double*) malloc(n * sizeof(double));
    }

    for (int i = 0; i < n; i++)
        MPI_Bcast(A[i], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < n; i++)
        x[i] = 0.0;

    iter = 0;
    do {
        for (int i = 0; i < n; i++)
            x_old[i] = x[i];

        for (int i = rank; i < n; i += size) {
            local_x[i] = b[i];
            for (int j = 0; j < n; j++) {
                if (i != j)
                    local_x[i] -= A[i][j] * x_old[j];
            }
            local_x[i] /= A[i][i];
        }

        MPI_Allreduce(local_x, x, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        local_diff = 0.0;
        for (int i = rank; i < n; i += size) {
            local_diff += fabs(x[i] - x_old[i]);
        }
        
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        iter++;
    } while (global_diff > TOL && iter < MAX_ITER);

    if (rank == 0) {
        write_solution(argv[2], x, n);
        free(b);
        free(x);
        free(x_old);
        for (int i = 0; i < n; i++)
            free(A[i]);
        free(A);
    }

    free(local_x);
    MPI_Finalize();
    return 0;
}

void read_matrix(const char *filename, double ***A, double **b, int *n) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d", n);

    *A = (double**) malloc(*n * sizeof(double*));
    for (int i = 0; i < *n; i++)
        (*A)[i] = (double*) malloc(*n * sizeof(double));

    *b = (double*) malloc(*n * sizeof(double));

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


void write_solution(const char *filename, double *x, int n) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        fprintf(file, "%lf\n", x[i]);
    }

    fclose(file);
}
