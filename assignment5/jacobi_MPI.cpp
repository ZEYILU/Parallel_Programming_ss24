#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

// Function to read matrix and vector from file
void readMatrix(const std::string &filename, std::vector<std::vector<double>> &matrix, std::vector<double> &b) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int n;
    infile >> n;
    matrix.resize(n, std::vector<double>(n));
    b.resize(n);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            infile >> matrix[i][j];
        }
        infile >> b[i];
    }
}

// Function to write results to file
void writeResult(const std::string &filename, const std::vector<double> &x) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    for (const auto &val : x) {
        outfile << val << std::endl;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <input file> <output file>" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    
    std::vector<std::vector<double>> A;
    std::vector<double> b, x, x_new;
    
    if (rank == 0) {
        readMatrix(inputFile, A, b);
        int n = A.size();
        x.resize(n, 0.0);
        x_new.resize(n, 0.0);
    }
    
    int n;
    if (rank == 0) {
        n = A.size();
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        A.resize(n, std::vector<double>(n));
        b.resize(n);
        x.resize(n, 0.0);
        x_new.resize(n, 0.0);
    }
    
    for (int i = 0; i < n; ++i) {
        MPI_Bcast(A[i].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    int chunk_size = (n + size - 1) / size;
    int start = rank * chunk_size;
    int end = std::min(n, (rank + 1) * chunk_size);
    
    const int max_iterations = 1000;
    const double tolerance = 1e-6;
    bool converged = false;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        double local_diff = 0.0;
        for (int i = start; i < end; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
            local_diff += std::abs(x_new[i] - x[i]);
        }
        
        double global_diff;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        MPI_Allgather(x_new.data() + start, chunk_size, MPI_DOUBLE, x.data(), chunk_size, MPI_DOUBLE, MPI_COMM_WORLD);
        
        if (global_diff < tolerance) {
            converged = true;
        }
        
        MPI_Bcast(&converged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (converged) break;

        std::swap(x, x_new);
    }
    
    if (rank == 0) {
        writeResult(outputFile, x);
    }
    
    MPI_Finalize();
    return 0;
}
