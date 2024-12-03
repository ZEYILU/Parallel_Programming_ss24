#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <sstream>
#include <string>
#include <algorithm> 
#include "mmio.h" 

// Functions that use the MatrixMarket I/O library to read matrices
bool readMatrixMarketFile(const char* filename, std::vector<std::vector<double>>& matrix) {
    FILE* file;
    MM_typecode matcode;
    int M, N, nz;  

    if ((file = fopen(filename, "r")) == NULL) {
        return false;
    }

    // Read the banner of MatrixMarket
    if (mm_read_banner(file, &matcode) != 0) {
        std::cerr << "Could not process Matrix Market banner." << std::endl;
        fclose(file);
        return false;
    }

    if ((mm_is_matrix(matcode) && mm_is_dense(matcode) && mm_is_array(matcode)) && mm_is_real(matcode)) {
        // Read the size information
        if (mm_read_mtx_array_size(file, &M, &N) != 0) {
            fclose(file);
            return false;
        }

        matrix.resize(M, std::vector<double>(N));

        // Read the matrix data
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                double value;
                if (fscanf(file, "%lg", &value) != 1) {
                    fclose(file);
                    return false;
                }
                matrix[i][j] = value;
            }
        }
    } else {
        std::cerr << "This application does not support this Matrix Market format." << std::endl;
        fclose(file);
        return false;
    }

    fclose(file);
    return true;
}



void matrixProduct(int m, int n, int q, const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b,
                   std::vector<std::vector<double>>& cs, std::vector<std::vector<double>>& cp, std::vector<std::vector<double>>& dc,
                   double& ts, double& tp) {
    cs.resize(m, std::vector<double>(q, 0));
    cp.resize(m, std::vector<double>(q, 0));
    dc.resize(m, std::vector<double>(q, 0));
    
    // Sequential matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < q; j++) {
            for (int k = 0; k < n; k++) {
                cs[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    ts = std::chrono::duration<double>(end - start).count();

    // Parallel matrix multiplication using OpenMP
    start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < q; j++) {
            for (int k = 0; k < n; k++) {
                cp[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    tp = std::chrono::duration<double>(end - start).count();

    // Compute the difference
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < q; j++) {
            dc[i][j] = cs[i][j] - cp[i][j];
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [Matrix Market file A] [Matrix Market file B]" << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> A, B;
    if (!readMatrixMarketFile(argv[1], A) || !readMatrixMarketFile(argv[2], B)) {
        std::cerr << "Failed to read Matrix Market files." << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> C_seq, C_par, C_diff;
    double ts, tp;
    matrixProduct(A.size(), A[0].size(), B[0].size(), A, B, C_seq, C_par, C_diff, ts, tp);

    double maxVal = std::numeric_limits<double>::lowest(); // Initialize to smallest possible double
    double minVal = std::numeric_limits<double>::max();    // Initialize to largest possible double

    for (const auto& row : C_diff) {
        for (double val : row) {
            if (val > maxVal) maxVal = val;
            if (val < minVal) minVal = val;
        }
    }
    std::cout << "Largest coefficient in the difference matrix: " << maxVal << std::endl;
    std::cout << "Smallest coefficient in the difference matrix: " << minVal << std::endl;

    std::cout << "Sequential Time: " << ts << " seconds\n";
    std::cout << "Parallel Time: " << tp << " seconds\n";

    return 0;
}