#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <sstream>
#include <string>
#include <algorithm>  // For std::remove_if

void readMatrix(const std::string& filename, std::vector<std::vector<double>>& matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    std::string line;
    int lineCount = 0;  // To help identify problematic lines
    while (getline(file, line)) {
        lineCount++;
        std::istringstream iss(line);
        std::vector<double> row;
        std::string value;
        while (getline(iss, value, ',')) {
            value.erase(std::remove_if(value.begin(), value.end(), isspace), value.end());  // Remove all spaces
            try {
                double num = std::stod(value);
                row.push_back(num);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Conversion error on line " << lineCount << ": " << value << " could not be converted to double." << std::endl;
                continue;
            }
        }
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }
    file.close();
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
        std::cerr << "Usage: " << argv[0] << " <matrixA.txt> <matrixB.txt>\n";
        return 1;
    }
    
    std::vector<std::vector<double>> A, B, C_seq, C_par, Diff;
    readMatrix(argv[1], A);
    readMatrix(argv[2], B);
    
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        std::cerr << "Error: Invalid matrix dimensions.\n";
        return 1;
    }
    
    int m = A.size(), n = A[0].size(), q = B[0].size();
    double time_seq, time_par;

    matrixProduct(m, n, q, A, B, C_seq, C_par, Diff, time_seq, time_par);

    
    std::cout << "Matrix C (Sequential): \n";
    for (const auto& row : C_seq) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Matrix C (Parallel): \n";
    for (const auto& row : C_seq) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
    // std::cout << "Difference Matrix: \n";
    // for (const auto& row : Diff) {
    //     for (double val : row) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << "\n";
    // }
    double maxVal = std::numeric_limits<double>::lowest(); // Initialize to smallest possible double
    double minVal = std::numeric_limits<double>::max();    // Initialize to largest possible double

    for (const auto& row : Diff) {
        for (double val : row) {
            if (val > maxVal) maxVal = val;
            if (val < minVal) minVal = val;
        }
    }
    std::cout << "Largest coefficient in the difference matrix: " << maxVal << std::endl;
    std::cout << "Smallest coefficient in the difference matrix: " << minVal << std::endl;

    std::cout << "Sequential Time: " << time_seq << " seconds\n";
    std::cout << "Parallel Time: " << time_par << " seconds\n";
    return 0;
}
