#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ROWS 100
#define MAX_COLS 100

int main(int argc, char *argv[]) {
    float matrix1[MAX_ROWS][MAX_COLS] = {
        {1, 2, 3.4, 5, -3, -6.4},
        {4, 2, -2.4, -5, -0.001, 0.3}
    };

    float matrix2[MAX_ROWS][MAX_COLS] = {
        {1, 2, 3.4, 5, -3, -6.4},
        {-3, 5, 6, -1.111, 3.453e2, -123},
        {23, -4.444, 1.2, 61.2, -87.4, -0.000003},
        {4, 2, -2.4, -5, -0.001, 0.3}
    };

    float matrix3[MAX_ROWS][MAX_COLS] = {
        {1, 2, 3.4},
        {-3, 5, 6},
        {23, -4.444, 1.2},
        {4, 2, -2.4}
    };

    int rows1 = 2, cols1 = 6;
    int rows2 = 4, cols2 = 6;
    int rows3 = 4, cols3 = 3;

    float a11_1 = matrix1[0][0];
    float a11_2 = matrix2[0][0];
    float a11_3 = matrix3[0][0];

    float lastRowSum1 = 0, lastRowSum2 = 0, lastRowSum3 = 0;
    for (int j = 0; j < cols1; j++) {
        lastRowSum1 += fabs(matrix1[rows1 - 1][j]);
    }
    for (int j = 0; j < cols2; j++) {
        lastRowSum2 += fabs(matrix2[rows2 - 1][j]);
    }
    for (int j = 0; j < cols3; j++) {
        lastRowSum3 += fabs(matrix3[rows3 - 1][j]);
    }

    float maxRowSum1 = 0, maxRowSum2 = 0, maxRowSum3 = 0;
    for (int i = 0; i < rows1; i++) {
        float rowSum = 0;
        for (int j = 0; j < cols1; j++) {
            rowSum += matrix1[i][j];
        }
        if (i == 0 || rowSum > maxRowSum1) {
            maxRowSum1 = rowSum;
        }
    }
    for (int i = 0; i < rows2; i++) {
        float rowSum = 0;
        for (int j = 0; j < cols2; j++) {
            rowSum += matrix2[i][j];
        }
        if (i == 0 || rowSum > maxRowSum2) {
            maxRowSum2 = rowSum;
        }
    }
    for (int i = 0; i < rows3; i++) {
        float rowSum = 0;
        for (int j = 0; j < cols3; j++) {
            rowSum += matrix3[i][j];
        }
        if (i == 0 || rowSum > maxRowSum3) {
            maxRowSum3 = rowSum;
        }
    }

    printf("Matrix 1:\n");
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols1; j++) {
            printf("%.2f\t", matrix1[i][j]);
        }
        printf("\n");
    }
    printf("Coefficient a1,1 for Matrix 1: %.2f\n", a11_1);
    printf("Sum of absolute values of last row for Matrix 1: %.2f\n", lastRowSum1);
    printf("Maximum sum of each row for Matrix 1: %.2f\n\n", maxRowSum1);

    printf("Matrix 2:\n");
    for (int i = 0; i < rows2; i++) {
        for (int j = 0; j < cols2; j++) {
            printf("%.2f\t", matrix2[i][j]);
        }
        printf("\n");
    }
    printf("Coefficient a1,1 for Matrix 2: %.2f\n", a11_2);
    printf("Sum of absolute values of last row for Matrix 2: %.2f\n", lastRowSum2);
    printf("Maximum sum of each row for Matrix 2: %.2f\n\n", maxRowSum2);

    printf("Matrix 3:\n");
    for (int i = 0; i < rows3; i++) {
        for (int j = 0; j < cols3; j++) {
            printf("%.2f\t", matrix3[i][j]);
        }
        printf("\n");
    }
    printf("Coefficient a1,1 for Matrix 3: %.2f\n", a11_3);
    printf("Sum of absolute values of last row for Matrix 3: %.2f\n", lastRowSum3);
    printf("Maximum sum of each row for Matrix 3: %.2f\n", maxRowSum3);

    return 0;
}
