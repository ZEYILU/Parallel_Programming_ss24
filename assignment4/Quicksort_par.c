#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// swap two elements
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Parallelized Quicksort function
void quicksort(int* array, int low, int high) {
    if (low < high) {
        int pivot = array[high]; // Choose the pivot element
        int i = low - 1; // Index of the smaller element

        // Parallelize the partitioning process
        #pragma omp parallel for shared(array, pivot) private(i)
        for (int j = low; j < high; j++) {
            if (array[j] < pivot) {
                {
                    i++;
                    swap(&array[i], &array[j]);
                }
            }
        }

        
        swap(&array[i + 1], &array[high]);
        int pi = i + 1; 

        // Parallelize the recursive calls to sort the subarrays
        #pragma omp task shared(array)
        quicksort(array, low, pi - 1);

        #pragma omp task shared(array)
        quicksort(array, pi + 1, high);
    }
}

int main() {
    int n = 1000000;
    int* array = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        array[i] = rand() % 100000;
    }

    FILE* file = fopen("unsorted.txt", "w");
    for (int i = 0; i < n; i++) {
        fprintf(file, "%d\n", array[i]);
    }
    fclose(file);

    // Parallel sort the array
    #pragma omp parallel
    {
        quicksort(array, 0, n - 1);
    }

    // Write the sorted array to a file
    file = fopen("sorted.txt", "w");
    for (int i = 0; i < n; i++) {
        fprintf(file, "%d\n", array[i]);
    }
    fclose(file);

    free(array);
    return 0;
}
