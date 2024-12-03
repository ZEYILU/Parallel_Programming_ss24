#include <stdio.h>
#include <stdlib.h>

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void quicksort(int* array, int low, int high) {
    if (low < high) {
        int pivot = array[high];
        int i = low - 1; 
        for (int j = low; j < high; j++) {
            if (array[j] < pivot) {
                i++;
                swap(&array[i], &array[j]);
            }
        }
        swap(&array[i + 1], &array[high]);
        int pi = i + 1;

        quicksort(array, low, pi - 1);
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

    quicksort(array, 0, n - 1);

    file = fopen("sorted.txt", "w");
    for (int i = 0; i < n; i++) {
        fprintf(file, "%d\n", array[i]);
    }
    fclose(file);

    free(array);
    return 0;
}
