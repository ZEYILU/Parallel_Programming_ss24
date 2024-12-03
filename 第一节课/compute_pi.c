#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    int i, num_steps = 100000;
    double stepsize, x, pi, a, sum = 0.0;

    if (argc > 1) num_steps = atoi(argv[1]);
    stepsize = 1.0 / (double) num_steps;
    
    #pragma omp parallel for private(x)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * stepsize;
        sum = sum +  4.0 / (1.0 + x * x);
    } 
    pi = stepsize * sum;
    printf("Approx. value with %d steps: pi = %12.8f\n", num_steps, pi);

    return 0;
}

