#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
	int j;

	#pragma omp parallel for schedule(dynamic,10)
	for (j = 0; j < 100; j++)
 	printf("iteration %3d is handled by thread #%2d\n", j, omp_get_thread_num());

	return 0;
 }
