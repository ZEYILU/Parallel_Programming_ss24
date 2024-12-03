#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int my_rank, n_ranks, MPIerror;
    
    MPI_Init(&argc, &argv);
    MPIerror = MPI_comm_rank(MPI_COMM-WORLD, &my_rank);
    MPIerror = MPI_comm_size(MPI_COMM-WORLD, &n_ranks);
    printf("Hello, World from Process #%d\n", my_rank);
    if(my_rank == 0)
        printf("total number of processes: %d\n", n_ranks);


    // 释放 MPI 的一些资源
    MPI_Finalize();
    return 0;
}