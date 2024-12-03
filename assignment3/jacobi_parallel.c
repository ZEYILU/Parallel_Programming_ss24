/*
 *  OpenMP lecture exercises
 *  Copyright (C) 2011 by Christian Terboven <terboven@rz.rwth-aachen.de>
 *  Modifications (C) 2022 by Kai Diethelm <kai.diethelm@fhws.de>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define U(i,j) u[(i)*n+(j)]
#define UOLD(i,j) uold[(i)*n+(j)]
#define F(i,j) f[(i)*n+(j)]
/*****************************************************************
* Subroutine jacobi
* Solves Helmholtz equation on rectangular grid assuming
* (1) uniform discretization in each direction, and
* (2) Dirichlet boundary conditions
*
* Jacobi method is used in this routine
*
* Input : n,m    Number of grid points in the X/Y directions
*         dx,dy  Grid spacing in the X/Y directions
*         alpha  Helmholtz eqn. coefficient
*         omega  Relaxation factor
*         f(n,m) Right hand side function
*         u(n,m) Dependent variable/solution
*         tol    Tolerance for iterative solver
*         maxit  Maximum number of iterations
*
* Output : u(n,m) - Solution
*****************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void jacobi (int n, int m, double dx, double dy,
             double alpha, double omega, double *u, double *f,
             double tol, int maxit)
{
  int    i,j,k;
  double error, resid, ax, ay, b;
  double *uold;

  uold = (double *) malloc(sizeof(double)*n*m);
  if (!uold){
    fprintf(stderr, "Error: cant allocate memory\n");
    exit(1);
  }

  ax =  1.0/(dx * dx); /* X-direction coef */
  ay =  1.0/(dy * dy); /* Y_direction coef */
  b  = -2.0/(dx * dx) - 2.0/(dy * dy) - alpha; /* Central coeff */

  error = 10.0 * tol;

  k = 1;

  while (k <= maxit && error > tol) {
    error = 0.0;
    /* copy new solution into old */
    #pragma omp parallel for private(i)
    for (j=0; j<m; j++)
      for (i=0; i<n; i++){
        UOLD(j,i) = U(j,i);
      }

    /* compute stencil, residual and update */
    #pragma omp parallel for private(i, resid) reduction(+:error)
    for (j=1; j<m-1; j++){
      for (i=1; i<n-1; i++){
        resid =(
                ax * (UOLD(j,i-1) + UOLD(j,i+1))
                + ay * (UOLD(j-1,i) + UOLD(j+1,i))
                + b * UOLD(j,i) - F(j,i)
                ) / b;

        /* update solution */
        U(j,i) = UOLD(j,i) - omega * resid;

        /* accumulate residual error */
        error = error + resid * resid;
      }
    }
    /* error check */
    k++;
    error = sqrt(error) / (n * m);
  } /* while */
  printf("Total Number of Iterations %d\n", k);
  printf("Residual                   %.15g\n", error);
}


/************************************************************
* program to solve a finite difference
* discretization of Helmholtz equation:
* (d2/dx2)u + (d2/dy2)u - alpha u = f
* using Jacobi iterative method.
*
* Input :  n     - grid dimension in x direction
*          m     - grid dimension in y direction
*          alpha - Helmholtz constant (always greater than 0.0)
*          tol   - error tolerance for iterative solver
*          relax - Successive over relaxation parameter
*          mits  - Maximum iterations for iterative solver
*
* On output
*       : u(n,m) - Dependent variable (solutions)
*       : f(n,m) - Right hand side function
**************************************************************/

/******************************************************
* Initializes data
* Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
******************************************************/
void initialize(int n, int m, double alpha,
                double *dx, double *dy, double *u, double *f)
{
  int i, j, xx, yy;

  *dx = 2.0 / (n-1);
  *dy = 2.0 / (m-1);

  /* Initialize initial condition and RHS */
  for (j=0; j<m; j++){
    for (i=0; i<n; i++){
      xx = -1.0 + *dx * (i-1);
      yy = -1.0 + *dy * (j-1);
      U(j,i) = 0.0;
      F(j,i) = -alpha * (1.0 - xx*xx) * (1.0 - yy*yy)
                - 2.0 * (1.0 - xx*xx) - 2.0 * (1.0 - yy*yy);
    }
  }
}

/************************************************************
* Checks error between numerical and exact solution
************************************************************/
void error_check(int n, int m, double alpha,
                 double dx, double dy, double *u, double *f)
{
  int i,j;
  double xx, yy, temp, error;


  dx = 2.0 / (n-1);
  dy = 2.0 / (m-1);
  error = 0.0;

  #pragma omp parallel for private(i, xx, yy, temp) reduction(+:error)
  for (j=0; j<m; j++){
    for (i=0; i<n; i++){
      xx = -1.0 + dx * (i-1);
      yy = -1.0 + dy * (j-1);
      temp = U(j,i) - (1.0 - xx*xx) * (1.0 - yy*yy);
      error += temp*temp;
    }
  }
  error = sqrt(error) / (n*m);

  printf("Solution Error : %g\n", error);
}


int main(int argc, char* argv[])
{
    double *u, *f, dx, dy;
    double r1;
    int    n, m, mits;
    double tol, relax, alpha;

    /* Read info */
    // printf("Input n, m - grid dimension in x, y direction :\n");
    // scanf("%d,%d", &n, &m);
    // printf("Input alpha - Helmholtz constant :\n");
    // scanf("%lf", &alpha);
    // printf("Input relax - Successive over-relaxation parameter:\n");
    // scanf("%lf", &relax);
    // printf("Input tol - error tolerance for iterative solver:\n");
    // scanf("%lf", &tol);
    // printf("Input mits - Maximum iterations for solver:\n");
    // scanf("%d", &mits);
    n = atoi(argv[1]);
    m = atoi(argv[2]);
    alpha = atof(argv[3]);
    relax = atof(argv[4]);
    tol = atof(argv[5]);
    mits = atoi(argv[6]);
    printf("-> %d, %d, %f, %f, %f, %d\n", n, m, alpha, relax, tol, mits);

    u = (double *) malloc(n*m*sizeof(double));
    f = (double *) malloc(n*m*sizeof(double));

    if (!u || !f){
      fprintf(stderr, "Error: Can't allocate memory\n");
      exit(1);
    }

    /* arrays are allocated and initialized */
    initialize(n, m, alpha, &dx, &dy, u, f);

    /* Solve Helmholtz equation */
    r1 = omp_get_wtime();

    jacobi(n, m, dx, dy, alpha, relax, u, f, tol, mits);

    r1 = omp_get_wtime() - r1;

    printf(" elapsed time : %12.6f\n", r1);
    printf(" MFlops       : %12.6g\n",
           mits*(m-2)*(n-2)*0.000001*13 / r1);

    error_check(n, m, alpha, dx, dy, u, f);

    return 0;
}
