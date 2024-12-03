import numpy as np

def generate_input_file(filename,solution_filename, n):
    # Generate a random n x n matrix A with values between 1 and 10
    A = np.random.rand(n, n)
    
    # Make A diagonally dominant to ensure convergence
    for i in range(n):
        A[i, i] = sum(np.abs(A[i])) + 1

    # Generate a known solution vector x_true
    x_true = np.random.rand(n)
    
    # Calculate the corresponding right-hand side vector b
    b = np.dot(A, x_true)
    
    # Write the matrix and vector to the file
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in A:
            f.write(' '.join(map(str, row)) + '\n')
        for value in b:
            f.write(f"{value}\n")

    # Write the known solution vector x_true to the solution file
    with open(solution_filename, 'w') as f:
        for value in x_true:
            f.write(f"{value}\n")

# Example usage:
generate_input_file('input.txt','solution.txt', 100)
  
