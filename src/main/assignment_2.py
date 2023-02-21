import numpy as np
import math

np.set_printoptions(precision=7, suppress=True, linewidth=100)

def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            # skip if value is prefilled (we dont want to accidentally recalculate...)
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            
            # get left cell entry
            left: float = matrix[i][j-1]

            # get diagonal left entry
            diagonal_left: float = matrix[i-1][j-1]

            # order of numerator is SPECIFIC.
            numerator: float = left - diagonal_left
            
            # denominator is current i's x_val minus the starting i's x_val....
            denominator = matrix[i][0] - matrix[i-(j-1)][0]

            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation
    
    return matrix


def hermite_interpolation(x_points, y_points, slopes):
    # matrix size changes because of "doubling" up info for hermite 
    num_of_points = len(x_points)
    matrix = np.zeros((2 * num_of_points, 2 * num_of_points))

    # populate x values (make sure to fill every TWO rows)
    for x in range(0, 2 * num_of_points):
        matrix[x][0] = x_points[math.floor(x/2)]
    
    # prepopulate y values (make sure to fill every TWO rows)
    for x in range(0, 2 * num_of_points):
        matrix[x][1] = y_points[math.floor(x/2)]

    # prepopulate with derivates (make sure to fill every TWO rows. starting row CHANGES.)
    matrix[0][2] = 0
    for x in range(0, num_of_points):
        matrix[2*x+1][2] = slopes[x]

    filled_matrix = apply_div_dif(matrix)
    return filled_matrix

def divided_difference_table(x_points, y_points):
    # set up the matrix
    size: int = len(x_points)
    matrix: np.array = np.zeros((size, size))

    # fill the matrix
    for index, row in enumerate(matrix):
        row[0] = y_points[index]

    # populate the matrix (end points are based on matrix size and max operations we're using)
    for i in range(1, size):
        for j in range(1, i + 1):
            # the numerator are the immediate left and diagonal left indices...
            numerator = matrix[i][j-1] - matrix[i-1][j-1]

            # the denominator is the X-SPAN...
            denominator = x_points[i] - x_points[i-j]

            operation = numerator / denominator

            # cut it off to view it more simpler
            matrix[i][j] = '{0:.7g}'.format(operation)


    # print(matrix)
    return matrix

def nevilles_method(x_points, y_points, x, degree):
    # must specify the matrix size (this is based on how many columns/rows you want)
    num_of_points = len(x_points)
    matrix = np.zeros((num_of_points, num_of_points))

    # fill in value (just the y values because we already have x set)
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]

    # populate final matrix (this is the iterative version of the recursion explained in class)
    # the end of the second loop is based on the first loop...
    for i in range(1, num_of_points):
        for j in range(1, i+1):
            first_multiplication = (x - x_points[i-j]) * matrix[i][j-1]
            second_multiplication = (x - x_points[i]) * matrix[i-1][j-1]

            denominator = x_points[i] - x_points[i-j]

            # this is the value that we will find in the matrix
            coefficient = (first_multiplication - second_multiplication) / denominator
            matrix[i][j] = coefficient
            
#    for i in range(0, num_of_points):
#        for j in range(0, i+1):
#            print(matrix[i][j], end=" ")
#        print()

    return matrix[num_of_points-1][degree]

def get_approximate_result(matrix, x_points, value):
    # p0 is always y0 and we use a reoccuring x to avoid having to recalculate x 
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]
    
    # we only need the diagonals...and that starts at the first row...
    for index in range(1, len(x_points)):
        polynomial_coefficient = matrix[index][index]
        # we use the previous index for x_points....
        reoccuring_x_span *= (value - x_points[index-1])
        
        # get a_of_x * the x_span
        mult_operation = polynomial_coefficient * reoccuring_x_span
        # add the reoccuring px result
        reoccuring_px_result += mult_operation
    
    # final result
    return reoccuring_px_result

if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True, linewidth=100)
    
    # Q1
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    x = 3.7
    degree = 2
    nevilles_result = nevilles_method(x_points, y_points, x, degree)
    print(nevilles_result, end="\n\n")

    # Q2
    x_points = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    #x_points = [1, 1.3, 1.6, 1.9, 2.2]
    #y_points = [0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623]
    ddmatrix = divided_difference_table(x_points, y_points)
    print("[", end="")
    for i in range(1, len(x_points) - 1):
       print(ddmatrix[i][i], end=", ")
    print(ddmatrix[len(x_points)-1][len(x_points)-1], end="]\n\n")
    
    #Q3
    value = 7.3
    newtons_result = get_approximate_result(ddmatrix, x_points, value)
    print(newtons_result, end="\n\n")

    #Q4
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
    hermite_matrix = hermite_interpolation(x_points, y_points, slopes)
    print(hermite_matrix, end="\n\n")
