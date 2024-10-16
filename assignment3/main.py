import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data from CSV
data = pd.read_csv('linear_regression_data.csv', header=None)

A = data[0].values  # Independent variable
B = data[1].values  # Dependent variable


# function to compute the slope and intercept
def linear_regression(A, B):
    n = len(A)

    #  mean of A and B
    mean_a = np.mean(A)
    mean_b = np.mean(B)

    # covariance and variance
    covariance_ab = np.sum((A - mean_a) * (B - mean_b))
    variance_a = np.sum((A - mean_a)**2)

    # slope (m) and intercept (b)
    m = covariance_ab / variance_a
    b = mean_b - (m * mean_a)

    return m, b


# slope and intercept
slope, intercept = linear_regression(A, B)

print(f'Linear model: B = {slope:.2f}A + {intercept:.2f}')

# plot the data points
plt.scatter(A, B, color='blue', label='Data points')

# plot the regression line
B_pred = slope * A + intercept
plt.plot(A, B_pred, color='red', label='Regression line')

plt.xlabel('A - Independent Variable')
plt.ylabel('B - Dependent Variable')
plt.title('Linear Regression ')
plt.legend()

plt.show()
