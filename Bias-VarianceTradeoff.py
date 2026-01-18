# Code 3: Bias-Variance Tradeoff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Hàm thực sự
def true_function(x):
    return np.sin(2 * np.pi * x)

# Sinh dữ liệu huấn luyện
def generate_data(n=100, noise=0.3):
    x = np.random.uniform(0, 1, n)
    y = true_function(x) + np.random.normal(0, noise, n)
    return x, y

# Phân tích Bias và Variance
def bias_variance_analysis(degree, n_experiments=100):
    x_test = np.linspace(0, 1, 100)
    y_true = true_function(x_test)
    predictions = []

    for _ in range(n_experiments):
        x_train, y_train = generate_data()
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x_train.reshape(-1, 1), y_train)
        y_pred = model.predict(x_test.reshape(-1, 1))
        predictions.append(y_pred)

    predictions = np.array(predictions)
    y_pred_mean = predictions.mean(axis=0)  # E[f̂(x)]

    # Bias²
    bias_sq = np.mean((y_pred_mean - y_true) ** 2)


    # Variance
    variance = np.mean(predictions.var(axis=0))
    return bias_sq, variance

# Thí nghiệm với nhiều bậc đa thức
degrees = range(1, 15)
biases, variances = [], []

for d in degrees:
    b, v = bias_variance_analysis(d)
    biases.append(b)
    variances.append(v)

# Vẽ đồ thị
plt.figure(figsize=(10, 6))
plt.plot(degrees, biases, 'b-o', label='Bias²')
plt.plot(degrees, variances, 'r-o', label='Variance')
plt.plot(degrees, np.array(biases) + np.array(variances), 'g--', label='Bias² + Variance')
plt.xlabel('Polynomial Degree (Model Complexity)')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.grid(True)
plt.show()
