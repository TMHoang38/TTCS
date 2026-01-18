import numpy as np
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler

class StandardScaler:
    '''Chuẩn hóa dữ liệu: z = (x - μ) / σ'''
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        '''Tính mean và std từ dữ liệu'''
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)
        return self

    def transform(self, X):
        '''Áp dụng chuẩn hóa'''
        return (X - self.mean_) / (self.std_ + 1e-8)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        '''Chuyển về giá trị gốc'''
        return X_scaled * self.std_ + self.mean_
class MinMaxScaler:
    '''Chuẩn hóa Min-Max: x_norm = (x - min) / (max - min)'''
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.min_) / (self.max_ - self.min_ + 1e-8)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
# Code 2: So sánh Feature Scaling


# Tạo dữ liệu mẫu
np.random.seed(42)
X = np.random.randn(100, 3) * [10, 100, 1000] + [5, 50, 500]

print('Dữ liệu gốc:')
print(f' Mean: {X.mean(axis = 0)}')
print(f' Std: {X.std(axis = 0)}')

# Chuẩn hóa bằng StandardScaler tự viết
my_scaler = StandardScaler()
X_my = my_scaler.fit_transform(X)

# Chuẩn hóa bằng StandardScaler của Scikit-learn
sk_scaler = SkStandardScaler()
X_sk = sk_scaler.fit_transform(X)

print('\nSau Standardization (Custom):')
print(f' Mean: {X_my.mean(axis=0)}')
print(f' Std: {X_my.std(axis=0)}')

print('\nSau Standardization (Sklearn):')
print(f' Mean: {X_sk.mean(axis=0)}')
print(f' Std: {X_sk.std(axis=0)}')

# Kiểm tra sự khác biệt
print(f'\nMax difference: {np.abs(X_my - X_sk).max():.10f}')
