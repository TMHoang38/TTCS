# Code 4: PCA từ đầu
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SkPCA
from sklearn.datasets import load_iris


class PCA:
    '''Principal Component Analysis từ đầu'''
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None   # Eigenvectors
        self.explained_variance_ = None  # Eigenvalues
        self.mean_ = None

    def fit(self, X):
        # Bước 1: Chuẩn hóa dữ liệu (centering).
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Bước 2: Tính ma trận hiệp phương sai
        n = X.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n - 1)

        # Bước 3: Phân rã trị riêng, không quan tâm vecto âm hay dương (eigenvalue decomposition)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Bước 4: Sắp xếp và chọn các thành phần chính quan trọng nhất dựa trên phương sai giải thích.
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Bước 5: Chọn k components đầu tiên
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]

        # Tính tỷ lệ phương sai giải thích
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self
    #Chiếu dữ liệu từ không gian đặc trưng ban đầu xuống không gian có số chiều thấp hơn bằng cách nhân ma trận dữ liệu đã chuẩn hóa với các vector riêng đã chọn.
    def transform(self, X):
        '''Project data onto principal components'''
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Custom PCA
my_pca = PCA(n_components=2)
X_my = my_pca.fit_transform(X)

# Sklearn PCA
sk_pca = SkPCA(n_components=2)
X_sk = sk_pca.fit_transform(X)

print('Explained Variance Ratio:')
print(f' Custom: {my_pca.explained_variance_ratio_}')
print(f' Sklearn: {sk_pca.explained_variance_ratio_}')

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

scatter1 = axes[0].scatter(X_my[:, 0], X_my[:, 1],
                           c=y, cmap='viridis', alpha=0.7)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_title('Custom PCA')

scatter2 = axes[1].scatter(X_sk[:, 0], X_sk[:, 1],
                           c=y, cmap='viridis', alpha=0.7)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('Sklearn PCA')

plt.colorbar(scatter2, ax=axes[1], label='Class')
plt.tight_layout()
plt.show()

# Cumulative variance
print(f'\nTotal variance explained (2 PCs): {sum(my_pca.explained_variance_ratio_):.4f}')
