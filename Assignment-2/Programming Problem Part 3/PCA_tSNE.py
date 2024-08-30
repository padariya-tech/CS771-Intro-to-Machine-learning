import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#load the mnist_small.pkl file
data = np.load('mnist_small.pkl', encoding='bytes', allow_pickle=True)
X = data['X']
Y = data['Y']

# Ensure Y is one-dimensional
Y = np.squeeze(Y)
# Apply PCA to project the data into 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)


# Apply t-SNE to project the data into 2D
tSNE = TSNE(n_components=2, random_state=42)
tSNE_result = tSNE.fit_transform(X)

# Visualize PCA results
for i in range(10):
    ind = (Y == i)
    plt.scatter(pca_result[ind, 0], pca_result[ind, 1], label=str(i))
plt.title('PCA Visualization for projected data in 2D')
plt.legend()
plt.show()

# Visualize t-SNE results
for i in range(10):
    ind = (Y == i)
    plt.scatter(tSNE_result[ind, 0], tSNE_result[ind, 1], label=str(i))
plt.title('t-SNE Visualization for projected data in 2D')
plt.legend()
plt.show()