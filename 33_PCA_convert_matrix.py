from numpy import array
from sklearn.decomposition import PCA
# uy= ax1 +bx2 +cx3 +d
A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print(A)
pca = PCA(2) #????
pca.fit(A)
print("components\n",pca.components_)
print("variance\n", pca.explained_variance_)
print("ratio\n",pca.explained_variance_ratio_)
B = pca.transform(A)
print(B)