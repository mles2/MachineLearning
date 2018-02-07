import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.cm as cm

digits = load_digits()
data = digits.data

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

print(labels)


pca = PCA(n_components=10)
data_r = pca.fit(data).transform(data)

lda = LDA(n_components=2)
data_r2 = lda.fit(data, labels).transform(data)

print('wspolczynnik wyjasnionych wariancji (10 pierwszych skladowych): %s' % str(pca.explained_variance_ratio_))
print('suma wyjasnionych wariancji (10 pierwszych skladowych): %s' % str(sum(pca.explained_variance_ratio_)))

x = np.arange(2)
ys = [i+x+(i*x)**2 for i in range(10)]

plt.figure()
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
for c, i, target_name in zip(colors, [1,2,3,4,5,6,7,8,9,10], labels):
    plt.scatter(data_r[labels == i, 0], data_r[labels == i, 1], c=c, alpha = 0.4)
    plt.legend()
    plt.title('Wykres przedstawiajacy punkty \n' 'opisywane przez 10 glownych skladowych')
plt.show()