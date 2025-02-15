import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#generarate 10 linearly seperable (tiesiskai atskiriami) points
#samples = tasku kiekis, features = pozymiai, centers = tasku grupiu kiekis
X, y = make_blobs(n_samples=20, n_features=2, centers = 2, random_state=38)
# y = which class point belongs to
# X = samples

plt.scatter(X[:, 0], X[:, 1], c=y)      #x[row,column]
plt.title("Dvi tasku klases")
plt.show()
