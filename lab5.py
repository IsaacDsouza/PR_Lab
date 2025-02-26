import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from fcmeans import FCM
from sklearn.datasets import load_wine

X = StandardScaler().fit_transform(load_wine().data[:, :2])
fcm = FCM(n_clusters=3)
fcm.fit(X)  # Ensure fitting before prediction

labels, centers = fcm.predict(X), fcm.centers

for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i+1}')
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200, label='Centers')
plt.legend(), plt.title("Fuzzy C-Means Clustering"), plt.show()
