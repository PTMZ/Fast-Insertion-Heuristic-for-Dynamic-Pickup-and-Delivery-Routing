import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle

img = cv2.imread('road2_edit.png')

mask = cv2.inRange(img, np.array([100, 100, 100]), np.array([255, 255, 255]))

print(mask.shape)
print(np.sum(mask) // 255)
# 120k points

cv2.imwrite("temp.png", mask)

# Extract list of (y,x) coordinates
h, w = mask.shape
points = [(y,x) for y in range(h) for x in range(w) if mask[y,x] == 255]

# Run Kmeans
print('Running Kmeans...')
N = 900
kmeans = KMeans(n_clusters=N, random_state=0).fit(points)

centers = kmeans.cluster_centers_
centers = [(int(c[1]), int(c[0])) for c in centers]

# Save clusters
with open('centroids2.pkl', 'wb') as f:
    pickle.dump(centers, f)

t2 = np.array(img)
for i,c in enumerate(centers):
    t2 = cv2.putText(t2, f'{i}', c, cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
    t2 = cv2.circle(t2, c, 5, [0,0,255], 2)

cv2.imwrite("r2.png", t2)










