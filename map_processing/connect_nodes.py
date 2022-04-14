import cv2
import numpy as np
import pickle

img = cv2.imread('road1_edit.png')
mask = cv2.inRange(img, np.array([127, 127, 127]), np.array([255, 255, 255]))

# Load clusters
with open("centroids.pkl", 'rb') as f:
    centers = pickle.load(f)
N = len(centers)

# Adj_list
adj_list = [set() for _ in range(N)]

# Add / Delete Bidirectional edge function
def add_edge(adj_list, i, j):
    adj_list[i].add(j)
    adj_list[j].add(i)

def del_edge(adj_list, i, j):
    adj_list[i].remove(j)
    adj_list[j].remove(i)

# compute pairwaise distances between all pairs of centers
threshold = 160
for i in range(N):
    for j in range(i+1, N):
        x1, y1 = centers[i]
        x2, y2 = centers[j]
        if (y1-y2)*(y1-y2) + (x1-x2)*(x1-x2) < threshold * threshold:
            add_edge(adj_list, i, j)

# Manual deletion / addition of edges
del_pairs = [(29,61), (29,64), (1,32), (1,40), (46,74), (40,74), (1,66), (41,25), (31,92), (31,63), (80,11), (80,43),
            (75,15), (2,99), (2,79), (2,36), (36,44), (36,12), (90,56), (7,68), (60,5), (60,59), (65,50), 
            (65,78), (65,82), (0,13), (50,78), (13,53), (13,96), (45,53), (45,96), (47,82), (52,93), (37,34), (67,98),
            (20,8), (20,95), (95,8), (95,89), (90,44)]

add_pairs = [(24, 45), (44,12)]

for p1, p2 in del_pairs:
    del_edge(adj_list, p1, p2)

for p1,p2 in add_pairs:
    add_edge(adj_list, p1, p2)


t2 = np.array(img)
for i,c in enumerate(centers):
    t2 = cv2.putText(t2, f'{i}', c, cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
    t2 = cv2.circle(t2, c, 5, [0,0,255], 2)

for s_idx in range(N):
    for e_idx in adj_list[s_idx]:
        t2 = cv2.line(t2, centers[s_idx], centers[e_idx], [0,0,255], 2)

cv2.imwrite("t3.png", t2)

# Save adj_graph
with open('adj1.pkl', 'wb') as f:
    pickle.dump(adj_list, f)


