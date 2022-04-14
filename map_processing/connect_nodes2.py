import cv2
import numpy as np
import pickle

img = cv2.imread('road2_edit.png')
mask = cv2.inRange(img, np.array([127, 127, 127]), np.array([255, 255, 255]))

# Load clusters
with open("centroids2.pkl", 'rb') as f:
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
threshold = 50
for i in range(N):
    for j in range(i+1, N):
        x1, y1 = centers[i]
        x2, y2 = centers[j]
        if (y1-y2)*(y1-y2) + (x1-x2)*(x1-x2) < threshold * threshold:
            add_edge(adj_list, i, j)

# Delete edges checking pixels along the edge
for i in range(N):
    to_delete_list = []
    for j in adj_list[i]:
        x1, y1 = centers[i]
        x2, y2 = centers[j]
        cW, cB = 0, 0
        if x1 != x2:
            grad = (y2-y1) / (x2-x1)
            step = 1 if x2 > x1 else -1
            for curX in range(x1, x2, step):
                curY = int(y1 + (curX - x1) * grad)
                if mask[curY, curX] > 0:
                    cW += 1
                else:
                    cB += 1
        else:
            step = 1 if y2 > y1 else -1
            for curY in range(y1, y2, step):
                curX = x1
                if mask[curY, curX] > 0:
                    cW += 1
                else:
                    cB += 1
        
        if cW/(cW+cB) < 0.2:
            to_delete_list.append((i,j))
    
    for a, b in to_delete_list:
        if len(adj_list[a])>=3 and len(adj_list[b])>=3:
            print((a,b))
            del_edge(adj_list, a, b)

# Manual deletion / addition of edges
del_pairs = []

add_pairs = [(83,345), (214, 368), (583,39), (742,119), (597, 753), (564, 315), (288, 181), (7, 484),
            (395,854), (758, 485), (515,333), (184,447), (557,434), (328,587), (114,595), (552, 655),
            (276,322), (831,391), (246,590), (162,798), (524,469), (162,618), (69,458), (582,598),
            (224,302)]

for p1, p2 in del_pairs:
    del_edge(adj_list, p1, p2)

for p1,p2 in add_pairs:
    add_edge(adj_list, p1, p2)

t2 = np.array(img)
for i,c in enumerate(centers):
    color = [0,0,255]
    if len(adj_list[i]) <= 1:
        color = [0,255,0]
    t2 = cv2.putText(t2, f'{i}', c, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    t2 = cv2.circle(t2, c, 5, [0,0,255], 2)

for s_idx in range(N):
    for e_idx in adj_list[s_idx]:
        t2 = cv2.line(t2, centers[s_idx], centers[e_idx], [200,0,0], 2)

cv2.imwrite("r3.png", t2)

# Save adj_graph
with open('adj2.pkl', 'wb') as f:
    pickle.dump(adj_list, f)
