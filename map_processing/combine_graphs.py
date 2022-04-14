import cv2
import numpy as np
import pickle
from convert import yx2latlong

def euclid_dist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def approx(p, lst):
    # p is a tuple(x,y), lst is a list of tuples
    # returns index of point in lst closest to p
    best = 1e10
    ans = 0
    for i,p2 in enumerate(lst):
        dist =  euclid_dist(p, p2)
        if dist < best:
            best = dist
            ans = i
    
    return ans

def floyd_warshall(points, adj_list):
    N = len(adj_list)
    dist = [[float("inf")]*N for _ in range(N)]
    for i in range(N):
        for j in  adj_list[i]:
            dist[i][j] = euclid_dist(points[i], points[j])
    for i in range(N):
        dist[i][i] = 0
    
    for k in range(N):
        for i in range(N):
            for j in range(i+1,N):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    dist[j][i] = dist[i][k] + dist[k][j]
    
    return dist
                

img1 = cv2.imread('road1_edit.png')
mask1 = cv2.inRange(img1, np.array([127, 127, 127]), np.array([255, 255, 255]))

img2 = cv2.imread('road2_edit.png')
mask2 = cv2.inRange(img2, np.array([127, 127, 127]), np.array([255, 255, 255]))

full_img = mask1 + mask2
full_img[full_img > 255] = 255

full_img = np.stack((full_img, full_img, full_img), axis=2)
print(full_img.shape)

cv2.imwrite("f.png", full_img)

# Load clusters
with open("centroids.pkl", 'rb') as f:
    highway_points = pickle.load(f)
N1 = len(highway_points)

with open("centroids2.pkl", 'rb') as f:
    road_points = pickle.load(f)
N2 = len(road_points)

# Load Adj Lists
with open("adj1.pkl", 'rb') as f:
    adj1 = pickle.load(f)

with open("adj2.pkl", 'rb') as f:
    adj2 = pickle.load(f)

# Get closest highway node for each road node
approx_lst = [approx(p,highway_points) for p in road_points]

# Compute shortest distance from every pair of nodes in both road / highway graphs
print('Computing FW...')
pix_dist_highways = floyd_warshall(highway_points, adj1)
pix_dist_roads = floyd_warshall(road_points, adj2)

# Conversion of pix to meters 70 px / km
pix2meters = 1000/70

# Speed params: 60km/hr for highways, 40 km/hr for roads, coverted to m/s
highway_spd = 60 * 1000 / 3600
road_spd = 40 * 1000 / 3600

# Compute time matrices for both graphs
time_highways = np.array(pix_dist_highways) * pix2meters / highway_spd
time_roads = np.array(pix_dist_roads) * pix2meters / road_spd

print(time_roads[326][30]/60)

# Compute final time matrix considering road-highway-road path
print("Computing final time...")
final_time = np.array(time_roads)
for i in range(N2):
    for j in range(i+1, N2):
        h1 = approx_lst[i]
        h2 = approx_lst[j]
        d1 = euclid_dist(road_points[i], highway_points[h1])
        d2 = euclid_dist(road_points[j], highway_points[h2])
        t1 = d1 * pix2meters / road_spd
        t2 = d2 * pix2meters / road_spd
        rhr_time = time_highways[approx_lst[i]][approx_lst[j]] + t1 + t2
        final_time[i][j] = min(time_roads[i][j], rhr_time)
        final_time[j][i] = min(time_roads[i][j], rhr_time)


# Convert yx2latlong
final_points = [yx2latlong(p[1],p[0]) for p in road_points]

# Save final lat_long + time_matrix
with open('final_points.pkl', 'wb') as f:
    pickle.dump(final_points, f)

with open('final_matrix.pkl', 'wb') as f:
    pickle.dump(final_time, f)

print(final_time[846][522]/60)
