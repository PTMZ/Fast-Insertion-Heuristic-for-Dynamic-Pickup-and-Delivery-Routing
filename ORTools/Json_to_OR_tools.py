import pickle
import numpy as np
import json

# ----------  Helper Funcs  ----------------
# Load final lat_long + time_matrix
with open('final_points.pkl', 'rb') as f:
    final_points = pickle.load(f)

with open('final_matrix.pkl', 'rb') as f:
    final_time = pickle.load(f)
    final_time += 60

def euclid_dist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def approx(p, lst):
    # p is a tuple(x, y), lst is a list of tuples
    # The function returns index of point in lst closest to p -> p -> Input Point & Matrix List
    
    best = 1e10
    ans = 0
    for i, p2 in enumerate(lst):
        dist =  euclid_dist(p, p2)
        
        if dist < best:
            best = dist
            ans = i
    
    return ans

# default file 'waypoints_order.json'

# ------------   Json func  -----------------
def json2order(f):
    all_orders = []
    with open(f) as file:
        data = json.loads(file.read())
        #print(data)
        for test, subdata in data.items():
            for key_, initial_orders in subdata.items():
                for order in initial_orders:
                    current_order = []
                    #print(order)
                    order_id = (order['order_id'])
                    job_type = (order['job_type'])
                    date = (order['date'])
                    latitude = (order['latitude'])
                    #print('lat', latitude)
                    longitude = (order['longitude'])
                    #print('long', longitude)
                    route = (order['route'])
                    #print(route)
                    sequence = (order['sequence'])
                    window = (order['window'])
                    capacity = (order['capacity'])  
                    current_order.extend([job_type, date, latitude, longitude, route, sequence, window, capacity])
                    #print(current_order)
                    all_orders.append(current_order)
    return all_orders


# ----------------  Ortools func  ---------------------
class Node:
    def __init__(self, x, y, tw, demand):
        self.x = x
        self.y = y
        self.tw = tw
        self.demand = demand

def orders2Ortools(all_orders, depot_lat = 1.35, depot_lon = 103.7, depot_tw_str = "00:00-23:59", depot_demand = 0):
    node_list = []
    node_list.append(Node(depot_lat, depot_lon, depot_tw_str, depot_demand))

    # list of tuples containing (coor, tw)
    #['DROP', '2022-02-28', '1.3748425', '103.7399456', 10688, 3, '18:00-20:00', 160]
    for task_type, date, lat, lon, job_id, seq, tw_str, demand in all_orders:
        node_list.append(Node(float(lat), float(lon), tw_str, demand))

    N = len(node_list)

    orders_time_matrix = np.zeros((N, N))

    approx_list = [approx((n.x,n.y), final_points) for n in node_list]

    #time_matrix
    for i in range(N):
        for j in range(N):
            orders_time_matrix[i][j] = final_time[approx_list[i]][approx_list[j]] / 60

    # pickup_deliveries
    cur_pick_id = 0
    cur_row = 1
    pick_drop_list = []
    for task_type, date, lat, lon, job_id, seq, tw_str, demand in all_orders:
        if task_type == 'PICK':
            cur_pick_id = cur_row
        else:
            pick_drop_list.append((cur_pick_id, cur_row))
        cur_row += 1

    # time_windows
    def str2timeval(string):
        st_str, et_str = string.split('-')
        st_hr,st_min = st_str.split(':')
        st = int(st_hr) * 60 + int(st_min)
        et_hr,et_min = et_str.split(':')
        et = int(et_hr) * 60 + int(et_min)
        return (st, et)

    time_window_list = [str2timeval(n.tw) for n in node_list]
    #print(time_window_list)

    # demands
    demand_list = [0]
    for p,d in pick_drop_list:
        demand_list.append(node_list[d].demand)
        demand_list.append(-1 * node_list[d].demand)
    #print(demand_list)
    return orders_time_matrix, pick_drop_list, time_window_list, demand_list


if __name__ == '__main__':
    filename = "waypoints_order.json"
    all_orders = json2order(filename)
    # depot_lat = 1.35, depot_lon = 103.7, depot_tw_str = "00:00-23:59", depot_demand = 0 is default specified within orders2Ortools
    orders_time_matrix, pick_drop_list, time_window_list, demand_list = orders2Ortools(all_orders)

    


