

# ------- Classes and methods --------
# Job:
# - id
# - s_time
# - e_time
# - lat
# - lon
# - demand

# Order:
# - id
# - pickJob: job
# - dropJob: job

# Route:
# - order_set: set of job ids in the route
# - job_seq: list of Job objects
# - isFeasible(): returns Boolean if current path is feasible
# ---> Capacity constraint: Compute cumulative demand at each point < Max_cap
# - calcCost(): returns Float/Integer for given path
# ---> Compute Lateness



# Solution:
# route_list: list of Route objects
# obj_list: list of obj values for each route
# total_obj: sum of obj_list
# total_veh: len of route_list
# printSol(): methods to display solution



# ------- Initial Solution Generation ----------
# Option 1: Add each order to a unique vehicle
# Option 2: Add all orders to a single vehicle


# -------- Destroy operator: Random removal ---------
# Option 1: Remove K random orders out of all orders in the existing solution
# Option 2: Weighted removal based on Cost function


# -------- Repair operator: Sequential Cheapest insertion ----------
def greedyInsertion(order, existingRoutes):
    best_route_idx = None
    best_route_cost = None
    new_route = None
    for r_idx, r in enumerate(existingRoutes):
        curBestRoute = copy(r)
        curBestCost = curBestRoute.calcCost()
        for i,j in (pairs of pointsOfInsertion):
            candidate_route = copy(r)
            candidate_route.insert(i, order.pick)
            candidate_route.insert(j, order.drop)
            if candidate_route.isFeasible():
                candidate_cost = candidate_route.calcCost()
                if candidate_cost < curBestCost:
                    curBestRoute = copy(candidate_route)
                    curBestCost = candidateCost
                    if best_route_cost is None or curBestCost < best_route_cost:
                        best_route_idx = r_idx
                        best_route_cost = curBestCost
                        new_route = copy(candidate_route)
    
    r = GetBestRoute()
    # Beam Search top K routes
    r_list = GetBestRoutes()
    empty_route = Route()
    
    Obj = W_v * Num_veh + W_l * Lateness


    for r in r_list:
        for i,j in (pairs of pointsOfInsertion in route r):
            candidate_route = copy(r)
            candidate_route.insert(i, order.pick)
            candidate_route.insert(j, order.drop)
            if candidate_route.isFeasible():
                candidate_cost = candidate_route.calcCost()
                if candidate_cost < curBestCost:
                    curBestRoute = copy(candidate_route)
                    curBestCost = candidateCost
                    if best_route_cost is None or curBestCost < best_route_cost:
                        best_route_idx = r_idx
                        best_route_cost = curBestCost
                        new_route = copy(candidate_route)




# Orders are added sequentially until a solution is obtained
# To obtain a ground truth, we can assume all orders are known beforehand and compute
# the best solution found over different sequences of the orders and running 
# the LNS algorithm multiple times. There are other ways to modify this to SA / Tabu or
# adding in the logic of Beam Search for our greedy insertion.


# -------- Dynamic Order Insertion at time > 0 ----------
# Attempt insertion without destroying
# If cost increases substantially, need to attempt to reorder unvisited orders.
# Start with set of all orders in the existing routes, S
# order_pick_start: dict of (key = order_id, value = pick_start_time)
# When performing random removal, check if order_pick_start[order_idx] <= cur_time - buffer_time
#   If so, remove this order from S
# Random sample from S for destroy operator


# ---------- Reordering within Routes ---------------
# Due to the greedy sequential insertion, it is possible that improvements can be found
# by swapping positions of jobs within each route.
#
# def optimiseRoute(route):
#     best_cost = route.calcCost()
#     best_route = route
#     for i,j in (pairs of jobs in route):
#         candidate_route = copy(route)
#         candidate_route.swap(i,j)
#         if candidate_route.isFeasible():
#             candidate_cost = candidate_route.calcCost()
#             if candidate_cost < best_cost:
#                 best_cost = candidate_cost
#                 best_route = copy(candidate_route)
# 
# Run this optimisation at the end of repairing?







