import copy
from enum import unique
from multiprocessing.connection import wait
import numpy as np
import xml.etree.ElementTree as ET
import random
import json
import pickle

import sys
sys.path.append('./ALNS')
from alns import ALNS, State

# ------------ Load Map time matrix and points ---------------

with open('final_points.pkl', 'rb') as f:
    MAP_POINTS = pickle.load(f)

with open('final_matrix.pkl', 'rb') as f:
    MAP_TIME_MATRIX = pickle.load(f)
    MAP_TIME_MATRIX = MAP_TIME_MATRIX / 60
    # MAP_TIME_MATRIX = MAP_TIME_MATRIX.astype(int)

# --------------- Helper Functions --------------------------

def euclid_dist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def approx(p, lst):
    best = 1e10
    ans = 0
    for i, p2 in enumerate(lst):
        dist =  euclid_dist(p, p2)
        if dist < best:
            best = dist
            ans = i
    
    return ans

def get_time_taken(A,B, time_matrix):
    return time_matrix[A.tm_idx][B.tm_idx]

def json2data(f):
    all_orders = []
    with open(f) as file:
        data = json.loads(file.read())['data']
        depotData = data['depotData']
        vehicleData = data['vehicleData']
        all_orders = data['getWindowAllocationReport']

    return depotData, vehicleData, all_orders

def str2timeval(string):
    st_str, et_str = string.split('-')
    st_hr,st_min = st_str.split(':')
    st = int(st_hr) * 60 + int(st_min)
    et_hr,et_min = et_str.split(':')
    et = int(et_hr) * 60 + int(et_min)
    return (st, et)

def orders2jobs(orders):
    cur_pick = None
    pick_drop_list = []
    unique_id = 0
    last_pick = 1
    ortools2node_dict = dict()
    for row_num, o in enumerate(orders):
        st,et = str2timeval(o["window"])
        unique_id += 1
        id_str = f"J{unique_id}"
        if o["job_type"] == 'PICK':
            last_pick = row_num + 1
            cur_pick = Customer(0, 1, float(o["latitude"]), float(o["longitude"]), demand=0, start_time=st, end_time=et)
        else:
            P = copy.deepcopy(cur_pick)
            P.id = id_str
            P.demand = o["capacity"]
            D = Customer(id_str, 1, float(o["latitude"]), float(o["longitude"]), demand=-P.demand, start_time=st, end_time=et)
            pick_drop_list.append(Job(id_str, P, D))
            if last_pick in ortools2node_dict.keys():
                pass
                #print("WARNING: DOES NOT WORK WITH PICK DROP DROP")
            ortools2node_dict[last_pick] = P
            ortools2node_dict[row_num + 1] = D

    
    return pick_drop_list, ortools2node_dict

# -------------------------------------

class Parser(object):
    
    def __init__(self, json_file, CONCAT=0):
        '''initialize the parser
        Args:
            xml_file::str
                the path to the json file
        '''
        self.name = json_file
        depotData, vehicleData, jobData = json2data(json_file)
        self.depot = Depot(0, 0, float(depotData["latitude"]), float(depotData["longitude"]))
        self.vehicle = Vehicle(0, self.depot, self.depot, max_travel_time=vehicleData['max_time'], max_capacity=vehicleData['max_capacity'])
        self.jobs, self.ortool2node = orders2jobs(jobData)
        if CONCAT > 0:
            self.jobs = self.jobs[:CONCAT]


### Node class ###
class Node(object):
    
    def __init__(self, id, type, x, y):
        '''Initialize a node
        Args:
            id::int
                id of the node
            type::int
                0 for depot, 1 for customer, 2 for charging station
            x::float
                x coordinate of the node
            y::float
                y coordinate of the node
        '''
        self.id = id
        self.type = type
        self.x = x
        self.y = y
        self.tm_idx = approx((self.x, self.y), MAP_POINTS)
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id and self.type == other.type and self.x == other.x and self.y == other.y
        return False
        
    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}'.format(self.id, self.type, self.x, self.y)

### Depot class ###
class Depot(Node):
    
    def __init__(self, id, type, x, y):
        '''Initialize a depot
        Args:
            id::int
                id of the depot
            type::int
                0 for depot
            x::float
                x coordinate of the depot
            y::float
                y coordinate of the depot
        '''
        super(Depot, self).__init__(id, type, x, y)
        
### Customer class ###
# You should not change this class!
class Customer(Node):
    
    def __init__(self, id, type, x, y, demand, start_time=0, end_time=1440, service_time=0, ):
        '''Initialize a customer
        Args:
            id::int
                id of the customer
            type::int
                1 for customer
            x::float
                x coordinate of the customer
            y::float
                y coordinate of the customer
            service_time::float
                service time of the customer
        '''
        super(Customer, self).__init__(id, type, x, y)
        self.service_time = service_time
        # Earliest delivery time
        self.start_time = start_time
        # Latest delivery time
        self.end_time = end_time

        self.demand = demand
        
    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}, demand: {}'.format(self.id, self.type, self.x, self.y, self.demand)


class Job:
    def __init__(self, id, P, D):
        self.id = id
        self.P = P
        self.D = D
    
    def __repr__(self):
        return f"[id:{self.id}, P:{self.P}, D:{self.D}]"

### Vehicle class ###
# Vehicle class. You could add your own helper functions freely to the class, and not required to use the functions defined
# But please keep the rest untouched!
class Vehicle(object):
    
    def __init__(self, id, start_node, end_node, max_travel_time=180, max_capacity=2000):
        ''' Initialize the vehicle
        Args:
            id::int
                id of the vehicle
            start_node::Node
                starting node of the vehicle
            end_node::Node
                ending node of the vehicle
            max_travel_time::float
                maximum time allowed for the vehicle (including travel and charging time) (h)
            max_capacity
        '''
        self.id = id
        self.start_time = 0
        self.start_node = start_node
        self.end_node = end_node
        self.max_travel_time = max_travel_time
        # travel time of the vehicle
        self.travel_time = 0
        self.wait_time = 0
        self.cur_time = 0
        # all the nodes including depot, customers, or charging stations (if any) visited by the vehicle
        self.node_visited = [self.start_node] # start from depot
        # time visited at each node
        self.node_visited_time = []
        self.tardy = 0
        self.tardy_list = []
        self.current_capacity = 0
        self.max_capacity = max_capacity
        self.cap_feasible = True
        self.jobs = []

    def reset(self):
        self.travel_time = 0
        self.start_time = 0
        self.wait_time = 0
        self.node_visited = [self.start_node]
        self.node_visited_time = []
        self.tardy = 0
        self.tardy_list = []
        self.current_capacity = 0
        self.cap_feasible = True
        self.route = None

    def run_route(self, route, jobs):
        ''' route: [Node] containing [C1, C2, ..., CN]
        '''
        self.route = route
        self.jobs = jobs
        first_node = route[0]
        time_taken = get_time_taken(self.start_node, first_node, MAP_TIME_MATRIX)
        self.start_time = first_node.start_time - time_taken
        self.cur_time = self.start_time
        self.node_visited_time.append(self.start_time)

        for v in route:
            self.move_A_B(self.node_visited[-1], v)
        
        self.move_A_B(self.node_visited[-1], self.end_node)

        return self.travel_time, self.wait_time, self.tardy

    def move_A_B(self, A, B):
        ''' A and B are Nodes
        '''
        time_taken = get_time_taken(A,B, MAP_TIME_MATRIX)
        self.travel_time += time_taken
        self.cur_time += time_taken
        self.node_visited.append(B)
        self.node_visited_time.append(self.cur_time)
        wait_time_amt = 0
        tardy_amt = 0
        if isinstance(B, Customer):
            wait_time_amt = max(B.start_time - self.cur_time, 0)
            tardy_amt = max(self.cur_time - B.end_time, 0)
            self.wait_time += wait_time_amt
            self.cur_time += wait_time_amt
            self.tardy += tardy_amt
            self.tardy_list.append(tardy_amt)
            self.travel_time += B.service_time
            self.cur_time += B.service_time
            self.current_capacity += B.demand
            if self.current_capacity > self.max_capacity:
                self.cap_feasible = False

        
    def check_time(self):
        '''Check whether the vehicle's travel time  is over the maximum travel time or not
        Return True if it is not over the maximum travel time, False otherwise
        '''
        if self.travel_time + self.wait_time <= self.max_travel_time:
            return True
        return False

    def check_capacity(self):
        return self.cap_feasible
    
    def check_return(self):
        ''' Check whether the vehicle's return to the depot
        Return True if returned, False otherwise
        '''
        if len(self.node_visited) > 1:
            return self.node_visited[-1] == self.end_node
        return False
            
    def __str__(self):
        return 'Vehicle id: {}, start_node: {}, end_node: {}, max_travel_time: {}, speed_factor: {}'\
            .format(self.id, self.start_node, self.end_node, self.max_travel_time, self.speed_factor)

### EVRP state class ###
# EVRP state class. You could and should add your own helper functions to the class
# But please keep the rest untouched!
class CPDPTW(State):
    
    def __init__(self, name, depot, jobs, vehicle, vehicle_cost=60):
        '''Initialize the EVRP state
        Args:
            name::str
                name of the instance
            depot::Depot
                depot of the instance
            jobs::[Jobs]
                jobs of the instance
            vehicle::Vehicle
                vehicle of the instance
        '''
        self.name = name
        self.depot = depot
        self.jobs = jobs
        self.vehicle = vehicle
        # record the vehicle used
        self.vehicles = []
        # total travel time of the all the vehicle used
        self.travel_time = 0
        # record the all the customers who have been visited by all the vehicles, eg. [Customer1, Customer2, ..., Customer7, Customer8]
        # self.job_visited = []
        # record the unvisited customers, eg. [Customer9, Customer10]
        self.job_unvisited = []
        # the route visited by each vehicle, eg. [vehicle1.node_visited, vehicle2.node_visited, ..., vehicleN.node_visited]
        self.route = []
        # Weightage for objective function to penalise addition of vehicles
        self.vehicle_cost = vehicle_cost

        self.adjustment = 0
                    
    def random_initialize(self, seed=None):
        ''' Randomly initialize the state with split_route() (your construction heuristic)
        Args:
            seed::int
                random seed
        Returns:
            objective::float
                objective value of the state
        '''
        if seed is not None:
            random.seed(606)
        random_tour = copy.deepcopy(self.jobs)
        random.shuffle(random_tour)
        self.split_route(random_tour)
        return self.objective()
    
    def copy(self):
        return copy.deepcopy(self)
    
    def split_route(self, tour):
        '''Generate the route given a tour visiting all the customers
        Args:
            tour::[Customer]
                a tour visiting all the customers
        
        # You should update the following variables for the EVRP
        CPDPTW.vehicles
        CPDPTW.travel_time
        CPDPTW.job_visited
        CPDPTW.job_unvisited
        CPDPTW.route
        
        # You should update the following variables for each vehicle used
        Vehicle.travel_time
        Vehicle.wait_time
        Vehicle.node_visited
        '''
        # You should implement your own method to construct the route of EVRP from any tour visiting all the customers
        
        # Naive starting point: Assign each job to a vehicle
        
        for job in tour:
            new_vehicle = copy.deepcopy(self.vehicle)
            new_vehicle.id = len(self.vehicles)
            new_vehicle.run_route([job.P, job.D], [job])
            if not (new_vehicle.check_time() and new_vehicle.check_capacity()):
                print(f"Job {job} is not possible with a single vehicle!")
            self.vehicles.append(new_vehicle)
        

        # Update EVRP values
        self.update()
        # self.travel_time = sum(v.travel_time for v in self.vehicles) + sum(v.wait_time for v in self.vehicles)
        # self.tardiness = sum(v.tardy for v in self.vehicles)
        # # for n in tour:
        # #     self.job_visited.append(n)
        # #     self.job_unvisited.remove(n)
        # self.route = [v.node_visited for v in self.vehicles]
    
    def update(self):
        self.travel_time = sum(v.travel_time for v in self.vehicles) + sum(v.wait_time for v in self.vehicles)
        self.tardiness = sum(v.tardy for v in self.vehicles)
        self.route = [v.node_visited for v in self.vehicles]
        self.jobs = sum([v.jobs for v in self.vehicles], [])

    def objective(self):
        ''' Calculate the objective value of the state
        Return the total travel time and charging time of all vehicles used
        '''
        # or return sum([v.travel_time for v in self.vehicles]) + sum([v.charging_time for v in self.vehicles])
        self.travel_time = sum(v.travel_time for v in self.vehicles) + sum(v.wait_time for v in self.vehicles)
        self.tardiness = sum(v.tardy for v in self.vehicles)        
        return self.tardiness + len(self.vehicles) * self.vehicle_cost