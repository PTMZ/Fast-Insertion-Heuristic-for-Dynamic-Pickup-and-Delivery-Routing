# Fast Insertion Heuristic for Capacitated Pickup and Delivery Problem with Time Windows (CPDPTW)
### Problem Description
Our problem is the vehicle routing problem with pickup and delivery jobs within time window and 
capacity. We assume a homogenous fleet of vehicles with same speed and capacity.
A solution consists of a list of vehicles. Each vehicle contains a route which is the sequence of nodes it 
travels to, a maximum capacity, and the list of jobs it fulfils by going through the route. The number 
of vehicles for a solution is not bounded by a hard constraint and is modelled as a soft constraint in 
our objective function. A vehicle’s route starts from and ends at the depot node, and the pick and drop 
nodes which correspond to the same job have a precedence constraint where the pick node is to be 
visited before the drop node.

A job consists of 2 nodes, a pick node and a drop node. A node consists of a (x,y) pair representing the 
location’s latitude and longitude, a (start_time, end_time) which represent the time window for the 
pick / drop task, and the demand for the pick / drop task which represents the change in capacity 
when a vehicle reaches that node. A customer order according to the JSON input could comprise a
pick task followed by multiple drop tasks. We simplify this order into multiple jobs by splitting the pick 
task by the number of drop tasks in the order.

In our problem, the hard constraints are on order fulfilment, i.e. all customer orders must be fulfilled, 
as well as on capacity, i.e. the capacity of a vehicle must not be violated when items are picked up. 
The soft constraints are on the time window for the arrival of the vehicles, as well as the fixed cost 
incurred whenever a vehicle is deployed. Our aim is to minimise the sum of the tardiness cost 
(Tardiness = max(0, Arrival_time – Node.end_time) for each node) and the vehicle fixed cost. 

## NOTE
Input data file not uploaded due to data confidentiality.

## How to Run
To run `alns_main.py` file, run the following with seed 606:
```
python alns_main.py 606
```

To run `dynamic_insertion.py` file, run the following with seed 606:
```
python dynamic_insertion.py
```
This file will run the insertion of 10 additional jobs (Pick and Drop) into an initial route produced by Google ORtools containing 50 jobs.

The insertions make the assumption that the requests come at the same time as the jobs, Pickup start time.

You can see the outputs under `ORtools_full_config.json_v1.txt` for the inital route and `Dyn_small_full_config.json_v1.txt` for the route after insertions.

## Parameters

```
# First K jobs to extract from the full_config.json file
CONCAT_ORDERS = 50

# Objective cost weight of adding a vehicle
VEHICLE_PENALTY = 60

# Output file prefix
OUTFILE_PREFIX = "Small"

# Initial route threshold to use
INIT_ROUTE_LENGTH = 20

# Amount of increment to route threshold per unsuccesful insertion iteration
ADJ_ROUTE_LENGTH = 2

# Amount of increment to max allowable tardinessfor each customer per unsuccesful insertion iteration
ADJ_TARDINESS = 5

# Boolean true to enable adjustments, false to disable
BOOL_ADJ = True

# Boolean to enable route threshold to be used
ENABLE_ROUTE_THRESHOLD = False
```




