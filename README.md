## README

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




