
import argparse
import numpy as np
import numpy.random as rnd
import networkx as nx
import matplotlib.pyplot as plt
from lxml import etree as LET

from cpdptw import *
from ORToolsSampleOutputs import *
from pathlib import Path
from time import time

import sys
sys.path.append('./ALNS')
from alns import ALNS, State
from alns.criteria import HillClimbing, SimulatedAnnealing, RecordToRecordTravel

CONCAT_ORDERS = 0
VEHICLE_PENALTY = 60
INIT_ROUTE_LENGTH = 90
ADJ_ROUTE_LENGTH = 5
ADJ_TARDINESS = 2
BOOL_ADJ = False
ENABLE_ROUTE_THRESHOLD = False

# Comment out line 29 to test with time_metric
def sub_heuristic(A, B):
    W = 2500
    #return (A.x-B.x)**2 + (A.y-B.y)**2
    return (A.x-B.x)**2 + (A.y-B.y)**2 + ((A.start_time - B.start_time)/W)**2

def approx_vehicle(route, job):
    N = len(route)
    dist_drop = [sub_heuristic(job.D, n) for n in route]
    for i in range(N-1):
        dist_drop[N-2-i] = min(dist_drop[N-2-i], dist_drop[N-1-i])
    dist_pick = [sub_heuristic(job.P, n) for n in route]
    ans = min(a+b for a,b in zip(dist_pick, dist_drop))
    return ans


def convertORtoolOutput(route, jobs, ortools2node):
    route_pd_nodes = [ortools2node[i] for i in route[1:]]
    route_job_ids = set(n.id for n in route_pd_nodes)
    id2job = {j.id : j for j in jobs}
    route_jobs = [id2job[i] for i in route_job_ids]

    return route_pd_nodes, route_jobs

### draw and output solution ###
def save_output(YourName, route, suffix):
    '''Draw the EVRP instance and save the solution
    Args:
        YourName::str
            your name, eg. John_Doe
        evrp::EVRP
            an EVRP object
        suffix::str
            suffix of the output file, 
            'initial' for random initialization
            and 'solution' for the final solution
    '''
    generate_output(YourName, route, suffix)
    draw_route(YourName, route, suffix)

### visualize EVRP ###
def create_graph(routing):
    '''Create a directional graph from the EVRP instance
    Args:
        evrp::EVRP
            an EVRP object
    Returns:
        g::nx.DiGraph
            a directed graph
    '''
    g = nx.DiGraph(directed=True)
    g.add_node(routing.depot.id, pos=(routing.depot.x-1, routing.depot.y-103), type=routing.depot.type)
    for job in routing.jobs:
        p,d = job.P, job.D
        g.add_node("P"+p.id, pos=(p.x-1, p.y-103), type=p.type)
        g.add_node("D"+d.id, pos=(d.x-1, d.y-103), type=d.type)

    return g

def draw_route(YourName, routing, suffix):
    '''Draw the EVRP instance and the solution
    Args:
        YourName::str
            your name, eg. John_Doe
        evrp::EVRP
            an EVRP object
        suffix::str
            suffix of the output file, 
            eg. 'initial' for random initialization
            and 'solution' for the final solution
    '''
    g = create_graph(routing)
    def proc_id(node):
        if node.type == 0:
            return 0
        return ("P" if node.demand>0 else "D") + node.id
    
    route = list(proc_id(node) for node in sum(routing.route, []))
    
    edges = [(route[i], route[i+1]) for i in range(len(route) - 1) if route[i] != route[i+1]]
    g.add_edges_from(edges)
    colors = []
    
    for n in g.nodes:
        if g.nodes[n]['type'] == 0:
            colors.append('#0000FF')
        elif g.nodes[n]['type'] == 1:
            colors.append('#FF0000')
        else:
            colors.append('#00FF00')
    pos = nx.get_node_attributes(g, 'pos')
    fig, ax = plt.subplots(figsize=(24, 12))
    nx.draw(g, pos, node_color=colors, with_labels=True, ax=ax, 
            arrows=True, arrowstyle='-|>', arrowsize=12, 
            connectionstyle='arc3, rad = 0.025')

    # plt.text(0, 6, YourName, fontsize=12)
    # plt.text(0, 3, 'Instance: {}'.format(routing.name), fontsize=12)
    # plt.text(0, 0, 'Objective: {}'.format(routing.objective()), fontsize=12)
    plt.savefig('{}_{}_{}.jpg'.format(YourName, routing.name, suffix), dpi=300, bbox_inches='tight')
    
### generate output file for the solution ###
def generate_output(YourName, routing, suffix):
    '''Generate output file (.txt) for the evrp solution, containing the instance name, the objective value, and the route
    Args:
        YourName::str
            your name, eg. John_Doe
        evrp::EVRP
            an EVRP object
        suffix::str
            suffix of the output file,
            eg. 'initial' for random initialization
            and 'solution' for the final solution
    '''
    str_builder = ['{}\nInstance: {}\nObjective: {}\n'.format(YourName, routing.name, routing.objective())]
    for idx, r in enumerate(routing.route):
        str_builder.append('Route {}:'.format(idx))
        for node in r:
            if node.type == 0:
                str_builder.append('depot {}'.format(node.id))
            elif node.type == 1:
                str_builder.append(f'{"pick" if node.demand>0 else "drop"} {node.id}')
        str_builder.append('\n')
    with open('{}_{}_{}.txt'.format(YourName, routing.name, suffix), 'w') as f:
        f.write('\n'.join(str_builder))

### Repair operators ###
# You can follow the example and implement repair_2, repair_3, etc
def dynamic_insert(destroyed, request_time):

    while len(destroyed.job_unvisited) > 0:
        job = destroyed.job_unvisited[0]
        P,D = job.P, job.D

        found = False
        heuristic_ls = [(approx_vehicle(v.route, job),idx) for idx,v in enumerate(destroyed.vehicles)]
        heuristic_ls = sorted(heuristic_ls)
        # top_k = max(int(len(destroyed.vehicles)/10), 1)
        candidate_vehicles = [x[1] for x in heuristic_ls]
        
        # random_state.shuffle(destroyed.vehicles)
        for v_idx in candidate_vehicles:
            v = destroyed.vehicles[v_idx]
            cur_route = v.route
            if ENABLE_ROUTE_THRESHOLD and len(v.route) > INIT_ROUTE_LENGTH + destroyed.adjustment * ADJ_ROUTE_LENGTH:
                continue
            new_jobs = v.jobs + [job]

            for i in range(len(cur_route)+1):
                if i < len(cur_route) and v.node_visited_time[i+1] < request_time:
                    continue
                if i < len(cur_route) and cur_route[i].end_time < P.start_time:
                    continue
                for j in range(i+1, len(cur_route)+2):
                    if j < len(cur_route) and cur_route[j].end_time < D.start_time:
                        continue
                    new_v = copy.deepcopy(v)
                    new_route = cur_route[:i] + [P] + cur_route[i:j] + [D] + cur_route[j:]
                    new_v.reset()
                    new_v.run_route(new_route, new_jobs)
                    if new_v.check_time() and new_v.check_capacity():
                        if max(new_v.tardy_list) <= destroyed.adjustment * ADJ_TARDINESS:
                            destroyed.vehicles[v_idx] = new_v
                            del destroyed.job_unvisited[0]
                            found = True
                            break
                if found: break
            if found: 
                break

        if not found:
            if BOOL_ADJ:
                destroyed.adjustment += 1
            new_v = copy.deepcopy(destroyed.vehicle)
            new_v.run_route([P,D], [job])
            del destroyed.job_unvisited[0]
            destroyed.vehicles.append(new_v)
    
    destroyed.update()

    return destroyed



def output2object(output_file):
    with open(output_file, "r") as f:
        lines = f.readlines()
    
    json_file = lines[1].split(":")[1].strip()
    parsed = Parser(json_file, CONCAT=CONCAT_ORDERS)
    routing = CPDPTW(parsed.name, parsed.depot, parsed.jobs, parsed.vehicle, vehicle_cost=VEHICLE_PENALTY)

    job_dict = dict()
    for j in parsed.jobs:
        job_dict[j.id] = j

    current_id = 0
    current_route = []
    current_jobs = []
    for row in lines[4:]:
        if len(row) <= 1:
            continue
        
        node_type, node_id = row.strip().split()

        if node_type == "depot":
            current_route.append(routing.depot)
        if node_type == "pick":
            current_route.append(job_dict[node_id].P)
        if node_type == "drop":
            cur_job = job_dict[node_id]
            current_route.append(cur_job.D)
            current_jobs.append(cur_job)

        if len(current_route) > 2 and isinstance(current_route[-1], Depot):
            new_v = copy.deepcopy(routing.vehicle)
            new_v.id = current_id
            new_v.run_route(current_route[1:-1], current_jobs)
            routing.vehicles.append(new_v)
            current_route = []
            current_jobs = []
            current_id += 1

    routing.update()

    return routing


def load_ortools_route(ortools_route, NUM_ORTOOLS_NODES):
    _, _, all_orders = json2data(json_file)
    all_orders = all_orders[:NUM_ORTOOLS_NODES]
    jobs, ortools2node = orders2jobs(all_orders)
    ortools_route_ls = [convertORtoolOutput(r, jobs, ortools2node) for r in ortools_route]
    routing = CPDPTW(parsed.name, parsed.depot, parsed.jobs, parsed.vehicle, vehicle_cost=VEHICLE_PENALTY)
    
    for v_id, (cur_route,cur_jobs) in enumerate(ortools_route_ls):
        new_v = copy.deepcopy(routing.vehicle)
        new_v.id = v_id
        new_v.run_route(cur_route, cur_jobs)
        routing.vehicles.append(new_v)
    routing.update()
    return routing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data')
    # parser.add_argument(dest='data', type=str, help='data')
    # parser.add_argument(dest='seed', type=int, help='seed')
    args = parser.parse_args()
    
    # instance file and random seed
    # output_file = args.data
    # seed = int(args.seed)

    json_file = "full_config.json"
    parsed = Parser(json_file, CONCAT=CONCAT_ORDERS)

    print("-" * 10, "ORtools insertion test", "-" * 10)
    # Load ORTools route and find objective
    NUM_ORTOOLS_NODES = 100
    routing = load_ortools_route(ortools_route, NUM_ORTOOLS_NODES)
    print("ORtools obj:", routing.objective())
    print("ORtools num_vehicles:", len(routing.vehicles))
    print("ORtools tardiness:", routing.tardiness)
    print("ORtools num jobs:", sum(len(v.jobs) for v in routing.vehicles))
    print("Max tardiness for a customer:", max(max(v.tardy_list) for v in routing.vehicles))
    # print([v.tardy_list for v in routing.vehicles])
    save_output('ORtools', routing, '50')

    # Add next 10 jobs (20 nodes)
    json_file = "full_config.json"
    parsed = Parser(json_file, CONCAT=60)
    to_insert = parsed.jobs[50:60]
    to_insert = sorted(to_insert, key=lambda j: j.P.start_time)

    
    routing.adjustment = 3
    for i,j in enumerate(to_insert):
        # Assume request time = Pickup start time
        request_time = j.P.start_time
        routing.job_unvisited.append(j)
        start_time = time()
        routing = dynamic_insert(routing, request_time)
        print(f"Insertion {i} Time taken:", time()-start_time, "s")


    print("New obj:", routing.objective())
    print("New num_vehicles:", len(routing.vehicles))
    print("New tardiness:", routing.tardiness)
    print("New num jobs:", sum(len(v.jobs) for v in routing.vehicles))
    print("Max tardiness for a customer:", max(max(v.tardy_list) for v in routing.vehicles))

    save_output('Dyn_small', routing, 'v1')

    # Load ground truth
    NUM_ORTOOLS_NODES = 120
    routing = load_ortools_route(ortools_route_2, NUM_ORTOOLS_NODES)
    print("ORtools obj:", routing.objective())
    print("ORtools num_vehicles:", len(routing.vehicles))
    print("ORtools tardiness:", routing.tardiness)
    print("ORtools num jobs:", sum(len(v.jobs) for v in routing.vehicles))
    print("Max tardiness for a customer:", max(max(v.tardy_list) for v in routing.vehicles))

    save_output('ORtools', routing, '60')



    
    