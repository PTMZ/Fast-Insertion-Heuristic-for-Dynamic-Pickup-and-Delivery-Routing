import argparse
import numpy as np
import numpy.random as rnd
import networkx as nx
import matplotlib.pyplot as plt
from lxml import etree as LET

from cpdptw import *
from pathlib import Path

import sys
sys.path.append('./ALNS')
from alns import ALNS, State
from alns.criteria import HillClimbing, SimulatedAnnealing, RecordToRecordTravel

CONCAT_ORDERS = 0
VEHICLE_PENALTY = 60
OUTFILE_PREFIX = "Big"
INIT_ROUTE_LENGTH = 25
ADJ_ROUTE_LENGTH = 2
ADJ_TARDINESS = 5
BOOL_ADJ = True
ENABLE_ROUTE_THRESHOLD = True

def sub_heuristic(A, B):
    W = 2500
    # return (A.x-B.x)**2 + (A.y-B.y)**2
    return (A.x-B.x)**2 + (A.y-B.y)**2 + ((A.start_time - B.start_time)/W)**2

def approx_vehicle(route, job):
    N = len(route)
    dist_drop = [sub_heuristic(job.D, n) for n in route]
    for i in range(N-1):
        dist_drop[N-2-i] = min(dist_drop[N-2-i], dist_drop[N-1-i])
    dist_pick = [sub_heuristic(job.P, n) for n in route]
    ans = min(a+b for a,b in zip(dist_pick, dist_drop))
    return ans

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
    #draw_route(YourName, route, suffix)

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
    g.add_node(routing.depot.id, pos=(routing.depot.x, routing.depot.y), type=routing.depot.type)
    for job in routing.jobs:
        p,d = job.P, job.D
        g.add_node(p.id, pos=(p.x, p.y), type=p.type)
        g.add_node(d.id, pos=(d.x, d.y), type=d.type)
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
    route = list(node.id for node in sum(routing.route, []))
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

    plt.text(0, 6, YourName, fontsize=12)
    plt.text(0, 3, 'Instance: {}'.format(routing.name), fontsize=12)
    plt.text(0, 0, 'Objective: {}'.format(routing.objective()), fontsize=12)
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

### Destroy operators ###
# You can follow the example and implement destroy_2, destroy_3, etc
def destroy_1(current, random_state):
    ''' Destroy operator sample (name of the function is free to change)
    Args:
        current::CPDPTW
            a CPDPTW object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        destroyed::CPDPTW
            the CPDPTW object after destroying
    '''
    destroyed = current.copy()
    
    # 1) Remove K random vehicles
    frac_destroy = 0.2
    num_veh_destroy = int(len(destroyed.vehicles) * frac_destroy)
    route_lengths = sorted([(len(v.node_visited),idx) for idx,v in enumerate(destroyed.vehicles)])
    
    #destroy_v_idx_list = list(random_state.choice(len(destroyed.vehicles), num_veh_destroy, replace=False))
    destroy_v_idx_list = [idx for _,idx in route_lengths[:num_veh_destroy]]
    destroy_v_idx_list = sorted(destroy_v_idx_list, reverse=True)
    for v_idx in destroy_v_idx_list:
        destroyed.job_unvisited.extend(destroyed.vehicles[v_idx].jobs)
        del destroyed.vehicles[v_idx]

    destroyed.job_unvisited = sorted(destroyed.job_unvisited, key=lambda x: x.P.start_time)
    return destroyed


def destroy_2(current, random_state):

    destroyed = current.copy()
    
    # 2) Remove K random jobs from a vehicle
    frac_destroy = 0.1

    veh_to_destroy = []
    for v_idx,v in enumerate(destroyed.vehicles):
        rm_set = set()
        for j in v.jobs:
            if random.random() < frac_destroy:
                rm_set.add(j.id)
                destroyed.job_unvisited.append(j)
        if len(rm_set) == len(v.jobs):
            veh_to_destroy.append(v_idx)
        elif len(rm_set) > 0:
            new_route = [c for c in v.route if c.id not in rm_set]
            new_jobs = [j for j in v.jobs if j.id not in rm_set]
            destroyed.vehicles[v_idx].reset()
            destroyed.vehicles[v_idx].run_route(new_route, new_jobs)
    
    for idx in veh_to_destroy[::-1]:
        del destroyed.vehicles[idx]
    
    return destroyed

### Repair operators ###
# You can follow the example and implement repair_2, repair_3, etc
def repair_1(destroyed, random_state):
    ''' repair operator sample (name of the function is free to change)
    Args:
        destroyed::CPDPTW
            a CPDPTW object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::CPDPTW
            the CPDPTW object after repairing
    '''

    while len(destroyed.job_unvisited) > 0:
        job = destroyed.job_unvisited[0]
        P,D = job.P, job.D

        found = False
        # random_state.shuffle(destroyed.vehicles)
        for v_idx, v in enumerate(destroyed.vehicles):
            cur_route = v.route
            if len(v.route) >= INIT_ROUTE_LENGTH + destroyed.adjustment * ADJ_ROUTE_LENGTH:
                continue
            new_jobs = v.jobs + [job]

            for i in range(len(cur_route)+1):
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
        
def repair_2(destroyed, random_state):

    
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
                if i < len(cur_route) and v.node_visited_time[i+1] < P.start_time:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data')
    # parser.add_argument(dest='data', type=str, help='data')
    parser.add_argument(dest='seed', type=int, help='seed')
    args = parser.parse_args()
    
    # instance file and random seed
    # xml_file = args.data
    seed = int(args.seed)
    
    # load data and random seed
    # parsed = Parser(xml_file)
    json_file = "full_config.json"
    parsed = Parser(json_file, CONCAT=CONCAT_ORDERS)
    
    routing = CPDPTW(parsed.name, parsed.depot, parsed.jobs, parsed.vehicle, vehicle_cost=VEHICLE_PENALTY)
    
    # construct random initialized solution
    routing.random_initialize(seed)
    print("Initial solution objective is {}.".format(routing.objective()))
    
    # visualize initial solution and gernate output file
    save_output(OUTFILE_PREFIX, routing, 'init')
    
    # ALNS
    random_state = rnd.RandomState(seed)
    alns = ALNS(random_state)
    alns.add_destroy_operator(destroy_1)
    alns.add_repair_operator(repair_2)
    
    criterion = HillClimbing()
    omegas = [3, 2, 1, 0.5]
    lambda_ = 0.85
    result = alns.iterate(routing, omegas, lambda_, criterion,
                          iterations=15, collect_stats=True)

    # result
    solution = result.best_state
    print("Adjustment:", solution.adjustment)
    objective = solution.objective()
    print('Best heuristic objective is {}.'.format(objective))
    print('Best num veh is {}.'.format(len(solution.vehicles)))
    print('Best tardiness is {}.'.format(solution.tardiness))
    print('Max tardiness per Customer is {}.'.format(max(max(v.tardy_list) for v in solution.vehicles)))
    
    # visualize final solution and gernate output file
    save_output(OUTFILE_PREFIX, solution, 'sol')

    testing_output = output2object(f"{OUTFILE_PREFIX}_full_config.json_sol.txt")
    print(testing_output.objective())
    



