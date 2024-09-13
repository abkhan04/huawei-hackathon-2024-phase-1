import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution, load_solution, load_problem_data
from evaluation import get_actual_demand, evaluation_function


def get_my_solution(d):
    # Initialize solution list
    solution = []

    # Load necessary data
    _, datacenters, servers, _ = load_problem_data()

    # Example: Iterate through each time step's demand
    for time_step, row in d.iterrows():
        server_generation = row['server_generation']

        # Maintain a list of servers for each data center to track their status
        active_servers = {dc['datacenter_id']: [] for _, dc in datacenters.iterrows()}

        # Decision-making logic
        for dc_index, dc in datacenters.iterrows():
            dc_id = dc['datacenter_id']
            available_capacity = dc['slots_capacity']

            # Check for servers to dismiss
            for server in active_servers[dc_id]:
                if server['operating_time'] >= server['life_expectancy']:
                    action = {
                        "time_step": time_step + 1,
                        "datacenter_id": dc_id,
                        "server_generation": server['server_generation'],
                        "server_id": server['server_id'],
                        "action": "dismiss"
                    }
                    solution.append(action)
                    available_capacity += server['slots_size']  # Free up capacity
                    active_servers[dc_id].remove(server)
                    continue  # Move to the next server

            # Hold servers that are still effective
            for server in active_servers[dc_id]:
                action = {
                    "time_step": time_step + 1,
                    "datacenter_id": dc_id,
                    "server_generation": server['server_generation'],
                    "server_id": server['server_id'],
                    "action": "hold"
                }
                solution.append(action)

            # Check for opportunities to move servers
            for target_dc_index, target_dc in datacenters.iterrows():
                target_dc_id = target_dc['datacenter_id']
                if target_dc_id != dc_id and target_dc['slots_capacity'] > 0:
                    # Example heuristic: move servers to data centers with lower energy costs
                    if target_dc['cost_of_energy'] < dc['cost_of_energy']:
                        for server in active_servers[dc_id]:
                            if target_dc['slots_capacity'] >= server['slots_size']:
                                action = {
                                    "time_step": time_step + 1,
                                    "datacenter_id": dc_id,
                                    "server_generation": server['server_generation'],
                                    "server_id": server['server_id'],
                                    "action": "move",
                                    "target_datacenter_id": target_dc_id
                                }
                                solution.append(action)
                                target_dc['slots_capacity'] -= server['slots_size']
                                available_capacity += server['slots_size']
                                break

            # Example heuristic: Buy new servers if there is unmet demand and capacity is available
            if available_capacity > 0:
                for _, server in servers.iterrows():
                    if server['server_type'] == server_generation.split('.')[0]:  # Match CPU or GPU
                        if server['slots_size'] <= available_capacity:
                            action = {
                                "time_step": time_step + 1,
                                "datacenter_id": dc_id,
                                "server_generation": server['server_generation'],
                                "server_id": f"server_{time_step}_{dc_index}",  # Unique server ID
                                "action": "buy"
                            }
                            solution.append(action)
                            available_capacity -= server['slots_size']
                            active_servers[dc_id].append({
                                'server_id': f"server_{time_step}_{dc_index}",
                                'server_generation': server['server_generation'],
                                'operating_time': 0,
                                'life_expectancy': server['life_expectancy'],
                                'slots_size': server['slots_size']
                            })
                            break

    return solution


seeds = known_seeds('training')
demand = pd.read_csv('./data/demand.csv')


# GET SOLUTIONS
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    solution = get_my_solution(actual_demand)    

    # SAVE YOUR SOLUTION
    save_solution(solution, f'./output/{seed}.json')


# EVALUATE SOLUTIONS
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # LOAD SOLUTION
    solution = load_solution(f'./output/{seed}.json')

    # LOAD PROBLEM DATA
    demand, datacenters, servers, selling_prices = load_problem_data()

    # EVALUATE THE SOLUTION
    score = evaluation_function(solution, demand, datacenters, servers, selling_prices, seed=seed)

    print(f'Solution score: {score}')
