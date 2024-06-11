"""
This module defines the Network Emulator class, which can represent a network,
test his resilience and generate data for the network.
"""
import ast
import json
import os
import random
import sys
import time
import networkx as nx
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from tqdm import tqdm
import utils
from link import Link
from router import Router, ForwardTableElement

# Import necessary modules
import matplotlib.pyplot as plt
import scipy.stats as stats

class NetworkEmulator:
    """
    This class represents a network emulator that simulates the behavior of a network.

    Attributes:
        routers (list): List to store routers.
        links (list): List to store links.
        node_file (str): Node file.
        link_file (str): Link file.
        generation_rate (float): Generation rate.
        num_generation (int): Number of generations.
        duration (int): Duration of the simulation.
        net_graph (None or numpy.ndarray): Network graph.
        G (None or networkx.Graph): Networkx graph.
        net_state_graph (None or numpy.ndarray): Network state matrix.
        router_failure (bool): Status of router failure.
        link_failure (bool): Status of link failure.
        link_failed (list): List of failed links.
        isolated_routers (dict): Dictionary of isolated routers.
        node_degree (None or numpy.ndarray): Node degree.
        total_paths (int): Total number of paths in the network.
        max_fib_break (int): Maximum number of FIB breaks.
        fib_break_spread (float): Spread of FIB breaks, between 0 and 1 (default: 1).
        failure_combinations (list): List of failure combinations.
        prob_start (int): Probability start.
        link_failure_time (list): List of link failure times.
        session_id (None or str): Session ID.
        input_file (None or str): Input file.
        load_folder (None or str): Load folder.
        save_folder (None or str): Save folder.
    """

    def __init__(self, node_file, link_file, generation_rate, num_generation,
                 duration, max_fib_break=1, fib_break_spread=1, input_file=None, 
                 load_folder=None, save_folder=None):
        """
        Initialize the NetworkEmulator with node file, link file, generation rate, and number of generations.

        Args:
            node_file (str): Node file.
            link_file (str): Link file.
            generation_rate (float): Generation rate.
            num_generation (int): Number of generations.
            duration (int): Duration of the simulation.
            max_fib_break (int, optional): Maximum number of FIB breaks. Defaults to 1.
            fib_break_spread (float, optional): Spread of FIB breaks, between 0 and 1 (default: 1). Defaults to 1.
            input_file (str, optional): Input file. Defaults to None.
            load_folder (str, optional): Load folder. Defaults to None.
            save_folder (str, optional): Save folder. Defaults to None.
        """
        self.routers = list()
        self.links = list()

        self.node_file = node_file
        self.link_file = link_file
        self.generation_rate = generation_rate
        self.num_generation = num_generation
        self.duration = duration

        self.net_graph = None
        self.g = None
        self.net_state_graph = None

        self.router_failure = False
        self.link_failure = False
        self.link_failed = []
        self.isolated_routers = {}

        self.node_degree = None
        self.total_paths = 0

        self.max_fib_break = max_fib_break
        self.fib_break_spread = fib_break_spread
        self.failure_combinations = []

        self.prob_start = 0
        self.link_failure_time = []

        self.session_id = None
        self.input_file = input_file
        self.load_folder = load_folder
        self.save_folder = save_folder

    def build(self):
        """
        Build the network by building routers and links.
        """
        if self.load_folder is not None:
            self.__load_network()
        else:
            self.__build_routers()
            self.__build_links()
        print("Network build with " + str(len(self.routers)) +
              " routers and " + str(len(self.links)) + " links")

    def __load_network(self):
        """
        Load the network from saved files.
        """
        with open(f"{self.load_folder}/routers.json", 'r') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error: JSON parsing failed for file '{self.load_folder}'/routers.json")
                return 1

        for obj in json_data:
            node_name = obj.get('node_name', None)
            active = obj.get('active', 1)
            ip_address = obj.get('ip_address', None)

            self.routers.append(
                Router(node_name=node_name, active=active, ip_address=ip_address))

            for element in obj.get('forward_table', []):
                dest = element.get('destination', None)
                next_hop = element.get('next_hop', None)
                self.routers[-1].update_forward_table(ForwardTableElement(dest=dest, next_hop=next_hop))

        with open(f"{self.load_folder}/links.json", 'r') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error: JSON parsing failed for file '{self.link_file}'")
                return 1

        for obj in json_data:
            link_id = obj.get('id', None)
            source = obj.get('source', None)
            destination = obj.get('destination', None)
            delay = obj.get('delay', 0)
            cost = obj.get('cost', 0)

            self.links.append(
                Link(link_id=link_id, source=source, destination=destination, delay=delay, cost=cost))

    def __build_routers(self):
        """
        Build routers from the node file.
        """
        with open(self.node_file, 'r') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error: JSON parsing failed for file '{self.node_file}' ")
                return 1

        for obj in json_data:
            attributes_obj = obj.get('attributes', {})
            ls_attribute_obj = attributes_obj.get('ls_attribute', {})
            node_name_obj = ls_attribute_obj.get('node_name', {})
            node_name = node_name_obj.strip() if node_name_obj else None

            nlri = obj.get('nlri', {}).get('nlri_data', {}).get('lsnlri', {})
            local_node_descriptor = nlri.get('local_node_descriptor', {})
            igp_router_id = local_node_descriptor.get(
                'igp_router_id', {}).get('$binary', {}).get('base64', None)
            base64_str = igp_router_id.strip() if igp_router_id else None

            self.routers.append(
                Router(node_name=node_name, active=1, ip_address=base64_str))

    def __build_links(self):
        """
        Build links from the link file.
        """
        with open(self.link_file, 'r') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error: JSON parsing failed for file '{self.link_file}'")
                return 1

        for obj in json_data:
            source, destination, delay, igp_metric = None, None, 0, 0
            link_id = obj.get('_id', {}).get('$oid', None)
            nlri = obj.get('nlri', {}).get('nlri_data', {}).get('lsnlri', {})
            local_node_descriptor = nlri.get('local_node_descriptor', {})
            remote_node_descriptor = nlri.get('remote_node_descriptor', {})
            attributes_obj = obj.get('attributes', {})
            ls_attribute_obj = attributes_obj.get('ls_attribute', {})

            source = local_node_descriptor.get('igp_router_id', {}).get(
                '$binary', {}).get('base64', None)
            destination = remote_node_descriptor.get(
                'igp_router_id', {}).get('$binary', {}).get('base64', None)

            igp_metric = ls_attribute_obj.get('igp_metric', 0)
            delay = ls_attribute_obj.get('unidir_delay', {}).get('delay', 0)

            link = Link(link_id=link_id, source=source, destination=destination,
                        delay=delay, cost=igp_metric)
            self.links.append(link)

    def __save_network(self):
        """
        Save the network to files.
        """
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        with open(f"{self.save_folder}/routers.json", 'w') as file:
            json.dump([router.to_json() for router in self.routers], file, indent=4)

        with open(f"{self.save_folder}/links.json", 'w') as file:
            json.dump([link.to_json() for link in self.links], file, indent=4)
        
                
                    

    def start(self):
        # Record the start time and get the number of routers
        start = time.time()
        number_of_routers = len(self.routers)

        # Construct a network graph as a sparse matrix

        # Create the array
        net_graph = np.empty(
            (number_of_routers, number_of_routers), dtype=object)
        for i in range(number_of_routers):
            for j in range(number_of_routers):
                net_graph[i, j] = []

        # Create the array
        self.net_state_graph = np.empty(
            (number_of_routers, number_of_routers), dtype=object)
        for i in range(number_of_routers):
            for j in range(number_of_routers):
                self.net_state_graph[i, j] = []

        # Fill the network graph with link costs
        for i, router in enumerate(self.routers):
            # Find the links for the current router
            indices = [index for index, value in enumerate(
                self.links) if value.source == router.ip_address]
            for value in indices:
                # Find the destination router index for each link
                link = self.links[value]
                destination = next(index for index, value in enumerate(
                    self.routers) if value.ip_address == link.destination)
                # Update the network graph with the link cost
                net_graph[i, destination].append(self.links[value].cost)
                self.net_state_graph[i, destination].append(1)

        for i in range(number_of_routers):
            for j in range(number_of_routers):
                if len(net_graph[i, j]) == 0:
                    net_graph[i, j] = [0]

        # create a networkx graph
        G = nx.MultiDiGraph()

        # Add nodes to the graph
        G.add_nodes_from(range(len(net_graph)))

        # Add edges to the graph with weights
        for i in range(len(net_graph)):
            for j in range(len(net_graph[i])):
                for k in range(len(net_graph[i, j])):
                    if net_graph[i, j][k] != 0:
                        G.add_edge(i, j, weight=net_graph[i, j][k])
        self.net_graph = net_graph
        self.g = G
        # Print the time taken to build the graph
        end = time.time()
        print("Build graph in: {}".format(end - start))

        '''
        with ProcessPoolExecutor() as executor:
            #pathLists = list(executor.map(utils.dijkstra, [net_graph]*number_of_routers, range(number_of_routers)))

        # Update the forward table for each router with the shortest paths

        for i in range(number_of_routers):
            pathList = pathLists[i]
            for j in range(number_of_routers):
                elements = pathList[j]
                for k in range(len(elements)):
                    self.routers[i].updateForwardTable(ForwardTableElement(
                        dest=self.routers[elements[k][0]].ip_address,
                        next_hop=self.routers[elements[k][1]].ip_address,
                        cost=elements[k][2]
                    ))
        '''
        if(self.load_folder == None):
            with ThreadPoolExecutor() as executor:
                # Use submit to asynchronously submit tasks to the executor
                futures = []
                for i in range(number_of_routers):
                    for j in range(number_of_routers):
                        if i != j:
                            futures.append(executor.submit(
                                self.update_forward_table, G, i, j))

                # Wait for all tasks to complete
                progress_bar = tqdm(total=len(futures), desc="Processing")

                for future in futures:
                    future.result()
                    progress_bar.update(1)

                progress_bar.close()
            
            if(self.save_folder != None):
                self.__save_network()

            # Print the time taken to start the network
        end = time.time()
        print("Network started in: {}".format(end - start))

    def update_forward_table(self, G, source, target):
        # Find the shortest paths from the source to the destination
        if not self.routers[source].has_entry_for_destination(self.routers[target].ip_address):
            shortest_paths = list(nx.all_shortest_paths(
                G, source=source, target=target, weight='weight'))
            for path in shortest_paths:
                # Update the forward table for the source router
                self.routers[source].update_forward_table(ForwardTableElement(
                    next_hop=self.routers[path[1]].ip_address,
                    dest=self.routers[path[len(path) - 1]].ip_address
                ))
                for i in range(1, len(path) - 1):
                    self.routers[path[i]].update_forward_table(ForwardTableElement(
                        next_hop=self.routers[path[i + 1]].ip_address,
                        dest=self.routers[path[len(path) - 1]].ip_address
                    ))

    def export_sink_data(self, source, destination, time_array, latencies, gen_number, timestamp):

        # Export the data to a CSV file
        sink_csv_file = './src/results/sink.csv'

        if os.path.isfile(sink_csv_file):
            sink = pd.read_csv(sink_csv_file)
            if self.session_id == None:
                self.session_id = sink.iloc[-1]['Session ID'] + 1
        else:
            sink = pd.DataFrame(columns=['Session ID', 'Source', 'Destination', 'TimeStamp',
                                'Histogram template', 'Histogram values', 'Histogram type', 'Liveness flag'])
            self.session_id = 1
        histogram_type = None
        if gen_number % 2 == 0:
            histogram_type = 'Black'
        else:
            histogram_type = 'White'

        source = source
        destination = destination

        new_row = {'Session ID': self.session_id,
                   'Source': source,
                   'Destination': destination,
                   'TimeStamp': timestamp,
                   'Histogram template': time_array,
                   'Histogram values': latencies,
                   'Histogram type': histogram_type,
                   'Liveness flag': True}

        # Append the new row to the DataFrame
        sink.loc[len(sink)] = new_row
        pd.DataFrame.to_csv(sink, sink_csv_file, index=False)

    def export_source_data(self, source, destination, gen_number, timestamp):
        # Export the data to a CSV file
        source_csv_file = './src/results/source.csv'

        if os.path.isfile(source_csv_file):
            sink = pd.read_csv(source_csv_file)
            if self.session_id == None:
                self.session_id = sink.iloc[-1]['Session ID'] + 1
        else:
            sink = pd.DataFrame(columns=['Session ID', 'Source', 'Destination',
                                'TimeStamp', 'Probe GenRate', 'Packet Counter', 'Packet Counter Type'])
            self.session_id = 1
        packet_type = None
        if gen_number % 2 == 0:
            packet_type = 'Black'
        else:
            packet_type = 'White'

        source = source
        destination = destination

        new_row = {'Session ID': self.session_id,
                   'Source': source,
                   'Destination': destination,
                   'TimeStamp': timestamp,
                   'Probe GenRate': self.generation_rate,
                   'Packet Counter': self.generation_rate * self.duration,
                   'Packet Counter Type': packet_type}

        # Append the new row to the DataFrame
        sink.loc[len(sink)] = new_row
        pd.DataFrame.to_csv(sink, source_csv_file, index=False)

    def export_output_matrix(self, timestamp):
        file_path = f"src/results/net_states/{timestamp}.txt"
        # Title for the matrix
        title = f"{timestamp}"

        # Export the matrix to a text file
        with open(file_path, 'w') as file:
            file.write(title + '\n')  # Write the title
            for row in self.net_state_graph:
                file.write('\t'.join(map(str, row)) + '\n')
    
    def export_network_resilience(self, link_scores):
        csv_file = './src/results/resilience.csv'

        result = pd.DataFrame(columns=['Links ID', 'Mean latency increase', 'isolated network', 'isolated routers'])
        for link_score in link_scores:
            new_row = {'Links ID': [getattr(link, 'id') for link in link_score[0]],
                   'Mean latency increase': link_score[1],
                   'isolated network': link_score[2],
                   'isolated routers': link_score[3]}
            result.loc[len(result)] = new_row
        pd.DataFrame.to_csv(result, csv_file, index=False)

    def compare_latencies(self, default_latencies, latencies):
        total_difference = 0

        n = len(default_latencies)

        for i in range(n):
            for j in range(n):
                mean1 = np.mean(default_latencies[i][j])
                mean2 = np.mean(latencies[i][j])
                if mean2 < 10000000:
                    difference = mean2 - mean1
                    total_difference += difference
        return total_difference

    def network_resilience_testing(self):

        link_score = []
        if(len(self.failure_combinations) == 0):
            # Generate all possible failure scenarios
            self.generate_failure_scenarios()
            
        # Generate default latencies for the network as a matrix for all possibles routes
        default_latencies = utils.list_to_matrix(self.get_all_routes_delay())
        for combination in self.failure_combinations:
            self.link_failed = combination
            for link in combination:
                self.g.remove_edge(self.get_router_index(
                    link.source), self.get_router_index(link.destination))
            latencies = utils.list_to_matrix(self.get_all_routes_delay())
            link_score.append(
                [combination, self.compare_latencies(default_latencies, latencies)])
            for link in combination:
                self.g.add_edge(self.get_router_index(link.source), self.get_router_index(
                    link.destination), weight=link.cost)

        for j in range(len(self.failure_combinations)):
            if j in self.isolated_routers:
                count = 0
                link_score[j].append(len(self.isolated_routers[j]))
                for i in range(len(self.isolated_routers[j])):
                    count += len(self.isolated_routers[j][i])
                link_score[j].append(count)
            else:
                link_score[j].append(0)
                link_score[j].append(0)
        link_score = utils.sort_latency_list(link_score)
            
        self.export_network_resilience(link_score)
        print("Network resilience testing completed, result exported to resilience.csv")

    def get_all_routes_delay(self):
        latencies = []

        for i in range(len(self.routers)):
            for j in range(len(self.routers)):
                latencies.append(self.get_route_delay(
                    self.routers[i], self.routers[j]))
        return latencies

    def get_route_delay(self, source, destination):
        if source.ip_address == destination.ip_address:
            return [0]

        indices = [index for index, value in enumerate(
            source.forward_table) if value.destination == destination.ip_address]

        latencies = []

        for i in indices:
            index = next(index for index, value in enumerate(self.links) if value.source ==
                         source.ip_address and value.destination == source.forward_table[i].next_hop)
            if self.links[index] not in self.link_failed:
                latency = self.links[index].delay
                index = next(index for index, value in enumerate(
                    self.routers) if value.ip_address == source.forward_table[i].next_hop)
                next_router = self.routers[index]
                next_latencies = self.get_route_delay(next_router, destination)
                for next_latency in next_latencies:
                    latencies.append(latency + next_latency)
            else:
                try:
                    new_path = list(nx.shortest_path(self.g, source=self.get_router_index(
                        source.ip_address), target=self.get_router_index(destination.ip_address), weight='weight'))
                    latency = 0

                    for j in range(len(new_path)-1):
                        index = next(index for index, value in enumerate(self.links) if value.source ==
                                     self.routers[new_path[j]].ip_address and value.destination == self.routers[new_path[j+1]].ip_address)
                        latency += self.links[index].delay

                except:
                    latency = 1000000000
                    isolated_routers = self.get_isolated_routers(
                        source.ip_address)
                    if self.failure_combinations.index(self.link_failed) in self.isolated_routers:
                        if(isolated_routers not in self.isolated_routers[self.failure_combinations.index(self.link_failed)]):
                            self.isolated_routers[self.failure_combinations.index(self.link_failed)].append(isolated_routers)
                    else:
                        self.isolated_routers[self.failure_combinations.index(self.link_failed)] = [isolated_routers]
                finally:
                    latencies.append(latency)
        return latencies

    def get_isolated_routers(self, source):
        isolated_routers = set()
        isolated_routers.add(source)
        for router in self.routers:
            if router.ip_address == source:
                continue
            try: 
                nx.shortest_path_length(self.g, source=self.get_router_index(source), target=self.get_router_index(router.ip_address), weight='weight') 
                isolated_routers.add(router.ip_address)
            except:
                continue
        return isolated_routers

    def generate_failure_scenarios(self):
        # Generate all links failures combinations from 1 to max_fib_break
        for i in range(1, self.max_fib_break + 1):
            self.generate_failure_combinations(i)

    def generate_failure_combinations(self, n):
        # Generate all links failures combinations of n links
        self.generate_failure_combinations_helper(n, 0, [])

    def generate_failure_combinations_helper(self, n, start, combination):
        # Generate all links failures combinations of n links
        if n == 0:
            self.failure_combinations.append(combination)
            return
        for i in range(start, len(self.links)):
            self.generate_failure_combinations_helper(
                n - 1, i + 1, combination + [self.links[i]])

    def emulate_all(self):

        start = time.time()
        df = pd.DataFrame(columns=['source', 'destination', 'latency'])
        time_array = []
        for gen_number in range(self.num_generation):
            flow_label = 0
            with ThreadPoolExecutor() as executor:
                for t in range(self.generation_rate):
                    # Submit tasks for each source
                    futures = []
                    
                    for source in self.routers:
                        for destination in self.routers:
                            if source != destination:
                                future = executor.submit(
                                        self.send_prob, source, destination.ip_address, flow_label)
                                futures.append(future)

                    # Wait for all tasks to complete
                    for future in futures:
                        result = future.result()
                        condition = (df['source'] == result[0].ip_address) & (df['destination'] == result[1])
                        data = df.loc[condition]
                        if not data.empty: 
                            index = data.index[0]
                            df.at[index, 'latency'].append(result[2])
                        else: 
                            new_row = {'source': result[0].ip_address, 'destination': result[1], 'latency': [result[2]]}
                            df.loc[len(df)] = new_row
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(df)
            for index, row in df.iterrows():
                self.export_sink_data(row['source'], row['destination'], [], row['latency'], gen_number, timestamp)
                self.export_source_data(row['source'], row['destination'], gen_number, timestamp)
            df = pd.DataFrame(columns=['source', 'destination', 'latency'])
        print("Emulated all network for " + str(self.num_generation) + " generations with " +
              str(self.generation_rate) + " probes in " + str(time.time() - start))
        
    def send_prob(self, source, destination, flow_label):
        # Initialize a latency array
        latency = 0
        # For each generation, calculate the latency from source to destination
        indices = [index for index, value in enumerate(
        source.forward_table) if value.destination == destination]
        # Using hash function to determine which route to take
        chosen_route = source.select_route(
                destination, len(indices), flow_label)
        index = next(index for index, value in enumerate(self.links) if value.source ==
                        source.ip_address and value.destination == source.forward_table[indices[chosen_route]].next_hop)
        latency += self.links[index].delay

        # Find the next router
        index = next(index for index, value in enumerate(
            self.routers) if value.ip_address == source.forward_table[indices[chosen_route]].next_hop)
        next_router = self.routers[index]

        # Continue finding the next hop and adding the delay until the destination is reached
        while (next_router.ip_address != destination):
            indices = [index for index, value in enumerate(
                next_router.forward_table) if value.destination == destination]
            chosen_route = next_router.select_route(
                destination, len(indices), flow_label)
            index = next(index for index, value in enumerate(self.links) if value.source ==
                            next_router.ip_address and value.destination == next_router.forward_table[indices[chosen_route]].next_hop)
            # Look different case based on the link failure or not
            if not self.links[index] in self.link_failed:
                latency += self.links[index].delay
                index = next(index for index, value in enumerate(
                    self.routers) if value.ip_address == next_router.forward_table[indices[0]].next_hop)
                next_router = self.routers[index]
            elif self.g.has_edge(self.get_router_index(next_router.ip_address), self.get_router_index(next_router.forward_table[indices[chosen_route]].next_hop)):
                self.g.remove_edge(self.get_router_index(next_router.ip_address), self.get_router_index(
                    next_router.forward_table[indices[chosen_route]].next_hop))
                new_path = list(nx.shortest_path(self.g, source=self.get_router_index(
                    next_router.ip_address), target=self.get_router_index(destination), weight='weight'))
                for j in range(len(new_path)-1):
                    index = next(index for index, value in enumerate(self.links) if value.source ==
                                    self.routers[new_path[j]].ip_address and value.destination == self.routers[new_path[j+1]].ip_address)
                    latency += self.links[index].delay
                break
            else:
                new_path = list(nx.shortest_path(self.g, source=self.get_router_index(
                    next_router.ip_address), target=self.get_router_index(destination), weight='weight'))
                for j in range(len(new_path)-1):
                    index = next(index for index, value in enumerate(self.links) if value.source ==
                                    self.routers[new_path[j]].ip_address and value.destination == self.routers[new_path[j+1]].ip_address)
                    latency += self.links[index].delay
                break
        # Return the latency array
        return [source, destination, latency]

    def get_router_index(self, ip_address):
        for i in range(len(self.routers)):
            if (self.routers[i].ip_address == ip_address):
                return i
            
    def emulate(self, source, destination):
        # Generate traffic between the source and destination for a number of generations
        start = time.time()
        for gen_number in range(self.num_generation):
            index = next(index for index, value in enumerate(
                self.routers) if value.ip_address == source)
            source_router = self.routers[index]
            time_array = np.linspace(0, self.duration, self.generation_rate)
            latencies = self.send_probs(
                source=source_router, destination=destination)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.export_sink_data(source, destination,
                                  time_array, latencies, gen_number, timestamp)
            self.export_source_data(source, destination, gen_number, timestamp)
            self.export_output_matrix(timestamp)
            self.output_matrix_reset()
        end = time.time()
        # if len(self.link_failure_time) > 0:
        # print(f"Min time: {min(self.link_failure_time)}, Max time: {max(self.link_failure_time)}, average time {sum(self.link_failure_time)/len(self.link_failure_time)}, number of breaks {len(self.link_failure_time)}")
        print(f"Network emulated in: {
              end-start} for {self.num_generation} generations of {self.generation_rate} probes")

    def send_probs(self, source, destination):
        # Initialize a latency array
        latency = np.zeros(self.generation_rate * self.duration)
        base_pkt_nmbr = random.randint(1, 100000000)
        # For each generation, calculate the latency from source to destination
        for i in range(self.generation_rate * self.duration):

            # Find the next hop from the source to the destination
            indices = [index for index, value in enumerate(
                source.forward_table) if value.destination == destination]
            # Using hash function to determine which route to take
            chosen_route = source.select_route(
                destination, len(indices), base_pkt_nmbr + i)
            index = next(index for index, value in enumerate(self.links) if value.source ==
                         source.ip_address and value.destination == source.forward_table[indices[chosen_route]].next_hop)
            latency[i] += self.links[index].delay

            # Find the next router
            index = next(index for index, value in enumerate(
                self.routers) if value.ip_address == source.forward_table[indices[chosen_route]].next_hop)
            next_router = self.routers[index]

            # Continue finding the next hop and adding the delay until the destination is reached
            while (next_router.ip_address != destination):
                indices = [index for index, value in enumerate(
                    next_router.forward_table) if value.destination == destination]
                chosen_route = next_router.select_route(
                    destination, len(indices), base_pkt_nmbr + i)
                index = next(index for index, value in enumerate(self.links) if value.source ==
                             next_router.ip_address and value.destination == next_router.forward_table[indices[chosen_route]].next_hop)
                # Look different case based on the link failure or not
                if not self.links[index] in self.link_failed:
                    latency[i] += self.links[index].delay
                    index = next(index for index, value in enumerate(
                        self.routers) if value.ip_address == next_router.forward_table[indices[0]].next_hop)
                    next_router = self.routers[index]
                elif self.g.has_edge(self.get_router_index(next_router.ip_address), self.get_router_index(next_router.forward_table[indices[chosen_route]].next_hop)):
                    self.g.remove_edge(self.get_router_index(next_router.ip_address), self.get_router_index(
                        next_router.forward_table[indices[chosen_route]].next_hop))
                    new_path = list(nx.shortest_path(self.g, source=self.get_router_index(
                        next_router.ip_address), target=self.get_router_index(destination), weight='weight'))
                    for j in range(len(new_path)-1):
                        index = next(index for index, value in enumerate(self.links) if value.source ==
                                     self.routers[new_path[j]].ip_address and value.destination == self.routers[new_path[j+1]].ip_address)
                        latency[i] += self.links[index].delay
                    break
                else:
                    new_path = list(nx.shortest_path(self.g, source=self.get_router_index(
                        next_router.ip_address), target=self.get_router_index(destination), weight='weight'))
                    for j in range(len(new_path)-1):
                        index = next(index for index, value in enumerate(self.links) if value.source ==
                                     self.routers[new_path[j]].ip_address and value.destination == self.routers[new_path[j+1]].ip_address)
                        latency[i] += self.links[index].delay
                    break
        # Return the latency array
        return latency

