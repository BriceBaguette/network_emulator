"""
This module defines the Network Emulator class, which can represent a network,
test his resilience and generate data for the network.
"""
import ast
import json
import os
import random
import re
import sys
import time
import networkx as nx
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from scipy.interpolate import interp1d
from tqdm import tqdm
from typing import List
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
        net_graph (None or numpy.ndarray): Network graph.
        G (None or networkx.Graph): Networkx graph.
        net_state_graph (None or numpy.ndarray): Network state matrix.
        router_failure (bool): Status of router failure.
        link_failure (bool): Status of link failure.
        link_failed (list): List of failed links.
        isolated_routers (dict): Dictionary of isolated routers.
        prob_start (int): Probability start.
        session_id (None or str): Session ID.
        load_folder (None or str): Load folder.
        save_folder (None or str): Save folder.
    """

    def __init__(self, node_file: str, link_file: str, single_file: str ,generation_rate: int, num_generation: int,
                 load_folder: str = None, save_folder: str =None, treshold:int = 5):
        """
        Initialize the NetworkEmulator with node file, link file, generation rate, and number of generations.

        Args:
            node_file (str): Node file.
            link_file (str): Link file.
            generation_rate (float): Generation rate.
            num_generation (int): Number of generations.
            load_folder (str, optional): Load folder. Defaults to None.
            save_folder (str, optional): Save folder. Defaults to None.
        """
        self.routers: List[Router] = []
        self.links: List[Link] = []
        
        self.single_file = single_file
        self.node_file = node_file
        self.link_file = link_file
        self.generation_rate = generation_rate
        self.num_generation = num_generation
        self.treshold = treshold
        self.duration = 300

        self.net_graph = None
        self.g = None
        self.net_state_graph = None

        self.session_id = None
        self.load_folder = load_folder
        self.save_folder = save_folder
        
        self.hw_issue : pd.DataFrame = pd.DataFrame(columns=['Router ID', 'Start', 'End', 'Previous Table Element', 'New Table Element', 'Entry Index'])

    def build(self):
        """
        Build the network by building routers and links.
        """
        if self.load_folder is not None:
            self.__load_network()
        elif self.node_file is not None and self.link_file is not None:
            self.__build_routers()
            self.__build_links()
        elif self.single_file is not None:
            self.__build_network_from_file()
        else: 
            print("Error: No file to build a network.")
            exit(-1)
        print("Network build with " + str(len(self.routers)) +
              " routers and " + str(len(self.links)) + " links")
        
    def __build_network_from_file(self):
        with open(self.single_file, "r") as file:
            topo = file.read()

        routers_info = topo.split("\n\n")

        router_id_pattern = r"Router ID:\s*(\S+)"

        hostname_pattern = r"Hostname:\s*(\S+)"

        ip_address_pattern = r"IP Address:\s*(\S+)"

        for router in routers_info:
            router_match = re.search(router_id_pattern, router)
            host_match = re.search(hostname_pattern, router)
            ip_match = re.search(ip_address_pattern, router)
            if router_match and host_match and ip_match:
                router_id = router_match.group(1)
                hostname = host_match.group(1)
                ip_address = ip_match.group(1)
                self.routers.append(Router(router_id=router_id, node_name=hostname, ip_address=ip_address, active=1))
            
        pattern = re.compile(r'Metric:\s*\d+\s+IS-Extended.*?(?=Metric:|\Z)', re.DOTALL)
        
        delay_pattern = r"Link Average Delay:\s*(\d+)"

        neighbor_pattern = r"Metric:\s*(\d+).*IS-Extended (\S+)\.\d+"
        for router in routers_info:
            host_match = re.search(hostname_pattern, router)
            if host_match:
                hostname = host_match.group(1)
                source: Router = self.get_router_by_name(hostname)
                # Find all matches
                matches = pattern.findall(router) 
                for match in matches:
                    neighbor_match = re.search(neighbor_pattern, match, re.DOTALL)
                    delay_match = re.search(delay_pattern, match)
                    if neighbor_match and delay_match:
                        neighbor = self.get_router_by_name(neighbor_match.group(2))
                        if neighbor:
                            link = Link(link_id=len(self.links), source=source.ip_address, destination=neighbor.ip_address, delay=int(delay_match.group(1)), cost=int(neighbor_match.group(1)))
                            self.links.append(link)

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
            node_id = obj.get('id', None)
            node_name = obj.get('node_name', None)
            active = obj.get('active', 1)
            ip_address = obj.get('ip_address', None)
            self.routers.append(
                Router(node_name=node_name,router_id=node_id, active=active, ip_address=ip_address))

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
            node_id = obj.get('_id', {}).get('$oid', None)
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
                Router(node_name=node_name, router_id=node_id, active=1, ip_address=base64_str))

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

        # Define the directory path
        dir_path = Path(self.save_folder)

        # Check if the directory exists
        if not dir_path.exists():
            # Create the directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
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

    def update_forward_table(self, G, source: int, target:int):
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

    def export_sink_data(self, source, destination,fl, latencies, gen_number, timestamp, vrf="Measurement"):

        # Export the data to a CSV file
        sink_csv_file = './src/results/sink.csv'

        if os.path.isfile(sink_csv_file):
            sink = pd.read_csv(sink_csv_file)
            if self.session_id == None:
                self.session_id = sink.iloc[-1]['Session ID'] + 1
        else:
            sink = pd.DataFrame(columns=['Session ID', 'Source', 'Destination','FL', 'VRF', 'TimeStamp',
                                'Min Latency', 'Max Latency', 'D flag', 'Packet Counter', 'Histogram type', 'Liveness flag', 'Telemetry Period'])
            self.session_id = 1
        histogram_type = None
        if gen_number % 2 == 0:
            histogram_type = 'Black'
        else:
            histogram_type = 'White'

        source = source
        destination = destination
        
        if fl == None:
            fl = "Spray"
        else:
            fl = str(fl)

        new_row = {'Session ID': self.session_id,
                   'Source': source,
                   'Destination': destination,
                   'FL': fl,
                   'VRF': vrf,
                   'TimeStamp': timestamp,
                   'Min Latency': min(latencies),
                   'Max Latency': max(latencies),
                   'D flag': max(latencies) - min(latencies) >= self.treshold,
                   'Packet Counter': len(latencies),
                   'Histogram type': histogram_type,
                   'Liveness flag': True,
                   'Telemetry Period': self.duration/60}

        # Append the new row to the DataFrame
        sink.loc[len(sink)] = new_row
        pd.DataFrame.to_csv(sink, sink_csv_file, index=False)

    def export_source_data(self, source, destination,fl, gen_number, timestamp, vrf="Measurement"):
        # Export the data to a CSV file
        source_csv_file = './src/results/source.csv'
        if os.path.isfile(source_csv_file):
            sink = pd.read_csv(source_csv_file)
            if self.session_id == None:
                self.session_id = sink.iloc[-1]['Session ID'] + 1
        else:
            sink = pd.DataFrame(columns=['Session ID', 'Source', 'Destination', 'FL', 'VRF',
                                'TimeStamp', 'Probe GenRate', 'Packet Counter', 'Packet Counter Type', 'Telemetry Period'])
            self.session_id = 1
        packet_type = None
        if gen_number % 2 == 0:
            packet_type = 'Black'
        else:
            packet_type = 'White'

        source = source
        destination = destination
        if fl == None:
            fl = "Spray"
        else:
            fl = str(fl)

        new_row = {'Session ID': self.session_id,
                   'Source': source,
                   'Destination': destination,
                   'FL': fl,
                   'VRF': vrf,
                   'TimeStamp': timestamp,
                   'Probe GenRate': self.generation_rate,
                   'Packet Counter': self.generation_rate * self.duration,
                   'Packet Counter Type': packet_type,
                   'Telemetry Period': self.duration/60}

        # Append the new row to the DataFrame
        sink.loc[len(sink)] = new_row
        pd.DataFrame.to_csv(sink, source_csv_file, index=False)
        
    def add_hw_issue(self, start: int, end: int, source_id:str = None, destination_id:str = None, next_hop_id:str = None, new_next_hop_id:str = None):
        if source_id == None:
            source_id = random.choice(self.routers).id
        if destination_id == None:
            destination_id = random.choice(self.routers).id
        if next_hop_id == None:
            next_hop_id = random.choice(self.routers).id
        
        router = self.routers[self.get_router_index_from_id(source_id)]
        
        destination = self.routers[self.get_router_index_from_id(destination_id)]
        
        next_hop = self.routers[self.get_router_index_from_id(next_hop_id)]
        
        entry_index = next(index for index, value in enumerate(router.forward_table) if value.destination == destination.ip_address and value.next_hop == next_hop.ip_address)

        previous_element = router.forward_table[entry_index]
        
        if new_next_hop_id != None:
            new_next_hop = self.routers[self.get_router_index_from_id(new_next_hop_id)]
            new_element = ForwardTableElement(dest=previous_element.destination, next_hop=new_next_hop.ip_address)
        else:
            neighbor_links = [link for link in self.links if link.source == router.ip_address]
            random_next_hop = random.choice([r.ip_address for r in self.routers if r.ip_address in [link.destination for link in neighbor_links] and r.ip_address != previous_element.next_hop])
            new_element = ForwardTableElement(dest=previous_element.destination, next_hop=random_next_hop)
            
        self.hw_issue.loc[len(self.hw_issue)] = {'Router ID': source_id, 'Start': start, 'End': end, 'Previous Table Element': previous_element.to_json(), 'New Table Element': new_element.to_json(), 'Entry Index': entry_index}
        
    def hw_update_interface(self, row: pd.Series):
        router = self.routers[self.get_router_index_from_id(row['Router ID'])]
        router.forward_table[row['Entry Index']] = ForwardTableElement(dest=row['New Table Element']["destination"], next_hop=row['New Table Element']["next_hop"])
        
    def hw_revert_interface(self, row: pd.Series):
        router = self.routers[self.get_router_index_from_id(row['Router ID'])]
        router.forward_table[row['Entry Index']] = ForwardTableElement(dest=row['Previous Table Element']["destination"], next_hop=row['Previous Table Element']["next_hop"])

    def all_ipm_session(self, flow_label:int = None):

        start = time.time()
        
        for gen_number in range(self.num_generation):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with ThreadPoolExecutor() as executor:
                # Submit tasks for each source
                futures = []
                
                for source in self.routers:
                    for destination in self.routers:
                        if source != destination:
                            future = executor.submit(
                                    self.send_probs, source, destination, flow_label)
                            futures.append(future)

                # Wait for all tasks to complete
                for future in futures:
                    result =  future.result()
                    self.export_sink_data(result[0].ip_address, result[1].ip_address, flow_label, result[2], gen_number, timestamp)
                    self.export_source_data(result[0].ip_address, result[1].ip_address, flow_label, gen_number, timestamp)

        print("Emulated all ipm sessions for " + str(self.num_generation) + " generations with " +
              str(self.generation_rate*self.duration) + " probes in " + str(time.time() - start))
        
    def send_prob(self, source: Router, destination: Router, flow_label: int):
        # Initialize a latency array
        latency = 0
        # For each generation, calculate the latency from source to destination
        indices = [index for index, value in enumerate(
        source.forward_table) if value.destination == destination.ip_address]
        indices = self.get_multi_links(source, indices)
        # Using hash function to determine which route to take
        chosen_route = source.select_route(source.ip_address,
                destination.ip_address, len(indices), flow_label)
        index = next(index for index, value in enumerate(self.links) if value.source ==
                        source.ip_address and value.destination == source.forward_table[indices[chosen_route]].next_hop)
        latency += self.links[index].delay

        # Find the next router
        index = next(index for index, value in enumerate(
            self.routers) if value.ip_address == source.forward_table[indices[chosen_route]].next_hop)
        next_router = self.routers[index]

        # Continue finding the next hop and adding the delay until the destination is reached
        while (next_router.ip_address != destination.ip_address):
            indices = [index for index, value in enumerate(
                next_router.forward_table) if value.destination == destination.ip_address]
            chosen_route = next_router.select_route(source.ip_address,
                destination.ip_address, len(indices), flow_label)
            index = next(index for index, value in enumerate(self.links) if value.source ==
                            next_router.ip_address and value.destination == next_router.forward_table[indices[chosen_route]].next_hop)
            
            latency += self.links[index].delay
            index = next(index for index, value in enumerate(
                self.routers) if value.ip_address == next_router.forward_table[indices[0]].next_hop)
            next_router = self.routers[index]
        # Return the latency array
        return latency
    
    def send_probs(self, source:Router, destination: Router, flow_label: int):
        # Initialize a latency array
        latencies = []
        fl = flow_label
        
        start_time = []
        end_time = []
        
        for _, row in self.hw_issue.iterrows():
            start_time.append(row['Start'])
            end_time.append(row['End'])
        # For each generation, calculate the latency from source to destination
        for t in range(self.generation_rate*self.duration):
            start_indices = [index for index, value in enumerate(start_time) if value == t]
            end_indices = [index for index, value in enumerate(end_time) if value == t]
            for s in start_indices:
                self.hw_update_interface(self.hw_issue.iloc[s])
            for e in end_indices:
                self.hw_revert_interface(self.hw_issue.iloc[e])
            if flow_label == None:
                fl = random.randint(0, 32)
            latencies.append(self.send_prob(source, destination, fl))
        # Return the latency array
        return [source, destination, latencies]

    def get_router_index_from_ip(self, ip_address:str) -> int:
        for i in range(len(self.routers)):
            if (self.routers[i].ip_address == ip_address):
                return i
    
    def get_router_index_from_id(self, router_id:str) -> int:
        for i in range(len(self.routers)):
            if (self.routers[i].id == router_id):
                return i
    
    def get_router_by_name(self, name: str ) -> Router:
        for router in self.routers:
            if router.node_name == name:
                return router
        return None

            
    def ipm_session(self, source_id: str, destination_id: str,gen_number: int = 0, fl: int = None, t:str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        source = self.routers[self.get_router_index_from_id(source_id)]
        destination = self.routers[self.get_router_index_from_id(destination_id)]
        latencies = self.send_probs(source, destination, fl)[2]
        self.export_source_data(source=source.ip_address, destination=destination.ip_address,fl=fl, gen_number=gen_number, timestamp=t)
        self.export_sink_data(source=source.ip_address, destination=destination.ip_address,fl=fl, latencies=latencies, gen_number=gen_number, timestamp=t)
        
    def get_multi_links(self, source:Router, indices:List[int]) -> List[int]:
        new_indices = []
        for i in indices:
            next_hop = source.forward_table[i].next_hop
            link_indices = [index for index, value in enumerate(
                self.links) if value.destination == next_hop and value.source == source.ip_address]
            min_cost = min([self.links[index].cost for index in link_indices])
            min_cost_count = sum(1 for index in link_indices if self.links[index].cost == min_cost)
            for _ in range(min_cost_count-1):
                new_indices.append(i)
        indices += new_indices
        return indices
        

    def ecmp_analysis(self, source_id: str, show: bool = False):
        source = self.routers[self.get_router_index_from_id(source_id)]
        ecmp_dict = {}
        for destination in self.routers:
            if destination != source:
                indices = [index for index, value in enumerate(
                    source.forward_table) if value.destination == destination.ip_address]
                # Add a duplicate indices if there's two links to the same next_hop"
                indices = self.get_multi_links(source, indices)
                counter = 0
                for i in indices:
                    counter += self.get_number_of_paths(source, destination, i)
                if counter in ecmp_dict:
                    ecmp_dict[counter] += 1
                else:   
                    ecmp_dict[counter] = 1 
                    
        if show:
            plt.bar(ecmp_dict.keys(), ecmp_dict.values())
            plt.xlim(0, max(ecmp_dict.keys())+max(ecmp_dict.keys())/10)
            plt.xlabel('Number of paths')
            plt.ylabel('Number of destinations')
            plt.title('ECMP analysis for router ' + source_id)
            plt.show()   
            
        print(f"ECMP analysis for router {source_id}:\n Max number of paths: {max(ecmp_dict.keys())}\n Min number of paths: {min(ecmp_dict.keys())}\n Average number of paths: {sum([key*value for key, value in ecmp_dict.items()])/sum(ecmp_dict.values())}\n")       
                    
    def get_number_of_paths(self, source:Router, destination:Router, index:int):
        next_hop = source.forward_table[index].next_hop
        next_router = self.routers[self.get_router_index_from_ip(next_hop)]
        indices = [index for index, value in enumerate(
            next_router.forward_table) if value.destination == destination.ip_address]
        indices = self.get_multi_links(next_router, indices)
        
        counter = 0
        if next_router.ip_address == destination.ip_address:
            return 1
        for i in indices:
            counter += self.get_number_of_paths(next_router, destination, i)
        return counter
    
    def all_latency_test(self):
        for source in self.routers:
            for destination in self.routers:
                if source != destination:
                    self.latency_test(source.id, destination.id)
    
    def latency_test(self, source_id: str, destination_id: str):
        latencies =[]
        paths = []
        wrong_paths = []
        
        source:Router = self.routers[self.get_router_index_from_id(source_id)]
        destination:Router = self.routers[self.get_router_index_from_id(destination_id)]
        
        indices = [index for index, value in enumerate(
                source.forward_table) if value.destination == destination.ip_address]
        indices = self.get_multi_links(source, indices)
        for i in range(len(indices)):
            path, latency = self.get_latency_and_path(source, destination, i, [source.forward_table[i].next_hop], 0)
            paths.append(path)
            latencies.append(latency)
        
        min_latency = min(latencies)
        
        for i in range(len(latencies)):
            if latencies[i] >= min_latency + self.treshold*1000:
                wrong_paths.append((paths[i], latencies[i]-min_latency))
        
        print(f" {source.ip_address} to {destination.ip_address}, there's {len(paths)} paths, there's {len(wrong_paths)} paths with the latency difference: {wrong_paths}")
            
    
    def get_latency_and_path(self, current: Router, destination: Router, index: int, path: list, latency: int):
        next_hop = current.forward_table[index].next_hop
        next_router:Router = self.routers[self.get_router_index_from_ip(next_hop)]
        indices = [index for index, value in enumerate(
            next_router.forward_table) if value.destination == destination.ip_address]
        indices = self.get_multi_links(next_router, indices)
        if next_router.ip_address == destination.ip_address:
            return path, latency
        for i in indices:
            path.append(next_router.forward_table[i].next_hop)
            latency += self.links[next(index for index, value in enumerate(self.links) if value.source == next_router.ip_address and value.destination == next_router.forward_table[i].next_hop)].delay
            return self.get_latency_and_path(next_router, destination, i, path, latency)
    
    def hw_issue_detection(self, source_measure_file: str, sink_measure_file:str, latency:bool = False, loss:bool = False):
        sink_dataframe = pd.read_csv(sink_measure_file)
        source_dataframe = pd.read_csv(source_measure_file)   
        if latency:
            for _, row in sink_dataframe.iterrows():
                if row['D flag'] == True :
                    source_router = self.routers[self.get_router_index_from_ip(row['Source'])]
                    destination_router = self.routers[self.get_router_index_from_ip(row['Destination'])]
                    indices = [index for index, value in enumerate(
                        source_router.forward_table) if value.destination == destination_router.ip_address]
                    routers_id: List[str] = []
                    self.path_hw_detection(source=source_router, destination=destination_router, index=indices[0], routers_id=routers_id, sink_dataframe=sink_dataframe)
                    if len(routers_id) == 0: 
                        routers_id.append(source_router.id)
                    print(f"HW issue on: {routers_id}")
        elif loss:
            pass
        
        return 
    
    def path_hw_detection(self, source: Router, destination: Router, index: int, routers_id: List[str], sink_dataframe: pd.DataFrame):
        next_hop = source.forward_table[index].next_hop
        next_router = self.routers[self.get_router_index_from_ip(next_hop)]
        indices = [index for index, value in enumerate(
            next_router.forward_table) if value.destination == destination.ip_address]
        if next_router.ip_address == destination.ip_address:
            return
        if sink_dataframe[(sink_dataframe['Source'] == next_router.ip_address) & (sink_dataframe['Destination'] == destination.ip_address)]['D flag'].all() == False:
            routers_id.append(source.id)
            return
        for i in indices:
            return self.path_hw_detection(next_router, destination, i, routers_id, sink_dataframe)