# Import necessary modules
from router import Router, ForwardTableElement
from link import Link
import numpy as np
import json
import base64
import random
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import networkx as nx
from tqdm import tqdm
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Define the NetworkEmulator class


class NetworkEmulator:

    # Initialize the NetworkEmulator with node file, link file, generation rate, and number of generations
    def __init__(self, node_file, link_file, generation_rate, num_generation, duration, max_fib_break = 1, fib_break_spread = 1):
        self.routers = list()  # List to store routers
        self.links = list()  # List to store links
        
        self.node_file = node_file  # Node file
        self.link_file = link_file  # Link file
        self.generation_rate = generation_rate  # Generation rate
        self.num_generation = num_generation  # Number of generations
        self.duration = duration  # Duration of the simulation  
        
        self.net_graph = None # Network graph
        self.G = None # Networkx graph
        self.net_state_graph = None # Network state matrix
        
        self.router_failure = False # Status of router failure
        self.link_failure = False # Status of link failure
        self.link_failed = [] # List of failed links
        
        self.max_fib_break = max_fib_break # Maximum number of FIB breaks
        self.fib_break_spread = fib_break_spread # Spread of FIB breaks, between 0 and 1 (default: 1)
        
        self.prob_start = 0
        self.link_failure_time = []
        
        self.session_id = None


    # Method to build the network
    def build(self):
        self.__build_routers()  # Build routers
        self.__build_links()  # Build links
        # Print the number of routers and links in the network
        print("Network build with " + str(len(self.routers)) +
              " routers and " + str(len(self.links)) + " links")

    def __build_routers(self):
        # Open the node file and load the JSON data
        with open(self.node_file, 'r') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                # Handle JSON parsing errors
                print(f"Error: JSON parsing failed for file '{
                      self.node_file}' ")
                return 1

        # Iterate over each object in the JSON data
        for obj in json_data:
            # Extract necessary information from the object
            attributes_obj = obj.get('attributes', {})
            ls_attribute_obj = attributes_obj.get('ls_attribute', {})
            node_name_obj = ls_attribute_obj.get('node_name', {})
            node_name = node_name_obj.strip() if node_name_obj else None

            nlri = obj.get('nlri', {}).get('nlri_data', {}).get('lsnlri', {})
            local_node_descriptor = nlri.get('local_node_descriptor', {})
            igp_router_id = local_node_descriptor.get(
                'igp_router_id', {}).get('$binary', {}).get('base64', None)
            base64_str = igp_router_id.strip() if igp_router_id else None

            # Create a new Router object and append it to the routers list
            self.routers.append(
                Router(node_name=node_name, active=1, ip_address=base64_str))

    def __build_links(self):
        # Open the link file and load the JSON data
        with open(self.link_file, 'r') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                # Handle JSON parsing errors
                print(f"Error: JSON parsing failed for file '{
                      self.link_file}'")
                return 1

        # Iterate over each object in the JSON data
        for obj in json_data:
            # Initialize source, destination, delay, and igp_metric
            source, destination, delay, igp_metric = None, None, 0, 0

            # Extract necessary information from the object
            nlri = obj.get('nlri', {}).get('nlri_data', {}).get('lsnlri', {})
            local_node_descriptor = nlri.get('local_node_descriptor', {})
            remote_node_descriptor = nlri.get('remote_node_descriptor', {})
            attributes_obj = obj.get('attributes', {})
            ls_attribute_obj = attributes_obj.get('ls_attribute', {})

            # Get source and destination from the descriptors
            source = local_node_descriptor.get('igp_router_id', {}).get(
                '$binary', {}).get('base64', None)
            destination = remote_node_descriptor.get(
                'igp_router_id', {}).get('$binary', {}).get('base64', None)

            # Get igp_metric and delay from the attributes
            igp_metric = ls_attribute_obj.get('igp_metric', 0)
            delay = ls_attribute_obj.get('unidir_delay', {}).get('delay', 0)

            # Create a new Link object and append it to the links list
            link = Link(source=source, destination=destination,
                        delay=delay, cost=igp_metric)
            self.links.append(link)

    def start(self):
        # Record the start time and get the number of routers
        start = time.time()
        number_of_routers = len(self.routers)

        # Construct a network graph as a sparse matrix

        # Create the array
        net_graph = np.empty((number_of_routers, number_of_routers), dtype=object)
        for i in range(number_of_routers):
            for j in range(number_of_routers):
                net_graph[i, j] = []
                
        # Create the array
        self.net_state_graph = np.empty((number_of_routers, number_of_routers), dtype=object)
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
                self.net_state_graph[i,destination].append(1)
                
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
        self.G = G
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
        with ThreadPoolExecutor() as executor:
                # Use submit to asynchronously submit tasks to the executor
                futures = []
                for i in range(number_of_routers):
                    for j in range(number_of_routers):
                        if i != j:
                            futures.append(executor.submit(self.update_forward_table, G, i, j))

                # Wait for all tasks to complete
                progress_bar = tqdm(total=len(futures), desc="Processing")
                
                for future in futures:
                    future.result()
                    progress_bar.update(1)
                    
                progress_bar.close()


        # Print the time taken to start the network
        end = time.time()
        print("Network started in: {}".format(end - start))
        
    def update_forward_table(self, G, source, target):
    # Find the shortest paths from the source to the destination
        if not self.routers[source].has_entry_for_destination(self.routers[target].ip_address):
            shortest_paths = list(nx.all_shortest_paths(G, source=source, target=target, weight='weight'))
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

    def export_sink_data(self, source, destination, time_array, latencies, gen_number):
        
        # Export the data to a CSV file
        sink_csv_file = './src/results/sink.csv'
        
        if os.path.isfile(sink_csv_file):
            sink = pd.read_csv  (sink_csv_file)
            if self.session_id == None:
                self.session_id = sink.iloc[-1]['Session ID'] + 1
        else: 
            sink = pd.DataFrame(columns=['Session ID', 'Source', 'Destination', 'TimeStamp' ,'Histogram template', 'Histogram values', 'Histogram type', 'Liveness flag'])
            self.session_id = 1
        histogram_type = None
        if gen_number%2 == 0:
            histogram_type = 'Black'
        else:
            histogram_type = 'White'

        source = source
        destination = destination

        new_row = {'Session ID': self.session_id,
                'Source': source,
                'Destination': destination,
                'TimeStamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Histogram template': time_array,
                'Histogram values': latencies,
                'Histogram type': histogram_type,
                'Liveness flag': True}

        # Append the new row to the DataFrame
        sink.loc[len(sink)] = new_row
        pd.DataFrame.to_csv(sink, sink_csv_file,index=False)
        
    def export_source_data(self, source, destination, gen_number):
                # Export the data to a CSV file
        source_csv_file = './src/results/source.csv'
        
        if os.path.isfile(source_csv_file):
            sink = pd.read_csv (source_csv_file)
            if self.session_id == None:
                self.session_id = sink.iloc[-1]['Session ID'] + 1
        else: 
            sink = pd.DataFrame(columns=['Session ID', 'Source', 'Destination', 'TimeStamp' ,'Probe GenRate', 'Packet Counter', 'Packet Counter Type'])
            self.session_id = 1
        packet_type = None
        if gen_number%2 == 0:
            packet_type = 'Black'
        else:
            packet_type = 'White'

        source = source
        destination = destination
        
        new_row = {'Session ID': self.session_id,
                'Source': source,
                'Destination': destination,
                'TimeStamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Probe GenRate': self.generation_rate,
                'Packet Counter': self.generation_rate * self.duration,
                'Packet Counter Type': packet_type}

        # Append the new row to the DataFrame
        sink.loc[len(sink)] = new_row
        pd.DataFrame.to_csv(sink, source_csv_file,index=False)
        

    def emulate(self, source, destination):
        # Generate traffic between the source and destination for a number of generations
        start = time.time()
        for gen_number in range(self.num_generation):
            index = next(index for index, value in enumerate(
                self.routers) if value.ip_address == source)
            source_router = self.routers[index]
            time_array = np.linspace(0,self.duration,self.generation_rate)
            latencies = self.send_probs(source=source_router, destination=destination)
            self.export_sink_data(source, destination, time_array, latencies, gen_number)
            self.export_source_data(source,destination,gen_number)
        end = time.time()
        if len(self.link_failure_time) > 0:
            print(f"Min time: {min(self.link_failure_time)}, Max time: {max(self.link_failure_time)}, average time {sum(self.link_failure_time)/len(self.link_failure_time)}, number of breaks {len(self.link_failure_time)}")
        print(f"Network emulated in: {end-start} for {self.num_generation} generations of {self.generation_rate} probes")
        
    def emulate_all(self):
        
        start = time.time()
        for _ in range(self.num_generation):
            with ThreadPoolExecutor() as executor:
                # Submit tasks for each source
                futures = []
                for source in self.routers:
                    for destination in self.routers:
                        if source != destination:
                            future = executor.submit(self.send_probs, source, destination.ip_address)
                            futures.append(future)

            # Wait for all tasks to complete
            for future in futures:
                future.result()

        print("Emulated all network for " + str(self.num_generation) + " generations with " + str(self.generation_rate) + " probes in " + str(time.time() - start))
        
    def get_router_index(self, ip_address):
        for i in range(len(self.routers)):
            if(self.routers[i].ip_address == ip_address):
                return i

    def send_probs(self, source, destination):
        # Initialize a latency array
        latency = np.zeros(self.generation_rate* self.duration)
        base_pkt_nmbr = random.randint(1, 100000000)
        # For each generation, calculate the latency from source to destination
        for i in range(self.generation_rate * self.duration):
            # Look if we create a link failure
            if (not self.router_failure) and  len(self.link_failed) < self.max_fib_break:
                self.create_link_failure(i)
                
            if len(self.link_failed) > 0:
                self.restore_link_failure(i)
                    
            # Find the next hop from the source to the destination
            indices = [index for index, value in enumerate(
                source.forward_table) if value.destination == destination]
            # Using hash function to determine which route to take
            chosen_route = source.select_route(destination, len(indices), base_pkt_nmbr + i)
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
                chosen_route = next_router.select_route(destination, len(indices), base_pkt_nmbr + i)
                index = next(index for index, value in enumerate(self.links) if value.source ==
                             next_router.ip_address and value.destination == next_router.forward_table[indices[chosen_route]].next_hop)
                # Look different case based on the link failure or not
                if not self.links[index] in self.link_failed:
                    latency[i] += self.links[index].delay
                    index = next(index for index, value in enumerate(
                        self.routers) if value.ip_address == next_router.forward_table[indices[0]].next_hop)
                    next_router = self.routers[index]
                elif self.G.has_edge(self.get_router_index(next_router.ip_address),self.get_router_index(next_router.forward_table[indices[chosen_route]].next_hop)):
                    self.G.remove_edge(self.get_router_index(next_router.ip_address),self.get_router_index(next_router.forward_table[indices[chosen_route]].next_hop))
                    new_path = list(nx.shortest_path(self.G, source=self.get_router_index(next_router.ip_address), target=self.get_router_index(destination), weight='weight'))
                    for j in range(len(new_path)-1):
                        index = next(index for index, value in enumerate(self.links) if value.source ==
                                self.routers[new_path[j]].ip_address and value.destination == self.routers[new_path[j+1]].ip_address)
                        latency[i] += self.links[index].delay
                    break
                else:
                    new_path =  list(nx.shortest_path(self.G, source=self.get_router_index(next_router.ip_address), target=self.get_router_index(destination), weight='weight'))
                    for j in range(len(new_path)-1):
                        index = next(index for index, value in enumerate(self.links) if value.source ==
                                self.routers[new_path[j]].ip_address and value.destination == self.routers[new_path[j+1]].ip_address)
                        latency[i] += self.links[index].delay
                    break
        # Return the latency array
        return latency

    def latency_histogram(self, data):
        
        value_counts = {}
        for item in data:
            if item in value_counts:
                value_counts[item] += 1
            else:
                value_counts[item] = 1

        # Extract the values and counts
        values = list(value_counts.keys())
        counts = list(value_counts.values())

        # Plot the data
        plt.bar(values, counts)
        plt.xlabel('Latency in ms')
        plt.ylabel('Number of probes')
        plt.title('Number of probes vs Latency')
        plt.show()

    def create_link_failure(self, i):
        confidence_level = self.get_confidence_level()
        for link in self.links:
            if len(self.link_failed) < self.max_fib_break:
                x = stats.norm.rvs()
                # Compute the confidence interval

                confidence_interval = stats.norm.interval(confidence_level,)
                if x < confidence_interval[0] or x > confidence_interval[1]:
                    self.prob_start = i
                    self.link_failed.append(link)
                    link.failure = 1
                    
    def get_confidence_level(self):
        return 1 - (self.duration * len(self.links))/100000
        

    def restore_link_failure(self, j):
        indices = []
        for i in range(len(self.link_failed)):
            link = self.link_failed[i]
            x = stats.norm.rvs()
            # Compute the confidence interval
            confidence_level = 0.05 * (1+ link.break_time/100**link.failure)
            
            if confidence_level > 1:
                confidence_level = 1
                
            confidence_interval = stats.norm.interval(confidence_level)
            
            if  confidence_interval[0] < x < confidence_interval[1]:
                self.link_failure_time.append(j-self.prob_start)
                indices.append(i)
                
            link.break_time += 1
        indices.sort(reverse=True) 
        for i in indices:
            link = self.link_failed.pop(i)
            self.G.add_edge(self.get_router_index(link.source), self.get_router_index(link.destination), weight=link.cost)
            
    def output_matrix_reset(self):
        for i in range(len(self.net_state_graph)):
            for j in range(len(self.net_state_graph[i])):
                for k in range(len(self.net_state_graph[i, j])):
                    if self.net_state_graph[i, j][k] != 0:
                        self.net_state_graph[i, j][k] = 1
            
            
        