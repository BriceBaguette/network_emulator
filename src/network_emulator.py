# Import necessary modules
from router import Router, ForwardTableElement
from link import Link
import numpy as np
import json
import base64
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import networkx as nx
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

# Define the NetworkEmulator class


class NetworkEmulator:

    # Initialize the NetworkEmulator with node file, link file, generation rate, and number of generations
    def __init__(self, node_file, link_file, generation_rate, num_generation, max_fib_break = 2, fib_break_spread = 1):
        self.routers = list()  # List to store routers
        self.links = list()  # List to store links

        self.node_file = node_file  # Node file
        self.link_file = link_file  # Link file
        self.generation_rate = generation_rate  # Generation rate
        self.num_generation = num_generation  # Number of generations
        
        self.net_graph = None # Network graph
        self.G = None # Networkx graph
        
        self.router_failure = False # Status of router failure
        self.link_failure = False # Status of link failure
        self.link_failed = [] # List of failed links
        
        self.max_fib_break = max_fib_break # Maximum number of FIB breaks
        self.fib_break_spread = fib_break_spread # Spread of FIB breaks, between 0 and 1 (default: 1)


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

    def emulate(self, source, destination):
        # Generate traffic between the source and destination for a number of generations
        start = time.time()
        for _ in range(self.num_generation):
            index = next(index for index, value in enumerate(
                self.routers) if value.ip_address == source)
            source_router = self.routers[index]
            latencies = self.send_probs(source=source_router, destination=destination)
        end = time.time()
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
        latency = np.zeros(self.generation_rate)
        base_pkt_nmbr = random.randint(1, 100000000)
        # For each generation, calculate the latency from source to destination
        for i in range(self.generation_rate):
            # Look if we create a link failure
            if (not self.router_failure) and  len(self.link_failed) < self.max_fib_break:
                self.create_link_failure()
                
            if len(self.link_failed) > 0:
                self.restore_link_failure()
                    
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
        plt.xlabel('Value')
        plt.ylabel('Occurrences')
        plt.title('Occurrences of Values in List')
        plt.show()

    def create_link_failure(self):
        for link in self.links:
            if len(self.link_failed) < self.max_fib_break:
                x = stats.norm.rvs()
                # Compute the confidence interval
                confidence_level = 0.995
                confidence_interval = stats.norm.interval(confidence_level,)
                if x < confidence_interval[0] or x > confidence_interval[1]:
                    self.link_failed.append(link)
            
    def restore_link_failure(self):
        indices = []
        for i in range(len(self.link_failed)):
            link = self.link_failed[i]
            x = stats.norm.rvs()
            # Compute the confidence interval
            confidence_level = 0.1 * (1+ link.break_time/30)
            
            if confidence_level > 1:
                confidence_level = 1
                
            confidence_interval = stats.norm.interval(confidence_level)
            
            if  confidence_interval[0] < x < confidence_interval[1]:
                indices.append(i)
                
            link.break_time += 1
        indices.sort(reverse=True) 
        for i in indices:
            link = self.link_failed.pop(i)
            self.G.add_edge(self.get_router_index(link.source), self.get_router_index(link.destination), weight=link.cost)
            
            
        