# Import necessary modules
from router import Router, ForwardTableElement
from link import Link
import utils
import numpy as np
import json
import base64
from concurrent.futures import ProcessPoolExecutor
import time
import networkx as nx

# Define the NetworkEmulator class


class NetworkEmulator:

    # Initialize the NetworkEmulator with node file, link file, generation rate, and number of generations
    def __init__(self, node_file, link_file, generation_rate, num_generation):
        self.routers = list()  # List to store routers
        self.links = list()  # List to store links

        self.node_file = node_file  # Node file
        self.link_file = link_file  # Link file
        self.generation_rate = generation_rate  # Generation rate
        self.num_generation = num_generation  # Number of generations

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
        net_graph = np.zeros((number_of_routers, number_of_routers), dtype=int)

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
                net_graph[i, destination] = self.links[value].cost

        # Print the time taken to build the graph
        end = time.time()
        print(net_graph)
        print("Build graph in: {}".format(end - start))
        # create a networkx graph
        G = nx.Graph(net_graph)
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
        for i in range(number_of_routers):
            for j in range(number_of_routers):
                if i != j:
                    # Find the shortest paths from the source to the destination
                    shortest_paths = list(
                        nx.all_shortest_paths(G, source=i, target=j))
                    for path in shortest_paths:
                        # Update the forward table for the source router
                        self.routers[i].updateForwardTable(ForwardTableElement(
                            dest=self.routers[path[len(path)-1]].ip_address,
                            next_hop=self.routers[path[1]].ip_address
                        ))

        # Print the time taken to start the network
        end = time.time()
        print("Network started in: {}".format(end - start))

    def emulate(self, source, destination):
        # Generate traffic between the source and destination for a number of generations
        for _ in range(self.num_generation):
            index = next(index for index, value in enumerate(
                self.routers) if value.ip_address == source)
            source_router = self.routers[index]
            self.send_probs(source=source_router, destination=destination)

    def send_probs(self, source, destination):
        # Initialize a latency array
        latency = np.zeros(self.generation_rate)

        # For each generation, calculate the latency from source to destination
        for i in range(self.generation_rate):
            # Find the next hop from the source to the destination
            indices = [index for index, value in enumerate(
                source.forward_table) if value.destination == destination]
            index = next(index for index, value in enumerate(self.links) if value.source ==
                         source.ip_address and value.destination == source.forward_table[indices[0]].next_hop)
            latency[i] += self.links[index].delay

            # Find the next router
            index = next(index for index, value in enumerate(
                self.routers) if value.ip_address == source.forward_table[indices[0]].next_hop)
            next_router = self.routers[index]

            # Continue finding the next hop and adding the delay until the destination is reached
            while (next_router.ip_address != destination):
                indices = [index for index, value in enumerate(
                    next_router.forward_table) if value.destination == destination]
                index = next(index for index, value in enumerate(self.links) if value.source ==
                             next_router.ip_address and value.destination == next_router.forward_table[indices[0]].next_hop)
                latency[i] += self.links[index].delay
                index = next(index for index, value in enumerate(
                    self.routers) if value.ip_address == next_router.forward_table[indices[0]].next_hop)
                next_router = self.routers[index]

        # Return the latency array
        return latency
