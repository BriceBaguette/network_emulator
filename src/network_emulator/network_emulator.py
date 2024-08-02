"""
This module defines the Network Emulator class, which can represent a network,
test his resilience and generate data for the network.
"""
import ast
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from network_emulator.link import Link
from network_emulator.router import Router, ForwardTableElement


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

    def __init__(self, node_file: str, link_file: str, single_file: str, generation_rate: int,
                 num_generation: int, load_folder: str = None, save_folder: str = None,
                 threshold: int = 5):
        """
        Initialize the NetworkEmulator with node file, link file, generation rate, 
        and number of generations.

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
        self.threshold = threshold
        self.duration = 300
        self.current_step = 0

        self.net_graph = None
        self.g = None
        self.net_state_graph = None
        self.__is_running = False
        self.working = False

        self.session_id = None
        self.load_folder = load_folder
        self.save_folder = save_folder

        self.hw_issue: pd.DataFrame = pd.DataFrame(
            columns=['Router ID', 'Start', 'End', 'Previous Table Element',
                     'New Table Element', 'Entry Index'])

    def is_running(self) -> bool:
        """
        Check if the network is running.

        Returns:
            bool: True if the network is running, False otherwise.
        """
        return self.__is_running

    def get_router_from_id(self, router_id) -> Router:
        """
        Retrieve a router object based on its ID.

        Args:
            router_id (int): The ID of the router to retrieve.

        Returns:
            Router: The router object with the specified ID, or None if no such router exists.
        """
        for router in self.routers:
            if router.id == router_id:
                return router
        return None
    
    def get_router_from_ip(self, router_ip) -> Router:
        """
        Retrieve a router object based on its ip address.

        Args:
            router_ip (int): The ip address of the router to retrieve.

        Returns:
            Router: The router object with the specified ID, or None if no such router exists.
        """
        for router in self.routers:
            if router.ip_address == router_ip:
                return router
        return None

    def get_routers_ids(self) -> List[str]:
        """
        Retrieve a list of all router IDs in the network.

        Returns:
            List[str]: A list of IDs for all routers in the network.
        """
        return [router.id for router in self.routers]

    def get_topology(self) -> nx.MultiDiGraph:
        """
        Retrieve the network topology as a directed multigraph.

        Returns:
            nx.MultiDiGraph: The network topology represented as a directed multigraph.
        """
        return self.g

    def build(self) -> None:
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
        with open(self.single_file, "r", encoding="utf-8") as file:
            topo = file.read()

        # Find all positions of router sections
        router_start_positions = [match.start()
                                  for match in re.finditer(r'^\S', topo, re.MULTILINE)]

        # Split content into router sections
        routers_info = [topo[start:router_start_positions[i + 1]]
                        if i + 1 < len(router_start_positions)
                        else topo[start:] for i, start in enumerate(router_start_positions)]

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
                self.routers.append(Router(
                    router_id=router_id, node_name=hostname, ip_address=ip_address, active=1))

        pattern = re.compile(
            r'Metric:\s*\d+\s+IS-Extended.*?(?=Metric:|\Z)', re.DOTALL)

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
                    neighbor_match = re.search(
                        neighbor_pattern, match, re.DOTALL)
                    delay_match = re.search(delay_pattern, match)
                    if neighbor_match and delay_match:
                        neighbor = self.get_router_by_name(
                            neighbor_match.group(2))
                        if neighbor:
                            link = Link(link_id=len(self.links), source=source.ip_address,
                                        destination=neighbor.ip_address,
                                        delay=int(delay_match.group(1)),
                                        cost=int(neighbor_match.group(1)))
                            self.links.append(link)

    def __load_network(self):
        """
        Load the network from saved files.
        """
        with open(f"{self.load_folder}/routers.json", 'r', encoding="utf-8") as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error: JSON parsing failed for file '{
                      self.load_folder}'/routers.json")
                return 1

        for obj in json_data:
            node_id = obj.get('id', None)
            node_name = obj.get('node_name', None)
            active = obj.get('active', 1)
            ip_address = obj.get('ip_address', None)
            self.routers.append(Router(node_name=node_name,
                                       router_id=node_id, active=active, ip_address=ip_address))

            for element in obj.get('forward_table', []):
                dest = element.get('destination', None)
                next_hop = element.get('next_hop', None)
                self.routers[-1].update_forward_table(
                    ForwardTableElement(dest=dest, next_hop=next_hop))

        with open(f"{self.load_folder}/links.json", 'r', encoding="utf-8") as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error: JSON parsing failed for file '{
                      self.link_file}'")
                return 1

        for obj in json_data:
            link_id = obj.get('id', None)
            source = obj.get('source', None)
            destination = obj.get('destination', None)
            delay = obj.get('delay', 0)
            cost = obj.get('cost', 0)

            self.links.append(
                Link(link_id=link_id, source=source,
                     destination=destination, delay=delay, cost=cost))

    def __build_routers(self):
        """
        Build routers from the node file.
        """
        with open(self.node_file, 'r', encoding="utf-8") as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error: JSON parsing failed for file '{
                      self.node_file}' ")
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
        with open(self.link_file, 'r', encoding="utf-8") as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error: JSON parsing failed for file '{
                      self.link_file}'")
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

        with open(f"{self.save_folder}/routers.json", 'w', encoding="utf-8") as file:
            json.dump([router.to_json()
                      for router in self.routers], file, indent=4)

        with open(f"{self.save_folder}/links.json", 'w', encoding="utf-8") as file:
            json.dump([link.to_json() for link in self.links], file, indent=4)

    def start(self):
        """
        Initialize and start the network simulation.

        This method performs the following steps:
        1. Records the start time and gets the number of routers.
        2. Constructs a network graph as a sparse matrix.
        3. Fills the network graph with link costs.
        4. Creates a NetworkX graph from the sparse matrix.
        5. Prints the time taken to build the graph.
        6. If no load folder is specified, updates the forward table
           for each router pair asynchronously.
        7. Saves the network state if a save folder is specified.
        8. Prints the time taken to start the network and sets the network running state to True.

        Returns:
            None
        """
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
        g = nx.MultiDiGraph()

        # Add nodes to the graph
        g.add_nodes_from(range(len(net_graph)))

        # Add edges to the graph with weights
        for i, row in enumerate(net_graph):
            for j, col in enumerate(row):
                for _, weight in enumerate(col):
                    if weight != 0:
                        g.add_edge(i, j, weight=weight)
        self.net_graph = net_graph
        self.g = g
        # Print the time taken to build the graph
        end = time.time()
        print(f"Build graph in: {str(end - start)}")

        if self.load_folder is None:
            with ThreadPoolExecutor() as executor:
                # Use submit to asynchronously submit tasks to the executor
                futures = []
                for i in range(number_of_routers):
                    for j in range(number_of_routers):
                        if i != j:
                            futures.append(executor.submit(
                                self.update_forward_table, g, i, j))

                # Wait for all tasks to complete
                progress_bar = tqdm(total=len(futures), desc="Processing")

                for future in futures:
                    future.result()
                    progress_bar.update(1)

                progress_bar.close()

            if self.save_folder is not None:
                self.__save_network()

            # Print the time taken to start the network
        end = time.time()
        self.__is_running = True
        print(f"Network started in: {str(end - start)}")

    def update_forward_table(self, g: nx.MultiDiGraph, source: int, target: int):
        """
        Update the forward table for the routers in the network.

        This method finds the shortest paths from the source router to the target router
        and updates the forward table entries for the source router and all intermediate routers
        along the shortest paths.

        Args:
            g (nx.MultiDiGraph): The network graph.
            source (int): The index of the source router.
            target (int): The index of the target router.

        Returns:
            None
        """
        # Find the shortest paths from the source to the destination
        if not self.routers[source].has_entry_for_destination(self.routers[target].ip_address):
            shortest_paths: List[List[int]] = list(nx.all_shortest_paths(
                g, source=source, target=target, weight='weight'))
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

    def export_sink_data(self, sink_router: Router, fl, gen_number, timestamp,
                         vrf="Measurement") -> None:
        """
        Export the sink router data to a CSV file.

        This method collects histogram data from the sink router and exports it to a CSV file.
        It updates the session ID if necessary and appends the new data to the existing CSV file.

        Args:
            sink_router (Router): The sink router from which to collect data.
            fl (str): The flow label.
            gen_number (int): The generation number.
            timestamp (str): The timestamp of the data collection.
            vrf (str, optional): The VRF (Virtual Routing and Forwarding) instance. 
                                 Defaults to "Measurement".

        Returns:
            None
        """

        # Export the data to a CSV file
        sink_csv_file = './src/results/sink.csv'

        try:
            sink = pd.read_csv(sink_csv_file)
            if self.session_id is None:
                self.session_id = sink.iloc[-1]['Session ID'] + 1
        except (FileNotFoundError, pd.errors.EmptyDataError):
            sink = pd.DataFrame(columns=['Session ID', 'Source', 'Destination', 'FL', 'VRF',
                                         'TimeStamp', 'Histogram Template', 'Histogram Value',
                                         'Histogram type', 'Min', 'Max',
                                         'Liveness flag', 'Telemetry Period'])
            self.session_id = 1
        histogram_type = None
        if gen_number % 2 == 0:
            histogram_type = 'Black'
        else:
            histogram_type = 'White'

        if fl is None:
            fl = "Spray"
        else:
            fl = str(fl)

        bins, min_max = sink_router.get_bins()

        for source in bins.keys():
            histogram_template = [str(bin.start) + '-' + str(bin.end)
                                  for bin in bins[source]]
            histogram_value = [bin.value for bin in bins[source]]
            new_row = {'Session ID': self.session_id,
                       'Source': source,
                       'Destination': sink_router.ip_address,
                       'FL': fl,
                       'VRF': vrf,
                       'TimeStamp': timestamp,
                       'Histogram Template': histogram_template,
                       'Histogram Value': histogram_value,
                       'Histogram type': histogram_type,
                       'Min': min_max[source][0],
                       'Max': min_max[source][1],
                       'Liveness flag': True,
                       'Telemetry Period': self.duration/60}
            sink.loc[len(sink)] = new_row

        # Append the new row to the DataFrame
        pd.DataFrame.to_csv(sink, sink_csv_file, index=False)

    def export_source_data(self, source_router: Router, fl, gen_number, timestamp,
                           vrf="Measurement") -> None:
        """
        Export the source router data to a CSV file.

        This method collects data from the source router and exports it to a CSV file.
        It updates the session ID if necessary and appends the new data to the existing CSV file.

        Args:
            source_router (Router): The source router from which to collect data.
            fl (str): The flow label.
            gen_number (int): The generation number.
            timestamp (str): The timestamp of the data collection.
            vrf (str, optional): The VRF (Virtual Routing and Forwarding) instance. 
                                 Defaults to "Measurement".

        Returns:
            None
        """
        # Export the data to a CSV file
        source_csv_file = './src/results/source.csv'
        try:
            source_df = pd.read_csv(source_csv_file)
            if self.session_id is None:
                self.session_id = source_df.iloc[-1]['Session ID'] + 1
        except (FileNotFoundError, pd.errors.EmptyDataError):
            source_df = pd.DataFrame(columns=['Session ID', 'Source', 'Destination', 'FL', 'VRF',
                                              'TimeStamp', 'Probe GenRate', 'Packet Counter',
                                              'Packet Counter Type', 'Telemetry Period'])
            self.session_id = 1
        packet_type = None
        if gen_number % 2 == 0:
            packet_type = 'Black'
        else:
            packet_type = 'White'
        if fl is None:
            fl = "Spray"
        else:
            fl = str(fl)

        for destination in self.routers:
            bins, _ = destination.get_bins()
            if source_router.ip_address in bins.keys():
                new_row = {'Session ID': self.session_id,
                           'Source': source_router.ip_address,
                           'Destination': destination.ip_address,
                           'FL': fl,
                           'VRF': vrf,
                           'TimeStamp': timestamp,
                           'Probe GenRate': self.generation_rate,
                           'Packet Counter': self.generation_rate * self.duration,
                           'Packet Counter Type': packet_type,
                           'Telemetry Period': self.duration/60}
                source_df.loc[len(source_df)] = new_row

        # Append the new row to the DataFrame

        pd.DataFrame.to_csv(source_df, source_csv_file, index=False)

    def add_hw_issue(self, start: int, end: int, source_id: str = None, destination_id: str = None,
                     next_hop_id: str = None, new_next_hop_id: str = None) -> None:
        """
        Add a hardware issue to the network.

        This method simulates a hardware issue by modifying the forwarding table of a router.
        It updates the forwarding table entry for a specific destination and next hop, and 
        logs the change.

        Args:
            start (int): The start time of the hardware issue.
            end (int): The end time of the hardware issue.
            source_id (str, optional): The ID of the source router. 
                                       If None, a random router is chosen. Defaults to None.
            destination_id (str, optional): The ID of the destination router. 
                                            If None, a random router is chosen. Defaults to None.
            next_hop_id (str, optional): The ID of the current next hop router. 
                                         If None, a random next hop is chosen. Defaults to None.
            new_next_hop_id (str, optional): The ID of the new next hop router. 
                                             If None, a random next hop is chosen. Defaults to None.

        Returns:
            None
        """

        if source_id is None:
            source_id = random.choice(self.routers).id
        if destination_id is None:
            while destination_id == source_id or destination_id is None:
                destination_id = random.choice(self.routers).id
        router = self.routers[self.get_router_index_from_id(source_id)]

        destination = self.routers[self.get_router_index_from_id(
            destination_id)]

        if next_hop_id is None:
            next_hop_ips = [router.forward_table[index].next_hop for index in range(len(
                router.forward_table)) if router.forward_table[index].destination ==
                destination.ip_address]
            next_hop_ip = random.choice(next_hop_ips)
            next_hop_id = self.routers[self.get_router_index_from_ip(
                next_hop_ip)].id

        next_hop = self.routers[self.get_router_index_from_id(next_hop_id)]

        entry_index = next(index for index, value in enumerate(router.forward_table)
                           if value.destination == destination.ip_address
                           and value.next_hop == next_hop.ip_address)

        previous_element = router.forward_table[entry_index]

        new_element = None

        if new_next_hop_id is not None:
            new_next_hop = self.routers[self.get_router_index_from_id(
                new_next_hop_id)]
            new_element = ForwardTableElement(
                dest=previous_element.destination, next_hop=new_next_hop.ip_address)
        else:
            neighbor_links = [
                link for link in self.links if link.source == router.ip_address]
            random_next_hop = random.choice([r.ip_address for r in self.routers if r.ip_address in [
                                            link.destination for link in neighbor_links]
                and r.ip_address != previous_element.next_hop])
            new_element = ForwardTableElement(
                dest=previous_element.destination, next_hop=random_next_hop)

        self.hw_issue.loc[len(self.hw_issue)] = {'Router ID': source_id,
                                                 'Start': start,
                                                 'End': end,
                                                 'Previous Table Element': previous_element.to_json(
                                                 ),
                                                 'New Table Element': new_element.to_json(),
                                                 'Entry Index': entry_index}

        print(f"Hardware issue added to router {source_id} from {start} to {end} with destination {
              destination_id} and new next hop {new_element.next_hop} instead of {next_hop_id}")

    def hw_update_interface(self, row: pd.Series) -> None:
        """
        Update the hardware interface of a router.

        This method updates the forwarding table of a specified router based on the
        provided row data.
        It modifies the forwarding table entry at the specified index with the new
        destination and next hop.

        Args:
            row (pd.Series): A pandas Series containing the following keys:
                - 'Router ID': The ID of the router to update.
                - 'Entry Index': The index of the forwarding table entry to update.
                - 'New Table Element': A dictionary with keys 'destination' and
                'next_hop' representing the new forwarding table entry.

        Returns:
            None
        """

        router = self.routers[self.get_router_index_from_id(row['Router ID'])]
        router.forward_table[row['Entry Index']] = ForwardTableElement(
            dest=row['New Table Element']["destination"],
            next_hop=row['New Table Element']["next_hop"])

    def hw_revert_interface(self, row: pd.Series) -> None:
        """
        Revert the hardware interface of a router.

        This method reverts the forwarding table of a specified router to its previous
        state based on the provided row data.
        It modifies the forwarding table entry at the specified index with the previous
        destination and next hop.

        Args:
            row (pd.Series): A pandas Series containing the following keys:
                - 'Router ID': The ID of the router to update.
                - 'Entry Index': The index of the forwarding table entry to revert.
                - 'Previous Table Element': A dictionary with keys 'destination' and
                'next_hop' representing the previous forwarding table entry.

        Returns:
            None
        """
        router = self.routers[self.get_router_index_from_id(row['Router ID'])]
        router.forward_table[row['Entry Index']] = ForwardTableElement(
            dest=row['Previous Table Element']["destination"],
            next_hop=row['Previous Table Element']["next_hop"])

    def all_ipm_session(self, flow_label: int = None):
        """
        Emulate all IPM (Inter-Process Messaging) sessions.

        This method emulates IPM sessions for a specified number of generations. For each
        generation, it sends probes between all pairs of routers (excluding self-loops),
        exports sink and source data, and updates the bins for each router.

        Args:
            flow_label (int, optional): The flow label to be used for the probes. Defaults to None.

        Returns:
            str: A summary of the emulation process, including the number of generations,
            the total number of probes, and the duration of the emulation.
        """
        start = time.time()

        for gen_number in range(self.num_generation):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for source in self.routers:
                for destination in self.routers:
                    if source != destination:
                        self.send_probs(source, destination, flow_label)
            for router in self.routers:
                self.export_sink_data(
                    sink_router=router, fl=flow_label, gen_number=gen_number, timestamp=timestamp)
                self.export_source_data(
                    source_router=router, fl=flow_label, gen_number=gen_number, timestamp=timestamp)
                router.update_bins()

        return (
            f"Emulated all ipm sessions for {
                str(self.num_generation)} generations with "
            f"{str(self.generation_rate * self.duration)
               } probes in {str(time.time() - start)}"
        )

    def send_prob(self, source: Router, destination: Router, flow_label: int) -> int:
        """
        Send a probe from the source router to the destination router.

        This method calculates the latency from the source router to the destination router
        by traversing the network.
        It uses a hash function to determine the route to take at each hop and accumulates
        the delay for each link.

        Args:
            source (Router): The source router from which the probe is sent.
            destination (Router): The destination router to which the probe is sent.
            flow_label (int): The flow label used to determine the route.

        Returns:
            int: The total latency from the source to the destination in milliseconds.
        """
        # Initialize a latency array
        latency = 0
        # For each generation, calculate the latency from source to destination
        indices = [index for index, value in enumerate(
            source.forward_table) if value.destination == destination.ip_address]
        indices, link_indices = self.get_multi_links(source, indices)
        # Using hash function to determine which route to take
        chosen_route = source.select_route(source.ip_address,
                                           destination.ip_address, len(indices), flow_label)
        latency += self.links[link_indices[chosen_route]].delay

        # Find the next router
        index = next(index for index, value in enumerate(
            self.routers) if value.ip_address ==
            source.forward_table[indices[chosen_route]].next_hop)
        next_router = self.routers[index]

        # Continue finding the next hop and adding the delay until the destination is reached
        while next_router.ip_address != destination.ip_address:
            if latency/10**3 > 301:
                return latency
            indices = [index for index, value in enumerate(
                next_router.forward_table) if value.destination == destination.ip_address]
            indices, link_indices = self.get_multi_links(next_router, indices)

            chosen_route = next_router.select_route(source.ip_address,
                                                    destination.ip_address,
                                                    len(indices), flow_label)

            latency += self.links[link_indices[chosen_route]].delay

            index = next(index for index, value in enumerate(
                self.routers) if value.ip_address ==
                next_router.forward_table[indices[chosen_route]].next_hop)
            next_router = self.routers[index]

        destination.add_latency_to_bin(source.ip_address, latency/1000)
        return latency

    def send_probs(self, source: Router, destination: Router, flow_label: int) -> None:
        """
        Send multiple probes from the source router to the destination router
        over a specified duration.

        This method sends probes between the source and destination routers for each
        time step in the specified duration.
        It updates and reverts the hardware interface based on the start and end
        times of hardware issues.

        Args:
            source (Router): The source router from which the probes are sent.
            destination (Router): The destination router to which the probes are sent.
            flow_label (int): The flow label used to determine the route. If None, 
                              a random flow label is generated for each probe.

        Returns:
            None
        """
        fl = flow_label

        start_time = []
        end_time = []

        for _, row in self.hw_issue.iterrows():
            start_time.append(row['Start'])
            end_time.append(row['End'])
        # For each generation, calculate the latency from source to destination
        for t in range(self.generation_rate*self.duration):
            start_indices = [index for index,
                             value in enumerate(start_time) if value == t]
            end_indices = [index for index,
                           value in enumerate(end_time) if value == t]
            for s in start_indices:
                self.hw_update_interface(self.hw_issue.iloc[s])
            for e in end_indices:
                self.hw_revert_interface(self.hw_issue.iloc[e])
            if flow_label is None:
                fl = random.randint(0, 32)
            self.send_prob(source, destination, fl)

    def get_router_index_from_ip(self, ip_address: str) -> int:
        """
        Get the index of a router based on its IP address.

        This method iterates through the list of routers and returns
        the index of the router
        that matches the given IP address.

        Args:
            ip_address (str): The IP address of the router to find.

        Returns:
            int: The index of the router with the specified IP address.

        Raises:
            ValueError: If no router with the specified IP address is found.
        """
        for i, router in enumerate(self.routers):
            if router.ip_address == ip_address:
                return i
        raise ValueError(f"No router found with IP address: {ip_address}")

    def get_router_index_from_id(self, router_id: str) -> int:
        """
        Get the index of a router based on its ID.

        This method iterates through the list of routers and returns
        the index of the router that matches the given router ID.

        Args:
            router_id (str): The ID of the router to find.

        Returns:
            int: The index of the router with the specified ID.

        Raises:
            ValueError: If no router with the specified ID is found.
        """
        for i, router in enumerate(self.routers):
            if router.id == router_id:
                return i
        raise ValueError(f"No router found with ID: {router_id}")

    def get_router_by_name(self, name: str) -> Router:
        """
        Get a router based on its node name.

        This method iterates through the list of routers and returns the router
        that matches the given node name.

        Args:
            name (str): The node name of the router to find.

        Returns:
            Router: The router with the specified node name, or None if no such router is found.
        """
        for router in self.routers:
            if router.node_name == name:
                return router
        return None

    def get_next_hops(self, source: Router, destination: Router) -> List[str]:
        """
        Retrieves the next hop IP addresses from the source router to the destination router.

        This method iterates through the source router's forwarding table to find entries that
        match the destination router's IP address and collects the corresponding next hop
        IP addresses.

        Args:
            source (Router): The source router.
            destination (Router): The destination router.

        Returns:
            List[str]: A list of next hop IP addresses.
        """
        next_hops: List[str] = []
        for _, value in enumerate(source.forward_table):
            if value.destination == destination.ip_address:
                next_hops.append(value.next_hop)
        return next_hops

    def get_all_neighbors(self, source: Router) -> List[str]:
        """
        Retrieves all neighboring routers' IP addresses for the given source router.

        This method iterates through the network links to find links originating from the source
        router and collects the destination IP addresses of those links.

        Args:
            source (Router): The source router.

        Returns:
            List[str]: A list of neighboring routers' IP addresses.
        """
        neighbors: List[str] = []
        for link in self.links:
            if link.source == source.ip_address:
                neighbors.append(link.destination)
        return neighbors

    def ipm_session(self, source_ip: str, destination_ip: str, fl: int = None,
                    t: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) -> None:
        """
        Emulate an IPM (Inter-Process Messaging) session between two routers.

        This method emulates an IPM session by sending probes between the source and
        destination routers for a specified number of generations.
        It also exports sink and source data and updates the bins for the 
        destination router.

        Args:
            source_ip (str): The IP address of the source router.
            destination_ip (str): The IP address of the destination router.
            fl (int, optional): The flow label to be used for the probes. Defaults to None.
            t (str, optional): The timestamp for the session. Defaults to the current time.

        Returns:
            None
        """
        source = self.routers[self.get_router_index_from_ip(source_ip)]
        destination = self.routers[self.get_router_index_from_ip(
            destination_ip)]
        for gen_number in range(self.num_generation):
            self.send_probs(source, destination, fl)

            self.export_sink_data(destination, fl, gen_number, t)
            self.export_source_data(source, fl, gen_number, t)
            destination.update_bins()

    def get_multi_links(self, source: Router, indices: List[int]) -> Tuple[List[int], List[int]]:
        """
        Get multiple link indices for a given source router and a list of indices.

        This method finds the link indices with the minimum cost for each index in the provided
        list.
        It also handles cases where there are multiple links with the same minimum cost.

        Args:
            source (Router): The source router.
            indices (List[int]): A list of indices representing the forward table entries.

        Returns:
            Tuple[List[int], List[int]]: A tuple containing the updated list of indices
            and the corresponding multi-link indices.
        """
        new_indices = []
        multi_link_indices = [0 for _ in range(len(indices))]
        for i in indices:
            next_hop = source.forward_table[i].next_hop
            link_indices = [index for index, value in enumerate(
                self.links) if value.destination == next_hop and value.source == source.ip_address]
            min_cost = min([self.links[index].cost for index in link_indices])
            min_cost_count = sum(
                1 for index in link_indices if self.links[index].cost == min_cost)
            multi_link_indices[indices.index(i)] = link_indices[0]
            for j in range(min_cost_count-1):
                multi_link_indices.append(link_indices[j+1])
                new_indices.append(i)
        indices += new_indices
        return indices, multi_link_indices

    def ecmp_analysis(self, source_id: str, show: bool = False) -> Tuple[dict, dict]:
        """
        Perform ECMP (Equal-Cost Multi-Path) analysis for a given source router.

        This method analyzes the number of equal-cost paths from the source router to all 
        other routers in the network. It also computes and optionally displays the distribution
        and CDF (Cumulative Distribution Function) of the number of paths. Additionally, 
        it performs latency tests and calculates the average latency per hop count.

        Args:
            source_id (str): The ID of the source router.
            show (bool, optional): Whether to display the analysis plots. Defaults to False.

        Returns:
            Tuple[dict, dict]: A tuple containing:
                - ecmp_dict: A dictionary where keys are the number of paths and values are
                the number of destinations with that many paths.
                - hop_latency: A dictionary where keys are the number of hops and values are
                the average latency for that hop count.
        """
        source = self.routers[self.get_router_index_from_id(source_id)]
        ecmp_dict = {}
        for destination in self.routers:
            if destination != source:
                indices = [index for index, value in enumerate(
                    source.forward_table) if value.destination == destination.ip_address]
                # Add a duplicate indices if there's two links to the same next_hop"
                indices, _ = self.get_multi_links(source, indices)
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

            # Sort ecmp_dict by number of paths
        sorted_counts = sorted(ecmp_dict.items())
        num_destinations = sum(ecmp_dict.values())

        # Compute CDF
        cdf_values = []
        cumulative_sum = 0
        for count, freq in sorted_counts:
            cumulative_sum += freq
            cdf_values.append((count, cumulative_sum / num_destinations))

        # Extract x and y values for plotting
        x_cdf = [item[0] for item in cdf_values]
        y_cdf = [item[1] for item in cdf_values]

        # Plot CDF
        if show:
            plt.step(x_cdf, y_cdf, where='post')
            plt.xlim(0, max(x_cdf) + max(x_cdf) / 10)
            plt.xlabel('Number of paths')
            plt.ylabel('CDF')
            plt.title('ECMP CDF analysis for router ' + source_id)
            plt.grid(True)
            plt.show()

        path = []

        for router in self.routers:
            if router != source:
                new_path, _ = self.latency_test(source.id, router.id)
                path.append(new_path)

        number_of_hop_count = {}
        latency_hop_count = {}

        for item in path:
            for path, latency in item:
                if len(path) in number_of_hop_count:
                    number_of_hop_count[len(path)] += 1
                    latency_hop_count[len(path)] += latency
                else:
                    number_of_hop_count[len(path)] = 1
                    latency_hop_count[len(path)] = latency

        hop_latency = {}
        for key, value in number_of_hop_count.items():
            hop_latency[key] = latency_hop_count[key]/(value * 1000)

        return ecmp_dict, dict(sorted(hop_latency.items()))

    def get_number_of_paths(self, source: Router, destination: Router, index: int) -> int:
        """
        Get the number of paths from the source router to the destination router.

        This method recursively calculates the number of paths from the source router to the 
        destination router by traversing the forward tables and considering multiple links with
        the same minimum cost.

        Args:
            source (Router): The source router.
            destination (Router): The destination router.
            index (int): The index in the source router's forward table.

        Returns:
            int: The number of paths from the source to the destination router.
        """
        next_hop = source.forward_table[index].next_hop
        next_router = self.routers[self.get_router_index_from_ip(next_hop)]
        indices = [index for index, value in enumerate(
            next_router.forward_table) if value.destination == destination.ip_address]
        indices, _ = self.get_multi_links(next_router, indices)

        counter = 0
        if next_router.ip_address == destination.ip_address:
            return 1
        for i in indices:
            counter += self.get_number_of_paths(next_router, destination, i)
        return counter

    def all_latency_test(self) -> List[Tuple[list, int]]:
        """
        Perform latency tests between all pairs of routers and collect wrong paths.

        This method iterates over all pairs of routers in the network and performs a latency test
        between each pair. It collects and returns all the wrong paths identified during the tests.

        Returns:
            List[Tuple[List, int]]: A list of tuples where each tuple contains a wrong path
            (as a list of routers) and the corresponding latency (as an integer).
        """
        wrong_paths = []
        for source in self.routers:
            for destination in self.routers:
                if source != destination:
                    _, wrong_path = self.latency_test(
                        source.id, destination.id)
                    wrong_paths.extend(wrong_path)
        return wrong_paths

    def latency_test(
        self,
        source_id: str,
        destination_id: str
    ) -> Tuple[zip, List[Tuple[List, int]]]:
        """
        Perform a latency test between a source router and a destination router.

        This method calculates the latency for all possible paths between the source and destination
        routers.
        It identifies and returns the paths with their latencies and collects paths that exceed a
        certain latency threshold.

        Args:
            source_id (str): The ID of the source router.
            destination_id (str): The ID of the destination router.

        Returns:
            Tuple[zip, List[Tuple[List, int]]]: A tuple containing:
                - A zip object of paths and their corresponding latencies.
                - A list of tuples where each tuple contains a wrong path (as a list of routers)
                and the corresponding latency (as an integer).
        """
        latencies = []
        paths = []
        wrong_paths = []

        source: Router = self.routers[self.get_router_index_from_id(source_id)]
        destination: Router = self.routers[self.get_router_index_from_id(
            destination_id)]

        indices = [index for index, value in enumerate(
            source.forward_table) if value.destination == destination.ip_address]
        indices, link_indices = self.get_multi_links(source, indices)

        for i, index in enumerate(indices):
            path_groups, _ = self.get_latency_and_path(source, destination, index, [
                source.ip_address], link_indices[i], 0)
            for path, latency in path_groups:
                paths.append(path)
                latencies.append(latency)

        min_latency = min(latencies)

        for i, latency in enumerate(latencies):
            if latency >= min_latency + self.threshold * 1000:
                wrong_paths.append((paths[i], latency - min_latency))

        return zip(paths, latencies), wrong_paths

    def get_latency_and_path(self, current: Router, destination: Router, index: int, path: list,
                             link_index: int, latency: int) -> Tuple[List[Tuple[list, int]], int]:
        """
        Calculate the latency and path from the current router to the destination router.

        This method recursively calculates the latency and path from the current router to the
        destination router by traversing the forward tables and considering multiple links with
        the same destination.

        Args:
            current (Router): The current router.
            destination (Router): The destination router.
            index (int): The index in the current router's forward table.
            path (list): The current path taken.
            link_index (int): The index of the link being used.
            latency (int): The current accumulated latency.

        Returns:
            Tuple[List[Tuple[list, int]], int]: A tuple containing:
                - A list of tuples where each tuple contains a path (as a list of routers) and the
                corresponding latency (as an integer).
                - An integer (always 0 in this implementation).
        """
        next_hop = current.forward_table[index].next_hop
        next_router: Router = self.routers[self.get_router_index_from_ip(
            next_hop)]
        path.append(next_router.ip_address)
        latency += self.links[link_index].delay

        if next_router.ip_address == destination.ip_address:
            return [(path, latency)], 0

        all_paths = []
        indices = [i for i, value in enumerate(
            next_router.forward_table) if value.destination == destination.ip_address]
        indices, next_multi_links = self.get_multi_links(next_router, indices)

        for i, idx in enumerate(indices):
            new_path_groups, _ = self.get_latency_and_path(
                next_router, destination, idx, path[:], next_multi_links[i], latency)
            all_paths.extend(new_path_groups)

        return all_paths, 0

    def hw_issue_detection(self, source_dataframe: pd.DataFrame, sink_dataframe: pd.DataFrame,
                           latency: bool = False, loss: bool = False,
                           bin_detection: bool = False) -> List[str]:
        """
        Detect hardware issues based on latency or loss in the network.

        This method analyzes the provided dataframes to detect hardware issues by checking for
        latency or loss.
        It filters the sink dataframe based on session ID and histogram type, and then checks for
        latency issues by comparing the 'Max' and 'Min' values. If a latency issue is detected,
        it identifies the source and destination routers and performs hardware detection on the
        path between them.

        Args:
            source_dataframe (pd.DataFrame): The dataframe containing source data.
            sink_dataframe (pd.DataFrame): The dataframe containing sink data.
            latency (bool): Flag to indicate if latency issues should be checked.
            loss (bool): Flag to indicate if loss issues should be checked.

        Returns:
            List[str]: A list of router IDs where hardware issues were detected.
        """

        gen_mark = sink_dataframe.iloc[-1]['Histogram type']
        mark = 'Black' if gen_mark == 'White' else 'White'
        session_id = sink_dataframe.iloc[-1]['Session ID']

        # Use boolean indexing to filter rows
        filtered_sink: pd.DataFrame = sink_dataframe[(sink_dataframe['Session ID'] == session_id) &
                                                     (sink_dataframe['Histogram type'] == mark)]
        last_index = filtered_sink.index[-1]

        if filtered_sink.empty:
            filtered_sink = sink_dataframe[sink_dataframe['Session ID'] == session_id]
        else:
            filtered_sink = sink_dataframe.loc[last_index+1:]

        rest_of_sink: pd.DataFrame = sink_dataframe[:last_index]
        opposite_mark = 'Black' if mark == 'White' else 'White'
        rest_of_sink = rest_of_sink[(rest_of_sink['Session ID'] == session_id) &
                                             (rest_of_sink['Histogram type'] == opposite_mark)]
        previous_index = rest_of_sink.index[-1]
        rest_of_sink = sink_dataframe.loc[previous_index+1:last_index]
        filtered_source: pd.DataFrame = source_dataframe[(source_dataframe['Session ID']
                                                          == session_id) &
                                                         (source_dataframe['Packet Counter Type']
                                                          == mark)]

        number_of_packet = filtered_source.iloc[-1]['Packet Counter']

        if filtered_source.empty:
            filtered_source = source_dataframe[source_dataframe['Session ID'] == session_id]
        else:
            # Use the last index from the filtered data
            last_index = filtered_source.index[-1]
            filtered_source = source_dataframe.loc[last_index:]

        routers_id: List[str] = []

        if latency:
            for _, row in filtered_sink.iterrows():
                if (row['Max'] - row['Min']) >= self.threshold:
                    zipped_list, _ = self.latency_test(
                        source_id=row['Source'], destination_id=row['Destination'])
                    default_latency = [x/10**3 for _, x in zipped_list]
                    if 1.05 > max(default_latency)/row['Max'] > 0.95:
                        continue
                    source_router = self.routers[self.get_router_index_from_ip(
                        row['Source'])]
                    destination_router = self.routers[
                        self.get_router_index_from_ip(row['Destination'])]
                    found_routers = self.latency_hw_detection(source=source_router,
                                                              destination=destination_router,
                                                              routers_id=routers_id,
                                                              sink_dataframe=filtered_sink)
                    for router in found_routers:
                        if router not in routers_id:
                            routers_id.append(router)
                    if len(found_routers) == 0 and source_router.id not in routers_id:
                        routers_id.append(source_router.id)
        if loss:
            for _, row in filtered_sink.iterrows():
                source_ipaddress = row['Source']
                destination_ipaddress = row['Destination']
                source_row: pd.Series = filtered_source[(filtered_source['Source']
                                                         == source_ipaddress) &
                                                        (filtered_source['Destination']
                                                         == destination_ipaddress)]
                source_row = source_row.iloc[0]
                histogram_values: List[str] = row['Histogram Value']
                int_histogram_values = ast.literal_eval(histogram_values)
                sink_packet_counter = sum(int_histogram_values)
                source_packet_counter: int = source_row['Packet Counter']
                if sink_packet_counter != source_packet_counter:
                    source_router = self.routers[
                        self.get_router_index_from_ip(source_ipaddress)
                    ]
                    destination_router = self.routers[
                        self.get_router_index_from_ip(destination_ipaddress)
                    ]
                    found_routers = self.loss_hw_detection(source=source_router,
                                                           destination=destination_router,
                                                           routers_id=routers_id,
                                                           sink_dataframe=filtered_sink,
                                                           source_dataframe=filtered_source)
                    for router in found_routers:
                        if router not in routers_id:
                            routers_id.append(router)
                    if len(found_routers) == 0 and source_router.id not in routers_id:
                        routers_id.append(source_router.id)

        if bin_detection:
            if rest_of_sink.empty:
                print("No data to analyze")
                return routers_id
            for _, row in filtered_sink.iterrows():
                source_ipaddress = row['Source']
                destination_ipaddress = row['Destination']
                source_router = self.routers[self.get_router_index_from_ip(source_ipaddress)]
                destination_router = self.routers[
                    self.get_router_index_from_ip(destination_ipaddress)
                    ]

                previous_row: pd.Series = rest_of_sink[(rest_of_sink['Source']
                                                         == source_ipaddress) &
                                                        (rest_of_sink['Destination']
                                                         == destination_ipaddress)]
                previous_row = previous_row.iloc[0]

                histogram_values: List[str] = row['Histogram Value']
                int_histogram_values = ast.literal_eval(histogram_values)

                previous_histogram_values: List[str] = previous_row['Histogram Value']
                int_previous_histogram_values = ast.literal_eval(previous_histogram_values)

                indices = [index for index, value in enumerate(
                source_router.forward_table) if value.destination == destination_ipaddress]
                indices, _ = self.get_multi_links(source_router, indices)
                total_number_of_path = 0
                for i in indices:
                    total_number_of_path += self.get_number_of_paths(source_router,
                                                                     destination_router,
                                                                     index=i)
                for index, value in enumerate(int_histogram_values):
                    if value == 0 and int_previous_histogram_values[index] == 0:
                        continue
                    elif ((value == 0 and int_previous_histogram_values[index] > 0)
                        or (value > 0 and int_previous_histogram_values[index] == 0)):
                        self.bin_distribution_hw_detection(source=source_router,
                                                           destination=destination_router,
                                                           routers_id=routers_id,
                                                           current_sink_dataframe=filtered_sink,
                                                           previous_sink_dataframe=rest_of_sink)
                        break
                    elif not (1 + ( number_of_packet/total_number_of_path) >
                              (int_previous_histogram_values[index] /value) >
                                1 - (number_of_packet/total_number_of_path)):
                        self.bin_distribution_hw_detection(source=source_router,
                                                           destination=destination_router,
                                                           routers_id=routers_id,
                                                           current_sink_dataframe=filtered_sink,
                                                           previous_sink_dataframe=rest_of_sink)
                        break

        return routers_id

    def latency_hw_detection(self, source: Router, destination: Router,
                             routers_id: List[str], sink_dataframe: pd.DataFrame) -> List[str]:
        """
        Detect hardware issues along the path from the source router to the destination router.

        This method recursively traverses the path from the source router to the destination
        router, checking for hardware issues based on the latency threshold. If an issue is
        detected, the source router's ID is added to the list of routers with detected issues.

        Args:
            source (Router): The source router.
            destination (Router): The destination router.
            routers_id (List[str]): A list to store the IDs of routers with detected hardware
            issues.
            sink_dataframe (pd.DataFrame): The dataframe containing sink data.
        """
        indices = [index for index, value in enumerate(
            source.forward_table) if value.destination == destination.ip_address]
        issue_index: List[int] = []
        for index in indices:
            next_hop = source.forward_table[index].next_hop
            next_router = self.routers[self.get_router_index_from_ip(next_hop)]
            if next_router.ip_address == destination.ip_address:
                continue
            if self.check_next_router_latency_issue(next_router, destination, sink_dataframe):
                issue_index.append(index)
        if len(issue_index) == 0:
            return [source.id]

        for index in issue_index:
            next_hop = source.forward_table[index].next_hop
            next_router = self.routers[self.get_router_index_from_ip(next_hop)]
            found_routers = self.latency_hw_detection(next_router, destination,
                                                      routers_id, sink_dataframe)
            for router in found_routers:
                if router not in routers_id:
                    routers_id.append(router)

        return routers_id

    def check_next_router_latency_issue(self, next_router: Router,
                                        destination: Router,
                                        sink_dataframe: pd.DataFrame) -> bool:
        """
        Check for hardware issues in the next router along the path.

        This method checks for hardware issues in the next router along the path from the source
        router to the destination router. If an issue is detected, the source router's ID is added
        to the list of routers with detected issues.

        Args:
            next_router (Router): The next router along the path.
            destination (Router): The destination router.
            sink_dataframe (pd.DataFrame): The dataframe containing sink data.
            source (Router): The source router.
        """
        if next_router.ip_address == destination.ip_address:
            return False

        next_row = sink_dataframe[
            (sink_dataframe['Source'] == next_router.ip_address) &
            (sink_dataframe['Destination'] == destination.ip_address)
        ]
        if next_row.empty:
            return

        next_row = next_row.iloc[0]
        if (next_row['Max'] - next_row['Min']) >= self.threshold:
            zipped_list, _ = self.latency_test(
                source_id=next_router.id, destination_id=destination.id)
            default_latency = [x / 10**3 for _, x in zipped_list]

            if not 1.05 > max(default_latency) / next_row['Max'] > 0.95:
                return True
        return False

    def loss_hw_detection(self, source: Router, destination: Router,
                          routers_id: List[str],
                          sink_dataframe: pd.DataFrame,
                          source_dataframe: pd.DataFrame) -> list[str]:
        """
        Detect hardware issues along the path from the source router to the destination router.

        This method recursively traverses the path from the source router to the destination
        router, checking for hardware issues based on the loss threshold. If an issue is
        detected, the source router's ID is added to the list of routers with detected issues.

        Args:
            source (Router): The source router.
            destination (Router): The destination router.
            routers_id (List[str]): A list to store the IDs of routers with detected hardware
            issues.
            sink_dataframe (pd.DataFrame): The dataframe containing sink data.
        """
        indices = [index for index, value in enumerate(
            source.forward_table) if value.destination == destination.ip_address]
        issue_index: List[int] = []
        for index in indices:
            next_hop = source.forward_table[index].next_hop
            next_router = self.routers[self.get_router_index_from_ip(next_hop)]
            if next_router.ip_address == destination.ip_address:
                continue
            if self.check_next_router_loss_issue(next_router,
                                                 destination,
                                                 sink_dataframe,
                                                 source_dataframe):
                issue_index.append(index)
        if len(issue_index) == 0:
            return [source.id]

        for index in issue_index:
            next_hop = source.forward_table[index].next_hop
            next_router = self.routers[self.get_router_index_from_ip(next_hop)]
            found_routers = self.loss_hw_detection(next_router, destination,
                                                   routers_id, sink_dataframe, source_dataframe)
            for router in found_routers:
                if router not in routers_id:
                    routers_id.append(router)

        return routers_id

    def check_next_router_loss_issue(self, next_router: Router, destination: Router,
                                     sink_dataframe: pd.DataFrame,
                                     source_dataframe: pd.DataFrame) -> bool:
        """
        Check for hardware issues in the next router along the path.

        This method checks for hardware issues in the next router along the path from the source
        router to the destination router. If an issue is detected, the source router's ID is added
        to the list of routers with detected issues.

        Args:
            next_router (Router): The next router along the path.
            destination (Router): The destination router.
            routers_id (list[str]): A list to store the IDs of routers with detected
            hardware issues.
            sink_dataframe (pd.DataFrame): The dataframe containing sink data.
            source_dataframe (pd.DataFrame): The dataframe containing source data.

        Returns:
            bool: True if a hardware issue is detected, False otherwise.
        """
        if next_router.ip_address == destination.ip_address:
            return False

        next_row = sink_dataframe[
            (sink_dataframe['Source'] == next_router.ip_address) &
            (sink_dataframe['Destination'] == destination.ip_address)
        ]
        if next_row.empty:
            return False

        next_row = next_row.iloc[0]
        source_ipaddress = next_row['Source']
        destination_ipaddress = next_row['Destination']
        source_row: pd.Series = source_dataframe[
            (source_dataframe['Source'] == source_ipaddress) &
            (source_dataframe['Destination'] == destination_ipaddress)
        ]
        source_row = source_row.iloc[0]

        histogram_values: List[str] = next_row['Histogram Value']
        int_histogram_values = ast.literal_eval(histogram_values)
        sink_packet_counter = sum(int_histogram_values)

        source_packet_counter = source_row['Packet Counter']

        if sink_packet_counter != source_packet_counter:
            return True
        return False

    def bin_distribution_hw_detection(self, source: Router, destination: Router,
                                      previous_sink_dataframe: pd.DataFrame,
                                      current_sink_dataframe: pd.DataFrame,
                                      routers_id: List[str]) -> List[str]:

        """
        Detect hardware issues along the path from the source router to the destination router.

        This method recursively traverses the path from the source router to the destination
        router, checking for hardware issues based on the bin distribution. If an issue is
        detected, the source router's ID is added to the list of routers with detected issues.

        Args:
            source (Router): The source router.
            destination (Router): The destination router.
            routers_id (List[str]): A list to store the IDs of routers with detected hardware
            issues.
            sink_dataframe (pd.DataFrame): The dataframe containing sink data.
        """
        indices = [index for index, value in enumerate(
            source.forward_table) if value.destination == destination.ip_address]
        issue_index: List[int] = []
        for index in indices:
            next_hop = source.forward_table[index].next_hop
            next_router = self.routers[self.get_router_index_from_ip(next_hop)]
            if next_router.ip_address == destination.ip_address:
                continue
            if self.check_next_router_bin_issue(next_router,
                                                 destination,
                                                 previous_sink_dataframe,
                                                 current_sink_dataframe):
                issue_index.append(index)
        if len(issue_index) == 0:
            return [source.id]

        for index in issue_index:
            next_hop = source.forward_table[index].next_hop
            next_router = self.routers[self.get_router_index_from_ip(next_hop)]
            found_routers = self.bin_distribution_hw_detection(next_router, destination,
                                                   previous_sink_dataframe, current_sink_dataframe,
                                                   routers_id)
            for router in found_routers:
                if router not in routers_id:
                    routers_id.append(router)

        return routers_id

    def check_next_router_bin_issue(self, next_router: Router, destination: Router,
                                    previous_sink_dataframe: pd.DataFrame,
                                    current_sink_dataframe: pd.DataFrame) -> bool:
        """
        Checks for bin issues in the next router's histogram values.

        This method compares the histogram values of the next router for a given destination
        between two dataframes (previous and current) to determine if there are any bin issues.
        A bin issue is identified if the current histogram values deviate beyond a certain
        threshold from the previous histogram values, adjusted by the number of paths.

        Args:
            next_router (Router): The next router in the path.
            destination (Router): The destination router.
            previous_sink_dataframe (pd.DataFrame): The dataframe containing previous sink data.
            current_sink_dataframe (pd.DataFrame): The dataframe containing current sink data.

        Returns:
            bool: True if a bin issue is detected, False otherwise.
        """

        previous_row = previous_sink_dataframe[
            (previous_sink_dataframe['Source'] == next_router.ip_address) &
            (previous_sink_dataframe['Destination'] == destination.ip_address)
        ]
        previous_row = previous_row.iloc[0]

        current_row = current_sink_dataframe[
            (current_sink_dataframe['Source'] == next_router.ip_address) &
            (current_sink_dataframe['Destination'] == destination.ip_address)
        ]
        current_row = current_row.iloc[0]

        histogram_values: List[str] = current_row['Histogram Value']
        int_histogram_values = ast.literal_eval(histogram_values)

        previous_histogram_values: List[str] = previous_row['Histogram Value']
        int_previous_histogram_values = ast.literal_eval(previous_histogram_values)

        indices = [index for index, value in enumerate(
                next_router.forward_table) if value.destination == destination.ip_address]
        indices, _ = self.get_multi_links(next_router, indices)
        total_number_of_path = 0
        for i in indices:
            total_number_of_path += self.get_number_of_paths(next_router, destination, index=i)

        number_of_packet = sum(int_histogram_values)
        for index, value in enumerate(int_histogram_values):
            if value == 0 and int_previous_histogram_values[index] == 0:
                continue
            elif ((value == 0 and int_previous_histogram_values[index] > 0) or
                  (value > 0 and int_previous_histogram_values[index] == 0)):
                return True
            elif not (1 + ( number_of_packet/total_number_of_path) >
                      (int_previous_histogram_values[index] / value) >
                    1 - number_of_packet/total_number_of_path):
                return True
        return False
