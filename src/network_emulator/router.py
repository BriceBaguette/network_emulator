"""
This module defines the Router class, which represents a router in a network 
and the ForwardTableElement class, which represents an element in the forwarding 
table of a router.
"""
import hashlib
from typing import List, Tuple, Dict

MAX_BINS = 30
EMPTY = 0
NEIGHBOR = 1
ROUTER = 2

class ForwardTableElement:
    """
    Represents an element in the forwarding table of a router.
    """

    def __init__(self, dest: str, next_hop:str):
        """
        Initializes a ForwardTableElement object.

        Parameters:
        - dest (str): IP address of the destination router.
        - next_hop (str): IP address of the next hop router.
        """
        self.destination = dest
        self.next_hop = next_hop

    def to_json(self):
        """
        Converts the ForwardTableElement object to a JSON representation.

        Returns:
        - dict: JSON representation of the ForwardTableElement object.
        """
        return {
            "destination": self.destination,
            "next_hop": self.next_hop
        }
        
class Bin: 
    def __init__(self, start: int, end:int):
        self.start = start
        self.end = end
        
        self.value = 0
        
    def reset(self):
        self.value = 0
        
    def increment(self):
        self.value += 1

class Router:
    """
    Represents a router in a network.
    """

    def __init__(self, router_id: str, node_name: str, active :int, ip_address: str):
        """
        Initializes a Router object.

        Parameters:
        - node_name (str): Name of the router.
        - active (bool): Status of the router.
        - ip_address (str): IP address of the router.
        """
        self.node_name = node_name
        self.ip_address = ip_address
        self.active = active
        self.id = router_id
        self.forward_table: List[ForwardTableElement] = []
        self.__bins: dict[str, List[Bin]] = {}
        self.__min_max_bins: dict[str, Tuple[int, int]] = {}
        
    def init_bins(self, source: str):
        self.__bins[source] = []
        for i in range(MAX_BINS):
            start = i * 10
            end = (i + 1) * 10
            self.__bins[source].append(Bin(start, end))
            
        self.__min_max_bins[source] = (1000, 0)
            
    def add_latency_to_bin(self, source: str, latency: int):
        if source not in self.__bins:
            self.init_bins(source)
            
        for bin in self.__bins[source]:
            if bin.start <= latency < bin.end:
                bin.increment()
                break
            
        if latency < self.__min_max_bins[source][0]:
            self.__min_max_bins[source] = (latency, self.__min_max_bins[source][1])
            
        if latency > self.__min_max_bins[source][1]:
            self.__min_max_bins[source] = (self.__min_max_bins[source][0], latency)
            
    def get_bins(self) -> Tuple[Dict[str, List[Bin]], Dict[str, Tuple[int, int]]]:
        return self.__bins, self.__min_max_bins
    
    def update_bins(self):
        for key, bins in self.__bins.items():
            # Step 1: Mark the bins
            marks = self.mark_bins(bins)
           
            # Step 2: Merge empty bins
            merged_bins, freed_bins = self.merge_empty_bins(bins, marks)
            
            # Step 3: Subdivide used and neighbor bins
            subdidvided_bins = self.subdivide_bins(bins, marks, freed_bins)
           
            # Step 4: subdivide empty bins with remaining bins
            empty_bins = self.subdivide_empty_bins(merged_bins, len(subdidvided_bins) + len(merged_bins))
            
            new_bins: List[Bin] = []
            
            empty_bin_index = 0
            subdidvided_bins_index = 0
            
            for i in range(MAX_BINS):
                if subdidvided_bins_index == len(subdidvided_bins) and empty_bin_index == len(empty_bins):
                    break
                
                elif subdidvided_bins_index == len(subdidvided_bins) or empty_bins[empty_bin_index].start < subdidvided_bins[subdidvided_bins_index].start and empty_bin_index < len(empty_bins):
                    new_bins.append(empty_bins[empty_bin_index])
                    empty_bin_index += 1
                else:
                    new_bins.append(subdidvided_bins[subdidvided_bins_index])
                    subdidvided_bins_index += 1

            for bin in new_bins:
                bin.reset()
            # Update the bins
            self.__bins[key] = new_bins

    def mark_bins(self, bins: List[Bin]) -> List[int]:
        marks = []
        for i, bin in enumerate(bins):
            if bin.value > 0:
                marks.append(ROUTER)
            elif (i > 0 and bins[i - 1].value > 0) or (i < len(bins) - 1 and bins[i + 1].value > 0):
                marks.append(NEIGHBOR)
            else:
                marks.append(EMPTY)
        return marks

    def merge_empty_bins(self, bins: List[Bin], marks: List[int]) -> Tuple[List[Bin], int]:
        merged_bins = []
        freed_bins = 0
        i = 0
        while i < len(bins):
            if marks[i] == EMPTY:
                start = bins[i].start
                j = 0
                while i < len(bins) and marks[i] == EMPTY:
                    j += 1
                    i += 1
                end = bins[i - 1].end
                merged_bins.append(Bin(start, end))
                freed_bins += j - 1
            else:
                i += 1
        return merged_bins, freed_bins

    def subdivide_bins(self, bins: List[Bin], marks: List[int], freed_bins: int) -> List[Bin]:
        used_and_neighbor_bins = [bin for i, bin in enumerate(bins) if marks[i] in (ROUTER, NEIGHBOR)]
        new_bins = []

        # Determine the total number of new bins to create
        total_new_bins = len(used_and_neighbor_bins) + freed_bins
        num_new_bins = max(1, total_new_bins // len(used_and_neighbor_bins))
        for index, bin in enumerate(used_and_neighbor_bins):
            num_new_bins = max(1, total_new_bins // (len(used_and_neighbor_bins)- index))
            bin_size = (bin.end - bin.start) // num_new_bins
            if bin_size < 1:
                bin_size = 1
                num_new_bins = bin.end - bin.start

            start = bin.start
            for _ in range(num_new_bins - 1):
                total_new_bins -= 1
                end = start + bin_size
                new_bins.append(Bin(start, end))
                start = end

            # Ensure the last bin ends exactly at bin.end
            new_bins.append(Bin(start, bin.end))

        return new_bins

    
    def subdivide_empty_bins(self, empty_bins: List[Bin], occupied_bins: int) -> List[Bin]:
        remaining_bins = MAX_BINS - occupied_bins
        max_empty_bins = max(empty_bins, key=lambda bin: bin.end - bin.start)
        new_empty_bins = []
        if remaining_bins > 0 and empty_bins:
            for index, bin in enumerate(empty_bins):
                num_new_bins = max(1, remaining_bins // (len(empty_bins)-index))
                if bin == max_empty_bins:
                    num_new_bins +=1
                bin_size = (bin.end - bin.start) // num_new_bins
                if bin_size < 1:
                    bin_size = 1
                    num_new_bins = bin.end - bin.start
                start = bin.start
                for _ in range(num_new_bins - 1):
                    remaining_bins -= 1
                    end = start + bin_size
                    new_empty_bins.append(Bin(start, end))
                    start = end
                # Ensure the last bin ends exactly at bin.end
                new_empty_bins.append(Bin(start, bin.end))
                
        else:
            new_empty_bins = empty_bins
        return new_empty_bins
                

    def has_entry_for_destination(self, dest_ip: str) -> bool:
        """
        Checks if the router has an entry for the given destination IP address.

        Parameters:
        - dest_ip (str): Destination IP address to check.

        Returns:
        - bool: True if the router has an entry for the destination IP address, False otherwise.
        """
        for entry in self.forward_table:
            if entry.destination == dest_ip:
                return True
        return False

    def update_forward_table(self, element: ForwardTableElement) -> None: 
        """
        Updates the forward table of the router with the given element.

        Parameters:
        - element (ForwardTableElement): Element to add to the forward table.
        """
        for entry in self.forward_table:
            if entry.destination == element.destination and entry.next_hop == element.next_hop:
                return
        self.forward_table.append(element)

    def ecmp_hash(self,source_ip: str, dest_ip: str, flow_label:int):
        """
        Performs ECMP hashing to select a route based on the destination IP address, packet number, 
        and distribution key.
        
        Parameters:
        - source_ip (str): Source IP address.
        - dest_ip (str): Destination IP address.
        - flow_label (int): Flow label.

        Returns:
        - int: Hash value used for route selection.
        """
       
        hash_input = f"{self.id}{source_ip}{dest_ip}{str(flow_label)}".encode('utf-8')

        hash_value = hashlib.sha256(hash_input).hexdigest()
        hash_int = int(hash_value, 16)

        return hash_int

    def select_route(self,source_ip: str, dest_ip:str , num_paths: int, flow_label: int):
        """
        Selects a route index based on the destination IP address, number of paths, 
        and packet number.

        Parameters:
        - dest_ip (str): Destination IP address.
        - num_paths (int): Number of available paths.
        - flow_label (int): flow_label.

        Returns:
        - int: Route index.
        """
        route_index = self.ecmp_hash(source_ip=source_ip, dest_ip=dest_ip, flow_label=flow_label) % num_paths
        return route_index

    def to_json(self):
        """
        Converts the Router object to a JSON representation.

        Returns:
        - dict: JSON representation of the Router object.
        """
        return {
            "id": self.id,
            "node_name": self.node_name,
            "ip_address": self.ip_address,
            "active": self.active,
            "forward_table": [entry.to_json() for entry in self.forward_table]
        }
