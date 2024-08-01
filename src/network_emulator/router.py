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
    """
    A class to represent a bin with a start and end range, and a value that can be incremented or
    reset.

    Attributes:
        start (int): The start of the bin range.
        end (int): The end of the bin range.
        value (int): The current value of the bin.
    """

    def __init__(self, start: int, end: int):
        """
        Initialize the Bin with a start and end range, and set the initial value to 0.

        Args:
            start (int): The start of the bin range.
            end (int): The end of the bin range.
        """
        self.start = start
        self.end = end
        self.value = 0

    def reset(self):
        """
        Reset the value of the bin to 0.
        """
        self.value = 0

    def increment(self):
        """
        Increment the value of the bin by 1.
        """
        self.value += 1

class Router:
    """
    Represents a router in a network.
    
    A router has differents attributes: 
        - An ip address
        - A name
        - An active variable telling if the router is on or off
        - An id that his unique for each router
        - A forward table to know where to forward packets
        - Some bins for each possible source to classify latency
        of packet dynamically
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
        """
        Initialize bins for a given source.

        This method creates a list of Bin objects for the specified source, 
        each with a defined start and end range.
        It also initializes the minimum and maximum bin values for the source.

        Args:
            source (str): The source for which bins are being initialized.
        """
        self.__bins[source] = []
        for i in range(MAX_BINS):
            start = i * 10
            end = (i + 1) * 10
            self.__bins[source].append(Bin(start, end))

        self.__min_max_bins[source] = (1000, 0)

    def add_latency_to_bin(self, source: str, latency: int):
        """
        Add latency to the appropriate bin for a given source.

        This method checks if the source has bins initialized. If not, it initializes them.
        It then increments the value of the bin that the latency falls into and updates the
        minimum and maximum latency values for the source.

        Args:
            source (str): The source for which latency is being added.
            latency (int): The latency value to be added to the appropriate bin.
        """
        if source not in self.__bins:
            self.init_bins(source)

        for source_bin in self.__bins[source]:
            if source_bin.start <= latency < source_bin.end:
                source_bin.increment()
                break

        if latency < 0:
            return

        if latency < self.__min_max_bins[source][0]:
            self.__min_max_bins[source] = (latency, self.__min_max_bins[source][1])

        if latency > self.__min_max_bins[source][1]:
            self.__min_max_bins[source] = (self.__min_max_bins[source][0], latency)

    def get_bins(self) -> Tuple[Dict[str, List[Bin]], Dict[str, Tuple[int, int]]]:
        """
        Get the bins and their min/max latency values.

        This method returns the bins and their corresponding minimum and maximum latency values for
        each source.

        Returns:
            Tuple[Dict[str, List[Bin]], Dict[str, Tuple[int, int]]]: A tuple containing
            two dictionaries:
                - The first dictionary maps each source to a list of Bin objects.
                - The second dictionary maps each source to a tuple containing the minimum and
                maximum latency values.
        """
        return self.__bins, self.__min_max_bins

    def update_bins(self, treshold:int = 5):
        """
        Update the bins for each source.

        This method performs the following steps for each source:
        1. Marks the bins.
        2. Merges empty bins.
        3. Subdivides used and neighbor bins.
        4. Subdivides empty bins with remaining bins.

        It then resets the new bins and updates the bins for the source.

        """
        for key, bins in self.__bins.items():
            # Step 1: Mark the bins
            marks = self.mark_bins(bins)

            no_update = True

            for index, old_bin in enumerate(bins):
                if marks[index] == ROUTER:
                    if old_bin.end - old_bin.start >= treshold:
                        no_update = False
                        break

            if no_update:
                for _, old_bin in enumerate(bins):
                    old_bin.reset()
                self.__bins[key] = bins
                continue

            # Step 2: Merge empty bins
            merged_bins, freed_bins = self.merge_empty_bins(bins, marks)

            # Step 3: Subdivide used and neighbor bins
            subdivided_bins, neighbors_bins = self.subdivide_bins(bins, marks, freed_bins)

            # Step 4: Subdivide empty bins with remaining bins
            empty_bins = self.subdivide_empty_bins(merged_bins,
                                                len(subdivided_bins) +
                                                len(merged_bins) +
                                                len(neighbors_bins))

            new_bins: List[Bin] = []
            empty_bin_index = 0
            subdivided_bins_index = 0
            neighbors_bins_index = 0

            while len(new_bins) < MAX_BINS:

                if (empty_bin_index < len(empty_bins) and
                    (subdivided_bins_index >= len(subdivided_bins) or
                    empty_bins[empty_bin_index].start <=
                    subdivided_bins[subdivided_bins_index].start) and
                    (neighbors_bins_index >= len(neighbors_bins) or
                    empty_bins[empty_bin_index].start <=
                    neighbors_bins[neighbors_bins_index].start)):

                    new_bins.append(empty_bins[empty_bin_index])
                    empty_bin_index += 1

                elif (subdivided_bins_index < len(subdivided_bins) and
                    (neighbors_bins_index >= len(neighbors_bins) or
                    subdivided_bins[subdivided_bins_index].start
                    <= neighbors_bins[neighbors_bins_index].start)):

                    new_bins.append(subdivided_bins[subdivided_bins_index])
                    subdivided_bins_index += 1

                elif neighbors_bins_index < len(neighbors_bins):
                    new_bins.append(neighbors_bins[neighbors_bins_index])
                    neighbors_bins_index += 1

                if (empty_bin_index == len(empty_bins) and
                    subdivided_bins_index == len(subdivided_bins) and
                    neighbors_bins_index == len(neighbors_bins)):
                    break

            for new_bin in new_bins:
                new_bin.reset()
            # Update the bins
            self.__bins[key] = new_bins


    def mark_bins(self, bins: List[Bin]) -> List[int]:
        """
        Mark bins based on their value and their neighbors' values.

        This method assigns a mark to each bin based on the following criteria:
        - If the bin's value is greater than 0, it is marked as ROUTER.
        - If the bin's value is 0 but it has a neighbor with a value greater than 0,
        it is marked as NEIGHBOR.
        - If the bin's value is 0 and it has no neighbors with a value greater than 0,
        it is marked as EMPTY.

        Args:
            bins (List[Bin]): A list of Bin objects to be marked.

        Returns:
            List[int]: A list of marks corresponding to each bin.
        """
        marks = []
        for i, r_bin in enumerate(bins):
            if r_bin.value > 0:
                marks.append(ROUTER)
            elif (i > 0 and bins[i - 1].value > 0) or (i < len(bins) - 1 and bins[i + 1].value > 0):
                marks.append(NEIGHBOR)
            else:
                marks.append(EMPTY)
        return marks

    def merge_empty_bins(self, bins: List[Bin], marks: List[int]) -> Tuple[List[Bin], int]:
        """
        Merge consecutive empty bins into larger bins.

        This method scans through the list of bins and merges consecutive bins marked as
        EMPTY into larger bins.
        It also counts the number of bins that are freed up as a result of the merging.

        Args:
            bins (List[Bin]): A list of Bin objects to be merged.
            marks (List[int]): A list of marks corresponding to each bin, indicating whether
            the bin is EMPTY, NEIGHBOR, or ROUTER.

        Returns:
            Tuple[List[Bin], int]: A tuple containing:
                - A list of merged Bin objects.
                - The number of bins that were freed up as a result of the merging.
        """
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

    def subdivide_bins(self,
                       bins: List[Bin],
                       marks: List[int],
                       freed_bins: int
                       ) -> Tuple[List[Bin], List[Bin]]:
        """
        Subdivide bins into used and neighbor bins.

        This method separates the bins into two categories:
        - Used bins: Bins marked as ROUTER.
        - Neighbor bins: Bins marked as NEIGHBOR.

        Args:
            bins (List[Bin]): A list of Bin objects to be subdivided.
            marks (List[int]): A list of marks corresponding to each bin,
            indicating whether the bin is EMPTY, NEIGHBOR, or ROUTER.
            freed_bins (int): The number of bins that were freed up during the merging process.

        Returns:
            Tuple[List[Bin], List[Bin]]: A tuple containing:
                - A list of used Bin objects.
                - A list of neighbor Bin objects.
        """
        used_bins = [bin for i, bin in enumerate(bins) if marks[i] == ROUTER]
        neighbors_bins = [bin for i, bin in enumerate(bins) if marks[i] == NEIGHBOR]
        new_bins = []

        # Determine the total number of new bins to create
        total_new_bins = len(used_bins) + freed_bins
        num_new_bins = max(1, total_new_bins // len(used_bins))
        for index, u_bin in enumerate(used_bins):
            num_new_bins = max(1, total_new_bins // (len(used_bins)- index))
            bin_size = (u_bin.end - u_bin.start) // num_new_bins
            if bin_size < 1:
                bin_size = 1
                num_new_bins = u_bin.end - u_bin.start

            start = u_bin.start
            for _ in range(num_new_bins - 1):
                total_new_bins -= 1
                end = start + bin_size
                new_bins.append(Bin(start, end))
                start = end

            # Ensure the last bin ends exactly at bin.end
            new_bins.append(Bin(start, u_bin.end))

        return new_bins, neighbors_bins

    def subdivide_empty_bins(self, empty_bins: List[Bin], occupied_bins: int) -> List[Bin]:
        """
        Subdivide empty bins into smaller bins.

        This method subdivides the empty bins into smaller bins based on the remaining available
        bins.
        It ensures that the largest empty bin gets an extra subdivision.

        Args:
            empty_bins (List[Bin]): A list of empty Bin objects to be subdivided.
            occupied_bins (int): The number of bins that are already occupied.

        Returns:
            List[Bin]: A list of newly subdivided empty Bin objects.
        """
        remaining_bins = MAX_BINS - occupied_bins
        max_empty_bins = max(empty_bins, key=lambda bin: bin.end - bin.start)
        new_empty_bins = []
        if remaining_bins > 0 and empty_bins:
            for index, e_bin in enumerate(empty_bins):
                num_new_bins = max(1, remaining_bins // (len(empty_bins)-index))
                if e_bin == max_empty_bins:
                    num_new_bins +=1
                bin_size = (e_bin.end - e_bin.start) // num_new_bins
                if bin_size < 1:
                    bin_size = 1
                    num_new_bins = e_bin.end - e_bin.start
                start = e_bin.start
                for _ in range(num_new_bins - 1):
                    remaining_bins -= 1
                    end = start + bin_size
                    new_empty_bins.append(Bin(start, end))
                    start = end
                # Ensure the last bin ends exactly at bin.end
                new_empty_bins.append(Bin(start, e_bin.end))

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
        route_index = self.ecmp_hash(source_ip=source_ip, dest_ip=dest_ip,
                                     flow_label=flow_label) % num_paths
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
