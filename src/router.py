"""
This module defines the Router class, which represents a router in a network 
and the ForwardTableElement class, which represents an element in the forwarding 
table of a router.
"""
import hashlib
from typing import List

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

    def has_entry_for_destination(self, dest_ip) -> bool:
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

    def update_forward_table(self, element) -> None: 
        """
        Updates the forward table of the router with the given element.

        Parameters:
        - element (ForwardTableElement): Element to add to the forward table.
        """
        for entry in self.forward_table:
            if entry.destination == element.destination and entry.next_hop == element.next_hop:
                return
        self.forward_table.append(element)

    def ecmp_hash(self,source_ip, dest_ip, flow_label):
        """
        Performs ECMP hashing to select a route based on the destination IP address, packet number, 
        and distribution key.
        
        Parameters:
        - dest_ip (str): Destination IP address.
        - flow_label (str): Flow label.

        Returns:
        - int: Hash value used for route selection.
        """
       
        hash_input = f"{self.id}{self.ip_address}{dest_ip}{flow_label}".encode('utf-8')

        hash_value = hashlib.sha256(hash_input).hexdigest()
        hash_int = int(hash_value, 16)

        return hash_int

    def select_route(self,source_ip, dest_ip, num_paths, flow_label):
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
