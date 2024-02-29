import numpy as np
import networkx as nx
import hashlib
import random

class ForwardTableElement:
    
    def __init__(self,dest, next_hop, ):
        self.destination = dest # IP address of the destination router
        self.next_hop = next_hop # IP address of the next hop router

class Router:
    
    def __init__(self,node_name, active,
                 ip_address,):
        self.node_name = node_name # Name of the router
        self.ip_address = ip_address # IP address of the router
        self.active = active # Status of the router
        self.forward_table = list() # Forwarding table for the router
        
    def updateForwardTable(self, element):
        # Add a new element to the forward table
        self.forward_table.append( element)

    def ecmp_hash(self, dest_ip, packet_nmbr):
        # Concatenate the input values to create a unique key
        hash_input = f"{self.ip_address}{dest_ip}{packet_nmbr}".encode('utf-8')


        # Use a hash function (e.g., MD5) to generate a hash value
        hash_value = hashlib.md5(hash_input).hexdigest()

        # Convert the hash value to an integer for ECMP hashing
        hash_int = int(hash_value, 16)

        return hash_int
    
    def select_route(self, dest_ip, num_paths, pkt_nbr):
        # Determine the route index based on the hash value and the number of paths
        route_index = self.ecmp_hash(dest_ip=dest_ip, packet_nmbr= pkt_nbr) % num_paths
        return route_index