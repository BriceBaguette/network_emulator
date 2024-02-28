import numpy as np

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
        