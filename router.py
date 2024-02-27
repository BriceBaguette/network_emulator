import numpy as np

class ForwardTableElement:
    
    def __init__(self,dest, next_hop, cost):
        self.dest = dest,
        self.next_hop = next_hop,
        self.cost = cost,

class Router:
    
    def __init__(self,node_name, active,
                 ip_address,):
        self.node_name = node_name
        self.ip_address = ip_address
        self.active = active
        self.forward_table = np.empty(0)
        
    def updateForwardTable(self, element):
        np.add(self.forward_table, element)
        