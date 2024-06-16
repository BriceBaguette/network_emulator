import re
from typing import List
from router import Router
from link import Link

def get_router_by_name(name: str, routers: List[Router]) -> Router:
    for router in routers:
        if router.node_name == name:
            return router
    return None

with open("../other_typo/my_topo.txt", "r") as file:
    topo = file.read()

routers_info = topo.split("\n\n")

router_id_pattern = r"Router ID:\s*(\S+)"

hostname_pattern = r"Hostname:\s*(\S+)"

ip_address_pattern = r"IP Address:\s*(\S+)"

routers: List[Router] = []
links: List[Link] = []

for router in routers_info:
    router_match = re.search(router_id_pattern, router)
    host_match = re.search(hostname_pattern, router)
    ip_match = re.search(ip_address_pattern, router)
    if router_match and host_match and ip_match:
        router_id = router_match.group(1)
        hostname = host_match.group(1)
        ip_address = ip_match.group(1)
        routers.append(Router(router_id=router_id, node_name=hostname, ip_address=ip_address, active=1))
    
pattern = re.compile(r'Metric:\s*\d+\s+IS-Extended.*?(?=Metric:|\Z)', re.DOTALL)
  
delay_pattern = r"Link Average Delay:\s*(\d+)"

neighbor_pattern = r"Metric:\s*(\d+).*IS-Extended (\S+)\.\d+"
for router in routers_info:
    host_match = re.search(hostname_pattern, router)
    if host_match:
        hostname = host_match.group(1)
        source: Router = get_router_by_name(hostname, routers)
        # Find all matches
        matches = pattern.findall(router) 
        for match in matches:
            neighbor_match = re.search(neighbor_pattern, match, re.DOTALL)
            delay_match = re.search(delay_pattern, match)
            if neighbor_match and delay_match:
                neighbor = get_router_by_name(neighbor_match.group(2), routers)
                if neighbor:
                    link = Link(link_id=len(links), source=source.ip_address, destination=neighbor.ip_address, delay=delay_match.group(1), cost=neighbor_match.group(1))
                    links.append(link)
            
for router in routers:
    print(f"Router ID: {router.id}, Hostname: {router.node_name}, IP Address: {router.ip_address}")
    print("\n")
        
for link in links:
    print(f"Link ID: {link.id}, Source: {link.source}, Destination: {link.destination}, Delay: {link.delay}, Cost: {link.cost}")
    print("\n")




