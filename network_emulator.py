from router import Router, ForwardTableElement
from link import Link

import utils
import numpy as np
import json
import base64
import time


class NetworkEmulator:

    def __init__(self, node_file, link_file, generation_rate, num_generation):
        self.routers = list()
        self.links = list()

        self.node_file = node_file
        self.link_file = link_file
        self.generation_rate = generation_rate
        self.num_generation = num_generation

    def build(self):
        self.__build_routers()
        self.__build_links()
        print("Network build with " + str(len(self.routers)) +
              " routers and " + str(len(self.links)) + " links")

    def __build_routers(self):
        with open(self.node_file, 'r') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error: JSON parsing failed for file '{
                      self.node_file}' ")
                return 1

            for obj in json_data:
                node_name, base64_str = None, None

                attributes_obj = obj.get('attributes', {})
                ls_attribute_obj = attributes_obj.get('ls_attribute', {})
                node_name_obj = ls_attribute_obj.get('node_name', {})
                if node_name_obj:
                    node_name = node_name_obj.strip()

                nlri = obj.get('nlri', {}).get(
                    'nlri_data', {}).get('lsnlri', {})
                local_node_descriptor = nlri.get('local_node_descriptor', {})
                igp_router_id = local_node_descriptor.get(
                    'igp_router_id', {}).get('$binary', {}).get('base64', None)
                if igp_router_id:
                    base64_str = igp_router_id.strip()
                self.routers.append(
                    Router(node_name=node_name, active=1, ip_address=base64_str))

    def __build_links(self):
        with open(self.link_file, 'r') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error: JSON parsing failed for file '{
                      self.link_file}'")
                return 1

        for obj in json_data:
            source, destination, delay, igp_metric = None, None, 0, 0

            nlri = obj.get('nlri', {}).get('nlri_data', {}).get('lsnlri', {})

            local_node_descriptor = nlri.get('local_node_descriptor', {})
            igp_router_id = local_node_descriptor.get(
                'igp_router_id', {}).get('$binary', {}).get('base64', None)
            if igp_router_id:
                source = igp_router_id.strip()

            remote_node_descriptor = nlri.get('remote_node_descriptor', {})
            igp_router_id = remote_node_descriptor.get(
                'igp_router_id', {}).get('$binary', {}).get('base64', None)
            if igp_router_id:
                destination = igp_router_id.strip()

            attributes_obj = obj.get('attributes', {})
            ls_attribute_obj = attributes_obj.get('ls_attribute', {})

            igp_metric = ls_attribute_obj.get('igp_metric', 0)

            unidir_delay_obj = ls_attribute_obj.get('unidir_delay', {})
            delay = unidir_delay_obj.get('delay', 0)
            link = Link(source=source, destination=destination,
                        delay=delay, cost=igp_metric)
            self.links.append(link)

    def start(self):
        start = time.time()
        number_of_routers = len(self.routers)

        # Construct graph using a sparse matrix representation
        net_graph = np.zeros((number_of_routers, number_of_routers), dtype=int)

        for i, router in enumerate(self.routers):
            indices = [index for index, value in enumerate(
                self.links) if value.source == router.ip_address]
            for value in indices:
                link = self.links[value]
                destination = next(index for index, value in enumerate(
                    self.routers) if value.ip_address == link.destination)
                net_graph[i, destination] = self.links[value].cost

        end = time.time()
        print("Build graph in: {}".format(end - start))

        for i in range(number_of_routers):
            print(i)
            pathList = utils.dijkstra(net_graph, i)

            for j in range(number_of_routers):
                elements = pathList[j]
                for k in range(len(elements)):
                    self.routers[i].updateForwardTable(utils.ForwardTableElement(
                        dest=self.routers[elements[k][0]].ip_address,
                        next_hop=self.routers[elements[k][1]].ip_address,
                        cost=elements[k][2]
                    ))

        end = time.time()
        print("Network started in: {}".format(end - start))

    def emulate(self, source, destination):
        for _ in range(self.num_generation):
            index = next(index for index, value in enumerate(
                self.routers) if value.ip_address == source)
            source_router = self.routers[index]
            self.send_probs(source=source_router, destination=destination)

    def send_probs(self, source, destination):

        latency = np.zeros(self.generation_rate)

        for i in range(self.generation_rate):
            indices = [index for index, value in enumerate(
                source.forward_table) if value.destination == destination]
            index = next(index for index, value in enumerate(self.links) if value.source ==
                         source.ip_address and value.destination == source.forward_table[indices[0]].next_hop)
            latency[i] += self.links[index].delay
            index = next(index for index, value in enumerate(
                self.routers) if value.ip_address == source.forward_table[indices[0]].next_hop)
            next_router = self.routers[index]

            while (next_router.ip_address != destination):
                indices = [index for index, value in enumerate(
                    next_router.forward_table) if value.destination == destination]
                index = next(index for index, value in enumerate(self.links) if value.source ==
                            next_router.ip_address and value.destination == next_router.forward_table[indices[0]].next_hop)
                latency[i] += self.links[index].delay
                index = next(index for index, value in enumerate(
                    self.routers) if value.ip_address == next_router.forward_table[indices[0]].next_hop)
                next_router = self.routers[index]
        
        return latency
