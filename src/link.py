"""
This module defines the Link class, which represents a link between two routers in a network.
"""

class Link:
    """
    Represents a link between two routers in a network.

    Attributes:
        id (int): The ID of the link.
        source (str): The IP address of the source router.
        destination (str): The IP address of the destination router.
        delay (float): The delay of the link.
        cost (float): The cost of the link.
        failure (int): The status of the link. 0 is safe, 1 is fast, 2 is medium, 3 is long.
        break_time (int): The counter to determine the time since the link is broken.
        number_of_paths (int): The number of paths that use this link.
    """

    def __init__(self, link_id, source, destination, delay, cost):
        self.id = link_id
        self.source = source
        self.destination = destination
        self.delay = delay
        self.cost = cost
        self.failure = 0
        self.break_time = 0
        self.number_of_paths = 0

    def to_json(self):
        """
        Converts the Link object to a JSON representation.

        Returns:
            dict: The JSON representation of the Link object.
        """
        return {
            "id": self.id,
            "source": self.source,
            "destination": self.destination,
            "delay": self.delay,
            "cost": self.cost,
            "failure": self.failure,
        }
