
class Link:
    
    def __init__(self, source, destination, delay, cost):
        self.source = source # IP address of the source router
        self.destination = destination # IP address of the destination router
        self.delay = delay # Delay of the link
        self.cost = cost # Cost of the link
        self.failure = False # Status of the link
        