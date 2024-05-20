
class Link:

    def __init__(self,id, source, destination, delay, cost):
        self.id = id
        self.source = source # IP address of the source router
        self.destination = destination # IP address of the destination router
        self.delay = delay # Delay of the link
        self.cost = cost # Cost of the link
        self.failure = 0 # Status of the link: 0 is safe, 1 is fast, 2 is medium, 3 is long
        self.break_time = 0 # Counter to determine the time since the link is broken
        