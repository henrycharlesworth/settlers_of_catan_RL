from game.enums import Resource

class Harbour(object):
    def __init__(self, resource: Resource = None, exchange_value=3, id=None):
        self.resource = resource
        self.exchange_value = exchange_value
        self.corners = []
        self.edge = None
        self.id = id