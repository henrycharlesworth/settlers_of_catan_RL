from game.enums import BuildingType, PlayerId

class Building(object):
    def __init__(self, type: BuildingType, owner: PlayerId, corner: None):
        self.type = type
        self.owner = owner
        self.corner = corner