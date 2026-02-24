class Car:
    """Represents a car."""

    def __init__(self, cord, illegal, frameAppear,lastCenter,crossed):
        """Initialize car attributes."""
        self.cord = cord
        self.illegal = illegal
        self.frameAppear = frameAppear
        self.lastCenter = lastCenter
        self.crossed = crossed
    def printIllegal(self):
        #print(self.illegal)
        return self.illegal
    def printFrameAppear(self):
        #print(self.illegal)
        return self.frameAppear
    def printLastCenter(self):
        return self.lastCenter
    def setLastCenter(self, lastCenter):
        self.lastCenter = lastCenter
    def printCrossed(self):
        return self.crossed
    
    
