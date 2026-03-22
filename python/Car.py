class Car:
    """Represents a car."""

    def __init__(self, cord, illegal, frameAppear,lastCenter,crossed):
        """Initialize car attributes."""
        self.cord = cord
        self.illegal = illegal
        self.frameAppear = frameAppear
        self.lastCenter = lastCenter
        self.crossed = crossed
    def returnIllegal(self):
        #print(self.illegal)
        return self.illegal
    def returnFrameAppear(self):
        #print(self.illegal)
        return self.frameAppear
    def returnLastCenter(self):
        return self.lastCenter
    def setLastCenter(self, lastCenter):
        self.lastCenter = lastCenter
    def returnCrossed(self):
        return self.crossed
    
    
def line_crossed(self, currentCenter, l1, l2):
    """
    使用向量叉積判斷線段 (p1,p2) 是否與 (l1,l2) 相交
    p1, p2: 車輛前一幀與這一幀的中心點
    l1, l2: 判定線的兩端
    """
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # 簡單的快速排斥實驗與跨立實驗邏輯
    cp1 = cross_product(l1, l2, self.returnLastCenter())
    cp2 = cross_product(l1, l2, currentCenter)
    cp3 = cross_product(self.returnLastCenter(), currentCenter, l1)
    cp4 = cross_product(self.returnLastCenter(), currentCenter, l2)

    # 如果兩組點分別在對方的兩側，則代表相交
    if ((cp1 * cp2 < 0) and (cp3 * cp4 < 0)):
        return True
    return False
    
