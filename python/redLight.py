import cv2
import numpy as np
from ultralytics import YOLO
import Car as Car
import trafficLight

model = YOLO("yolo11x.pt")
#roi_points = np.array([[[730, 463], [842, 388], [947, 472], [834, 560]]], np.int3
# 2)
carDict = {}
frame_count = 0 
RorG = "GREEN"
cap = cv2.VideoCapture(2)
ret, frame = cap.read()
height, width, channels = frame.shape
print(height, width)
line_point1 = (int(0.27*width), int(0.55*height))
line_point2 = (int(0.9*width), int(0.58*height))
"""
def is_crossing_line(p1, p2, l1, l2):
    
    使用向量叉積判斷線段 (p1,p2) 是否與 (l1,l2) 相交
    p1, p2: 車輛前一幀與這一幀的中心點
    l1, l2: 判定線的兩端
    
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    # 簡單的快速排斥實驗與跨立實驗邏輯
    cp1 = cross_product(l1, l2, p1)
    cp2 = cross_product(l1, l2, p2)
    cp3 = cross_product(p1, p2, l1)
    cp4 = cross_product(p1, p2, l2)

    # 如果兩組點分別在對方的兩側，則代表相交
    if ((cp1 * cp2 < 0) and (cp3 * cp4 < 0)):
        return True
    return False
"""

def detect_traffic_light_color(roi):
    """
    輸入：紅綠燈 ROI（BGR）
    輸出：'RED' / 'GREEN' / 'UNKNOWN'
    """
    if roi is None or roi.size == 0:
        return (225,225,225)#"UNKNOWN"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 紅色（兩段）
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    # 綠色
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + \
                cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    red_count = cv2.countNonZero(mask_red)
    green_count = cv2.countNonZero(mask_green)
    threshold = 100  # 面積門檻，避免雜訊
    if red_count > green_count and red_count > threshold:
        return "RED",(0,0,225)#"RED"
    elif green_count > red_count and green_count > threshold:
        return "GREEN",(0,225,0)#"GREEN"
    else:
        return "UNKNOWN",(225,225,225)#"UNKNOWN"
while cap.isOpened():
    ret, frame = cap.read()
    frame_count+=1
    
    #cv2.polylines(frame, [roi_points], True, (0, 0, 255), 2)
    #=======windowns => device = 0; mac => device='mps'=======
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes = [2,3,5,7,9], verbose=False, device='mps')
    #results = model.track(frame, persist=True, tracker="my_track.yaml", classes=[2], verbose=False)
    if results[0].boxes.id is not None:
        # 取得 座標、ID 與 類別
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        for box, track_id , cls,conf in zip(boxes, ids, clss,confs):
            #print(track_id, cls)
            x1, y1, x2, y2 = box
            color = (0, 255, 0)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            #print("=====")
            #print(track_id)
            if cls == 9:
                roi = frame[y1:y2, x1:x2]
                RorG, color = trafficLight.trafficLight.detect_traffic_light_color(roi)
                cv2.line(frame, line_point1, line_point2, color, 3)
            elif cls in [2,3,5,7]:
                if track_id not in carDict:
                    #print("ENter")
                    carDict[track_id] = Car.Car(cord=[x1,y1,x2,y2], illegal=False, frameAppear=frame_count, lastCenter = (center_x, center_y), crossed=False)#加入物件 不是改動
                    """
                    hit = cv2.pointPolygonTest(roi_points,(float(center_x), float(center_y)),False)
                    if hit >= 0:
                    # 在 ROI 內：畫紅框
                        color = (0, 0, 255)
                        carDict[track_id].illegal = True
                    """
                elif carDict[track_id].illegal == True:
                    color = (0,0,225)
                else:
                    
                    carDict[track_id].crossed = Car.line_crossed(carDict[track_id],(center_x, center_y),line_point1, line_point2)
                    cv2.putText(frame, f"ID: {track_id}_CLS:{cls} CROSSED:{carDict[track_id].crossed}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                if RorG == "GREEN" and carDict[track_id].crossed == True:
                    carDict[track_id].illegal = True
                    color = (0,0,225)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (center_x, center_y), 5, (0,225,0), -1)
            cv2.putText(frame, f"ID: {track_id}_CLS:{cls}\n x1x2{int(x1),int(x2)}\n y1y2{int(y1),int(y2)}", (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv2.imshow("ByteTrack Detection", frame)
    if frame_count % 20 == 0: #定期刪除不然會炸掉
        print("frame_count renew")
        deleteList = []
        for key, value in carDict.items():
            #print(frame_count - value.printFrameAppear())
            if value.returnIllegal() == False and frame_count - value.returnFrameAppear() > 50:
                print(key,"deleted")
                deleteList.append(key)
        for key in deleteList:
            del carDict[key]
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 33
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        #triggered_ids.clear()
        print("已重置追蹤名單")
cap.release()
cv2.destroyAllWindows()
for key, value in carDict.items():
    isLegal = value.returnIllegal()
    print(f"{key} : {isLegal}")
