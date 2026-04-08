import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# ==================== 配置區 ====================
VIDEO_SOURCE = 2  # 0 為預設攝影機, 或改為影片路徑如 "traffic.mp4"
MODEL_PATH = "yolo11x.pt"
DEVICE = 'mps'  # Windows: '0' 或 '0', Mac: 'mps'
CLASSES_TO_DETECT = [2, 3, 5, 7, 9]  # 2:car, 3:motorcycle, 5:bus, 7:truck, 9:traffic light

def detect_traffic_light_color(roi):
    if roi is None or roi.size == 0:
        return "UNKNOWN", (225, 225, 225)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ==========================================
    # 1. 紅色區間 (亮度 V 下限降至 120)
    # ==========================================
    lower_red1 = np.array([0, 40, 120])
    upper_red1 = np.array([15, 255, 255])
    
    lower_red2 = np.array([150, 40, 120])
    upper_red2 = np.array([180, 255, 255])

    # ==========================================
    # 2. 綠色區間 (飽和度 S 降至 40，亮度 V 降至 120)
    # ==========================================
    lower_green = np.array([35, 40, 120])
    upper_green = np.array([95, 255, 255])

    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                              cv2.inRange(hsv, lower_red2, upper_red2))
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    red_count = cv2.countNonZero(mask_red)
    green_count = cv2.countNonZero(mask_green)

    # ==========================================
    # 3. 【關鍵】面積門檻大幅降低
    # 原本是 100，現在降到 30。遠處的小燈泡也能被算進去。
    # ==========================================
    threshold = 30  

    # 💡 終極 Debug 技巧：把這行印出來看！
    # 如果畫面一直顯示 UNKNOWN，看終端機印出的數字就能知道是「差多少」才及格
    # print(f"Red: {red_count}, Green: {green_count}")

    if red_count > green_count and red_count > threshold:
        return "RED", (0, 0, 225)      
    elif green_count > red_count and green_count > threshold:
        return "GREEN", (0, 225, 0)    
    else:
        return "UNKNOWN", (225, 225, 225)
# ==================== 車輛類別 ====================
class Vehicle:
    def __init__(self, track_id, center):
        self.track_id = track_id
        self.last_center = center
        self.crossed_in = False   # 進入方向
        self.crossed_out = False  # 離開方向
        self.illegal = False
        self.frame_appear = 0
        self.cross_direction = None  # 'IN', 'OUT', 或 None
    
    def update_position(self, center):
        self.last_center = center

# ==================== 主程式 ====================
def main():
    # 初始化
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # 取得影片資訊
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影片")
        return
    
    height, width = frame.shape[:2]
    print(f"影片解析度: {width}x{height}")
    
    # 設定偵測線位置（可根據實際場景調整）
    line_start = sv.Point(x=50, y=500)
    line_end = sv.Point(x=2000, y=500)
    
    # Supervision 工具初始化
    tracker = sv.ByteTrack()
    line_counter = sv.LineZone(start=line_start, end=line_end)
    line_annotator = sv.LineZoneAnnotator()
    box_annotator = sv.BoxAnnotator()
    
    # 狀態變數
    vehicle_dict = {}  # 儲存所有車輛物件
    frame_count = 0
    traffic_light_state = "GREEN"
    traffic_light_color = (0, 255, 0)
    
    print("開始處理... 按 'q' 結束, 按 'r' 重置")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # YOLO 偵測與追蹤
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=CLASSES_TO_DETECT,
            verbose=False,
            device=DEVICE
        )
        
        if results[0].boxes.id is None:
            cv2.imshow("Traffic Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # 取得偵測結果
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        
        # 分離車輛和紅綠燈的 detections
        vehicle_detections_list = []
        vehicle_boxes = []
        vehicle_ids = []
        vehicle_classes = []
        vehicle_confs = []
        
        # 處理每個偵測物件
        for box, track_id, cls, conf in zip(boxes, ids, classes, confs):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center = (center_x, center_y)
            
            # 紅綠燈偵測
            if cls == 9:
                roi = frame[y1:y2, x1:x2]
                traffic_light_state, traffic_light_color = detect_traffic_light_color(roi)
                
                # 繪製紅綠燈框
                cv2.rectangle(frame, (x1, y1), (x2, y2), traffic_light_color, 3)
                cv2.putText(frame, traffic_light_state, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, traffic_light_color, 2)
            
            # 車輛處理
            elif cls in [2, 3, 5, 7]:
                vehicle_boxes.append(box)
                vehicle_ids.append(track_id)
                vehicle_classes.append(cls)
                vehicle_confs.append(conf)
                
                # 新車輛
                if track_id not in vehicle_dict:
                    vehicle_dict[track_id] = Vehicle(track_id, center)
                    vehicle_dict[track_id].frame_appear = frame_count
        
        # 使用 Supervision 建立車輛 detections 並觸發穿越計數
        if len(vehicle_boxes) > 0:
            vehicle_boxes_array = np.array(vehicle_boxes)
            vehicle_ids_array = np.array(vehicle_ids)
            vehicle_classes_array = np.array(vehicle_classes)
            vehicle_confs_array = np.array(vehicle_confs)
            
            detections = sv.Detections(
                xyxy=vehicle_boxes_array,
                tracker_id=vehicle_ids_array,
                class_id=vehicle_classes_array,
                confidence=vehicle_confs_array
            )
            
            # 觸發穿越線檢測
            crossed_in, crossed_out = line_counter.trigger(detections)
            
            # 更新車輛穿越狀態
            for i, track_id in enumerate(vehicle_ids_array):
                vehicle = vehicle_dict[track_id]
                
                # 記錄進入方向穿越
                if crossed_in[i] and not vehicle.crossed_in:
                    vehicle.crossed_in = True
                    vehicle.cross_direction = 'IN'
                    print(f"車輛 ID {track_id} 穿越線 (進入方向)")
                    
                    # 違規判定：紅燈時穿越
                    if traffic_light_state == "RED":
                        vehicle.illegal = True
                        print(f"  --> 違規！紅燈穿越")
                
                # 記錄離開方向穿越
                if crossed_out[i] and not vehicle.crossed_out:
                    vehicle.crossed_out = True
                    vehicle.cross_direction = 'OUT'
                    print(f"車輛 ID {track_id} 穿越線 (離開方向)")
                    
                    # 違規判定：紅燈時穿越
                    if traffic_light_state == "RED":
                        vehicle.illegal = True
                        print(f"  --> 違規！紅燈穿越")
                
                # 決定顏色和標籤
                if vehicle.illegal:
                    color = (0, 0, 255)  # 紅色 = 違規
                    label_text = "ILLEGAL"
                else:
                    color = (0, 255, 0)  # 綠色 = 正常
                    label_text = "OK"
                
                # 繪製車輛框與資訊
                x1, y1, x2, y2 = vehicle_boxes[i]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                
                # 顯示穿越方向
                cross_info = ""
                if vehicle.crossed_in:
                    cross_info += "IN "
                if vehicle.crossed_out:
                    cross_info += "OUT"
                
                info_text = f"ID:{track_id} {label_text} {cross_info}"
                cv2.putText(frame, info_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 更新位置
                vehicle.update_position((center_x, center_y))
        
        # 使用 Supervision 繪製偵測線（會顯示 IN/OUT 計數）
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        
        # 顯示紅綠燈狀態
        status_text = f"Traffic Light: {traffic_light_state} | Frame: {frame_count}"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, traffic_light_color, 2)
        
        # 顯示穿越統計
        stats_text = f"IN: {line_counter.in_count} | OUT: {line_counter.out_count}"
        cv2.putText(frame, stats_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 定期清理不活躍的車輛
        if frame_count % 20 == 0:
            delete_list = []
            for track_id, vehicle in vehicle_dict.items():
                if not vehicle.illegal and (frame_count - vehicle.frame_appear) > 50:
                    delete_list.append(track_id)
            
            for track_id in delete_list:
                del vehicle_dict[track_id]
            
            if delete_list:
                print(f"清理 {len(delete_list)} 個不活躍車輛")
        
        # 顯示畫面
        cv2.imshow("Traffic Monitoring System", frame)
        
        # 鍵盤控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            vehicle_dict.clear()
            line_counter.in_count = 0
            line_counter.out_count = 0
            print("已重置所有追蹤記錄")
    
    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()
    
    # 輸出最終違規統計
    print("\n========== 違規統計 ==========")
    illegal_count = 0
    for track_id, vehicle in vehicle_dict.items():
        if vehicle.illegal:
            illegal_count += 1
            direction = ""
            if vehicle.crossed_in:
                direction += "IN "
            if vehicle.crossed_out:
                direction += "OUT"
            print(f"車輛 ID {track_id}: 違規 (方向: {direction})")
    
    print(f"\n總追蹤車輛數: {len(vehicle_dict)}")
    print(f"違規車輛數: {illegal_count}")
    print(f"總進入數: {line_counter.in_count}")
    print(f"總離開數: {line_counter.out_count}")
    print("==============================")

if __name__ == "__main__":
    main()