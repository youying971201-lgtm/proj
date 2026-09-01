import os
import json
import cv2
import yt_dlp
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# 開啟跨網域權限
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定義儲存資料夾路徑
SAVE_FOLDER = "cord_data"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# ==================== 影片串流處理 ====================
def generate_frames():
    yt_url = "https://www.youtube.com/watch?v=wqctLW0Hb_0" 
    ydl_opts = {'format': 'best'}
    
    # 取得影片串流網址
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(yt_url, download=False)
        stream_url = info['url']
        
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("❌ 無法讀取影片！")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 將 OpenCV 的畫面編碼成 JPEG 格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # 使用 yield 產生 HTTP 串流格式 (MJPEG)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.get("/video_feed")
def video_feed():
    """ 提供前端影像來源的 API """
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# ==================== 接收座標處理 ====================
@app.post("/save_points")
async def save_points(request: Request):
    """ 接收前端傳來的座標並存成 JSON """
    data = await request.json()
    
    file_path = os.path.join(SAVE_FOLDER, "points_data.json")
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ 成功！座標已存入: {file_path}")
    return {"status": "success", "message": f"Saved to {file_path}"}


if __name__ == "__main__":
    print("🚀 伺服器啟動中... 請開啟 index.html")
    uvicorn.run(app, host="127.0.0.1", port=8000)