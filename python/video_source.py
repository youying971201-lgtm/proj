import cv2


CAMERA_BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
}


def open_video_capture(
    video_source,
    camera_backend="auto",
    camera_width=None,
    camera_height=None,
    camera_fps=None,
):
    """開啟影片或攝影機；數字來源代表實體／虛擬攝影機索引。"""
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source, CAMERA_BACKENDS[camera_backend])
        if camera_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        if camera_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        if camera_fps:
            cap.set(cv2.CAP_PROP_FPS, camera_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(video_source)


def list_available_cameras(camera_backend="auto", max_index=10):
    """探測可由 OpenCV 開啟的攝影機索引與實際解析度。"""
    available = []
    for index in range(max_index):
        cap = open_video_capture(index, camera_backend=camera_backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                available.append((index, width, height, fps))
        cap.release()
    return available
