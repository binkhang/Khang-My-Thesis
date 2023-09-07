import cv2

def gstreamer_pipeline(
        camera_id=0,
        capture_width=960, #1920
        capture_height=540, #1080
        display_width=960,
        display_height=540,
        framerate=30,
        flip_method=0,
        ):
        return (
                "nvarguscamerasrc sensor-id=%d ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True"
                % (
                    camera_id,
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
                )
        )
        