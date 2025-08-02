import os
import torch
import cv2
import time
import traceback
import numpy as np
from tkinter import Tk, filedialog
from collections import defaultdict
from detector import ObjectDetector
from counter import TrafficCounter

# ========== 环境配置 ==========
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True

def detect_available_cameras(max_test=3):
    """检测可用的摄像头设备"""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            info = f"设备 {i}"
            if ret and frame is not None:
                info += f" ({frame.shape[1]}x{frame.shape[0]})"
            available.append({"index": i, "info": info})
            cap.release()
        time.sleep(0.1)
    return available

def get_user_choice():
    """获取用户选择的输入源"""
    print("\n" + "="*40)
    print("交通流量检测系统 - 输入源选择")
    print("1. 使用摄像头实时检测")
    print("2. 使用视频文件检测")
    print("="*40)
    
    while True:
        choice = input("请选择输入源 (1/2): ").strip()
        if choice in ["1", "2"]:
            return choice
        print("无效输入，请重新选择")

def get_video_path():
    """使用文件对话框选择视频文件"""
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
    )
    root.destroy()
    return path

class CameraManager:
    """摄像头/视频源管理类"""    
    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.is_video = isinstance(source, str)
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def initialize(self):
        """初始化视频源"""
        self.release()
        try:
            if self.is_video:
                self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                return False
                
            if not self.is_video:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            return True
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            return False

    def read_frame(self):
        """读取下一帧"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        if not ret:
            self.release()
            return False, None
        return True, frame

    def release(self):
        """释放资源"""
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None

def main():
    # 获取用户输入源选择
    choice = get_user_choice()
    
    # 系统配置
    config = {
        "camera": {
            "source": 0,
            "width": 1280,
            "height": 720,
            "fps": 60
        },
        "detection": {
            "model_path": "VD_s.pt",
            "process_interval": 1
        },
        "display": {
            "show_fps": True,
            "window_name": "TrafficFlowDetector"
        }
    }
    
    # 根据用户选择配置输入源
    if choice == "1":
        cams = detect_available_cameras()
        if not cams:
            print("未检测到可用摄像头")
            return
            
        print("\n检测到以下摄像头设备：")
        for cam in cams:
            print(f"[{cam['index']}] {cam['info']}")
            
        while True:
            try:
                cam_idx = int(input("请输入摄像头编号: "))
                if any(cam["index"] == cam_idx for cam in cams):
                    config["camera"]["source"] = cam_idx
                    break
                print("无效的摄像头编号")
            except ValueError:
                print("请输入数字")
    else:
        video_path = get_video_path()
        if not video_path:
            print("未选择视频文件，程序退出")
            return
        config["camera"]["source"] = video_path

    # 初始化摄像头/视频源
    camera = CameraManager(**config["camera"])
    if not camera.initialize():
        print("初始化失败，请检查输入源")
        return

    # 获取目标分辨率（固定值）
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720
    
    # 创建显示窗口
    window_name = config["display"]["window_name"]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, TARGET_WIDTH, TARGET_HEIGHT)

    # 模型选择逻辑
    if choice == "1" and config["camera"]["source"] == 0:
        model_path = "yolov8n.pt"
        print("\n使用前置摄像头模式，加载yolov8n基础模型")
    else:
        model_path = config["detection"]["model_path"]
        print(f"\n使用自定义模型: {model_path}")

    # 初始化检测器和计数器
    detector = ObjectDetector(model_path)
    counter = TrafficCounter(config_path="config.json")

    # 运行状态变量
    frame_count = 0
    fps = 0
    last_time = time.time()
    paused = False
    video_ended = False
    last_valid_frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), np.uint8)

    print(f"创建主显示窗口: {window_name}")

    try:
        while True:
            # 处理帧读取
            if not paused and not video_ended:
                ret, frame = camera.read_frame()
                
                if not ret:
                    if camera.is_video:  # 视频结束
                        video_ended = True
                        paused = True
                        print("Playback Ended [SPACE]Replay  [Q]Quit")
                    else:  # 摄像头断开
                        time.sleep(0.1)
                else:
                    # 缩放并填充到目标尺寸
                    h, w = frame.shape[:2]
                    scale = min(TARGET_WIDTH/w, TARGET_HEIGHT/h)
                    resized = cv2.resize(frame, (int(w*scale), int(h*scale)))
                    
                    # 创建带黑边的画布
                    canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), np.uint8)
                    y_offset = (TARGET_HEIGHT - resized.shape[0]) // 2
                    x_offset = (TARGET_WIDTH - resized.shape[1]) // 2
                    canvas[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
                    last_valid_frame = canvas.copy()  # 使用固定尺寸
                    
                # 只有在成功获取帧时才进行检测处理
                if ret:
                    frame_count += 1
                    
                    # 计算FPS（每10帧）
                    if frame_count % 10 == 0:
                        fps = 10 / (time.time() - last_time)
                        last_time = time.time()
                    
                    # 检测处理（根据间隔）
                    if frame_count % config["detection"]["process_interval"] == 0:
                        processed_frame, raw_detections, vehicle_ids, ped_ids = detector.detect(last_valid_frame)
                        # 只有在非暂停和非结束状态下才更新计数器可视化
                        if not paused and not video_ended:
                            count_info = counter.update(
                                vehicle_ids=vehicle_ids,
                                pedestrian_ids=ped_ids,
                                raw_detections=raw_detections
                            )
                        last_valid_frame = processed_frame.copy()

            # ==== 显示处理 ====
            if video_ended:
                # 创建固定尺寸黑屏帧
                display_frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
                # 添加提示
                text = "Playback Finished [SPACE]Replay  [Q]Quit"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (TARGET_WIDTH - text_size[0]) // 2
                text_y = (TARGET_HEIGHT + text_size[1]) // 2
                cv2.putText(display_frame, text, 
                          (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                display_frame = last_valid_frame.copy()
                
                # 视频进度条（仅视频文件）
                if camera.is_video and camera.cap.isOpened():
                    total_frames = camera.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    current_frame = camera.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if total_frames > 0:
                        progress = current_frame / total_frames
                        cv2.rectangle(display_frame, 
                                     (10, TARGET_HEIGHT-20), 
                                     (int(10 + 300*progress), TARGET_HEIGHT-5), 
                                     (0, 255, 0), -1)
                
                # 显示FPS
                if config["display"]["show_fps"]:
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                               (10, TARGET_HEIGHT-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # 添加计数器可视化
                if not paused:
                    display_frame = counter.draw_visuals(display_frame)

            # 显示最终帧
            cv2.imshow(window_name, display_frame)

                        
            # ==== 键盘控制 ====
            key = cv2.waitKey(1 if not paused else 100) & 0xFF
            
            # 退出程序
            if key == ord('q') or key == 27:
                break
                
            # 暂停/继续
            if key == ord(' '):
                if video_ended:
                    # 重置视频
                    if camera.initialize():
                        video_ended = False
                        paused = False
                        frame_count = 0
                        # 重置最后一帧
                        ret, frame = camera.read_frame()
                        if ret:
                            last_valid_frame = frame.copy()
                        else:
                            last_valid_frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), np.uint8)
                else:
                    paused = not paused
                    
            # 重置计数器
            elif key == ord('r'):
                counter.history_ids = defaultdict(set)
                
            # 视频快退
            elif key == ord('a') and camera.is_video and not video_ended and camera.cap.isOpened():
                current_frame = camera.cap.get(cv2.CAP_PROP_POS_FRAMES)
                camera.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 10 * config["camera"]["fps"]))
                
            # 视频快进
            elif key == ord('d') and camera.is_video and not video_ended and camera.cap.isOpened():
                current_frame = camera.cap.get(cv2.CAP_PROP_POS_FRAMES)
                total_frames = camera.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                camera.cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames-1, current_frame + 10 * config["camera"]["fps"]))
                
    except Exception as e:
        print(f"程序异常: {str(e)}")
        print(traceback.format_exc())
    finally:
        # 清理资源
        camera.release()
        cv2.destroyAllWindows()
        print("系统资源已释放")

if __name__ == "__main__":
    main()
    