import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

class ObjectDetector:
    """目标检测与追踪器"""
    
    def __init__(self, model_path='VD_s.pt'):
        # 硬件配置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        self.model = YOLO(model_path).to(self.device)
        self.model.fuse()  # 融合模型层以提高推理速度

        # 模型参数
        self.imgsz = 640  # 输入图像尺寸
        self.half = self.device != 'cpu'  # 是否使用半精度推理

        # 轨迹管理
        self.track_history = defaultdict(lambda: {'points': [], 'last_seen': 0})
        self.max_trail_length = 30  # 最大轨迹点数量
        self.track_timeout = 30  # 轨迹消失超时帧数

        # 类别配置
        self.class_config = {
            'vehicles': {'ids': [2, 3, 5, 7], 'color': (255, 255, 255)},  # 车辆类
            'pedestrians': {'ids': [0], 'color': (0, 0, 255)},  # 行人类
            'two_wheelers': {'ids': [1, 4, 6, 8, 9], 'color': (0, 165, 255)}  # 两轮车类
        }

        # 乘骑检测参数
        self.proximity_thresh = 0.08  # 空间接近阈值
        self.dir_angle_thresh = 30  # 运动方向一致性阈值(度)
        self.min_hist_frames = 10  # 最小历史帧数要求

    def detect(self, frame):
        """执行目标检测与追踪"""
        results = self.model.track(
            source=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            imgsz=self.imgsz,
            half=self.half,
            device=self.device,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml"
        )

        vis_frame = frame.copy()
        raw_detections = []
        vehicle_ids = set()
        pedestrian_ids = set()

        # 解析检测结果
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, tid, cid in zip(boxes, track_ids, class_ids):
                det = {
                    "bbox": box,
                    "class_id": cid,
                    "class_name": self.model.names[cid],
                    "track_id": tid
                }
                raw_detections.append(det)
                
                # 分类ID集合
                if cid in self.class_config['vehicles']['ids']:
                    vehicle_ids.add(tid)
                elif cid in self.class_config['pedestrians']['ids']:
                    pedestrian_ids.add(tid)

        # 检测乘骑关系
        suppressed_ids = self._detect_riding_relations(raw_detections, frame.shape)

        # 可视化处理
        self._draw_detections(vis_frame, raw_detections, suppressed_ids)
        vis_frame = self._update_tracks(raw_detections, vis_frame)

        return vis_frame, raw_detections, vehicle_ids, pedestrian_ids

    # 目标类别判断辅助函数
    def _is_vehicle(self, d): 
        return d['class_id'] in self.class_config['vehicles']['ids']
    
    def _is_pedestrian(self, d): 
        return d['class_id'] in self.class_config['pedestrians']['ids']
    
    def _is_two_wheeler(self, d): 
        return d['class_id'] in self.class_config['two_wheelers']['ids']

    def _detect_riding_relations(self, detections, frame_shape):
        """检测需要抑制显示的目标（乘骑关系）"""
        suppressed = set()
        pedestrians = [d for d in detections if self._is_pedestrian(d)]
        two_wheelers = [d for d in detections if self._is_two_wheeler(d)]

        # 检查所有行人-两轮车组合
        for ped in pedestrians:
            for vehicle in two_wheelers:
                if self._check_riding_relation(ped, vehicle, frame_shape):
                    suppressed.add(vehicle['track_id'])
                    
        return suppressed

    def _check_riding_relation(self, ped, vehicle, shape):
        """乘骑关系判定逻辑"""
        return (self._check_proximity(ped['bbox'], vehicle['bbox'], shape) and 
                self._check_movement_consistency(
                    self.track_history.get(ped['track_id'], {'points': []}),
                    self.track_history.get(vehicle['track_id'], {'points': []})
                ))

    def _check_proximity(self, box1, box2, frame_shape):
        """空间接近性检测"""

        # 计算两个框的中心点
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2
        
        # 计算归一化距离
        img_diag = np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        return distance < self.proximity_thresh * img_diag

    def _check_movement_consistency(self, track1, track2):
        """运动方向一致性检测"""

        # 检查轨迹历史长度是否足够
        if len(track1['points']) < self.min_hist_frames or len(track2['points']) < self.min_hist_frames:
            return False
            
        # 获取最近的历史点
        pts1 = track1['points'][-self.min_hist_frames:]
        pts2 = track2['points'][-self.min_hist_frames:]
        
        # 计算平均移动向量
        vec1 = np.mean([np.array(pts1[i]) - np.array(pts1[i-1]) for i in range(1, len(pts1))], axis=0)
        vec2 = np.mean([np.array(pts2[i]) - np.array(pts2[i-1]) for i in range(1, len(pts2))], axis=0)
        
        # 计算方向夹角
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2) + 1e-9)
        angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        return angle_deg < self.dir_angle_thresh

    def _draw_detections(self, frame, detections, suppressed):
        """绘制检测框"""
        for det in detections:
            # 跳过被抑制的目标
            if det['track_id'] in suppressed: 
                continue
            
            # 根据类别确定颜色
            if self._is_vehicle(det):
                color = self.class_config['vehicles']['color']
            elif self._is_pedestrian(det):
                color = self.class_config['pedestrians']['color']
            else:
                color = self.class_config['two_wheelers']['color']
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    def _update_tracks(self, detections, frame):
        """更新轨迹并绘制方向箭头"""

        # ========== 可调参数区域 ============
        ARROW_LENGTH = 15           # 箭头长度
        MIN_MOVEMENT = 0.8          # 视为静止的移动阈值（像素/帧）
        DIRECTION_HOLD_FRAMES = 5   # 方向保持帧数（调节刷新频率）
        MIN_THICK = 1               # 最细线宽
        MAX_THICK = 4               # 最粗线宽 
        SPEED_SCALE = 0.5           # 速度敏感度
        ARROW_COLOR = (0, 200, 100) # 箭头颜色
        # ===================================
        
        current_ids = set()
        
        # 更新轨迹点
        for det in detections:
            track_id = det['track_id']
            current_ids.add(track_id)
            
            # 计算中心点
            box = det['bbox']
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            
            # 更新轨迹历史
            self.track_history[track_id]['points'].append((cx, cy))
            self.track_history[track_id]['last_seen'] = 0
            
            # 限制轨迹历史长度
            if len(self.track_history[track_id]['points']) > self.max_trail_length:
                self.track_history[track_id]['points'].pop(0)
        
        # 清理过期轨迹
        for track_id in list(self.track_history.keys()):
            if track_id not in current_ids:
                self.track_history[track_id]['last_seen'] += 1
                if self.track_history[track_id]['last_seen'] > self.track_timeout:
                    del self.track_history[track_id]
        
        # 绘制轨迹方向箭头
        for track_id, data in self.track_history.items():
            points = data['points']
            if len(points) < 2:  # 需要至少两个点才能确定方向
                continue

            # 初始化方向保持计数器
            if 'dir_hold_counter' not in data:
                data['dir_hold_counter'] = 0
                data['last_valid_angle'] = None
                data['last_movement'] = 0.0

            # 方向保持逻辑：减少箭头抖动
            data['dir_hold_counter'] += 1
            if data['dir_hold_counter'] < DIRECTION_HOLD_FRAMES and data['last_valid_angle'] is not None:
                angle = data['last_valid_angle']
                movement = data['last_movement']
            else:
                # 重置计数器
                data['dir_hold_counter'] = 0
                
                # 计算加权移动量（最近的点权重更高）
                dx, dy = 0, 0
                max_samples = min(3, len(points)-1)
                weights = [0.6, 0.3, 0.1][:max_samples]
                
                for i in range(1, max_samples+1):
                    dx += (points[-i][0] - points[-i-1][0]) * weights[i-1]
                    dy += (points[-i][1] - points[-i-1][1]) * weights[i-1]
                
                # 计算移动量
                movement = np.hypot(dx, dy)
                data['last_movement'] = movement
                
                # 更新有效角度
                if movement > MIN_MOVEMENT:
                    angle = np.arctan2(dy, dx)
                    data['last_valid_angle'] = angle
                else:
                    angle = data['last_valid_angle']  # 保持上次有效角度

            # 跳过无效角度
            if angle is None:
                continue

            # 动态计算线宽（速度越快线越粗）
            line_thickness = int(np.clip(
                data['last_movement'] * SPEED_SCALE,
                MIN_THICK, 
                MAX_THICK
            ))

            # 计算箭头终点
            end_x = int(points[-1][0] + ARROW_LENGTH * np.cos(angle))
            end_y = int(points[-1][1] + ARROW_LENGTH * np.sin(angle))
            
            # 绘制方向箭头
            cv2.arrowedLine(
                frame,
                (int(points[-1][0]), int(points[-1][1])),
                (end_x, end_y),
                ARROW_COLOR,
                line_thickness,
                tipLength=0.3
            )
        
        return frame
    