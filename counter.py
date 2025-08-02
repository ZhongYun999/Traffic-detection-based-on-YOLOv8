import cv2
import json
from collections import defaultdict
import os
import time

class TrafficCounter:
    def __init__(self, config_path):
        # 配置加载
        self.default_config = {
            "text_position": [20, 50],
            "class_alias": {
                "car": "vehicle", "van": "vehicle",
                "truck": "vehicle", "bus": "vehicle",
                "motorcycle": "vehicle",
                "two_wheelers": "pedestrian"  # 新增两轮车映射
            },
            "display_colors": {
                "pedestrian": [0, 255, 0],
                "vehicle": [255, 165, 0]
            },
            "counting_rules": {  # 调整计数参数
                "min_persist_sec": 0.8,      # 降低时间阈值
                "min_move_distance": 30,     # 降低位移要求
                "max_disappear_sec": 3.0,    # 延长消失时间
                "min_continuous_frames": 3   # 减少连续帧要求
            }
        }

        self.config = self._deep_merge(self.default_config, self._load_user_config(config_path))
        self.text_position = tuple(self.config["text_position"])

        # 初始化计数系统
        self.last_update = time.time()
        self.interval = 1.0
        self.history_ids = defaultdict(set)  # 全量ID存储（用于total）
        self.current_ids = defaultdict(set)  # 当前ID存储（用于current）
        self.last_detections = []

        # 新增轨迹记录
        self.track_records = defaultdict(lambda: {
            "first_seen": None,    # 首次出现时间戳
            "last_seen": None,     # 最后出现时间
            "positions": [],       # 历史位置队列（用于位移分析）
            "counted": False,      # 是否已计入total
            "continuous": 0        # 连续出现帧数
        })

        self.tracking_params = {
            "min_duration": 1.5,    # 最小存在时间(秒)
            "min_frames": 5,        # 最小连续帧数
            "min_distance": 50,     # 最小移动距离(像素)
            "cleanup_interval": 5.0 # 清理间隔(秒)
        }

    def _load_user_config(self, path):
        """安全加载用户配置"""
        try:
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
            return {}
        except:
            return {}
        
    def _deep_merge(self, base, user):
        """递归深度合并字典"""
        merged = base.copy()
        for key, value in user.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def update(self, vehicle_ids, pedestrian_ids, raw_detections):
        current_time = time.time()
        current_frame_ids = defaultdict(set)

        # 预处理：合并两轮车到行人计数
        modified_detections = []
        for obj in raw_detections:
            # 复制对象避免修改原始数据
            new_obj = obj.copy()
            if new_obj["class_name"] == "two_wheelers":
                new_obj["class_name"] = "pedestrian"
            modified_detections.append(new_obj)
        raw_detections = modified_detections

        # ========== 轨迹记录系统优化 ==========
        active_ids = set()
        
        for obj in raw_detections:
            cls_name = self.config["class_alias"].get(
                obj["class_name"], obj["class_name"]
            )
            track_id = obj["track_id"]
            
            # 过滤非目标类别（新增自行车类别处理）
            if cls_name not in ["pedestrian", "vehicle"]:
                continue

            # 更新当前帧ID记录
            current_frame_ids[cls_name].add(track_id)
            active_ids.add(track_id)

            # 轨迹记录逻辑优化
            if track_id not in self.track_records:
                self.track_records[track_id] = {
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "positions": [self._get_center(obj["bbox"])],
                    "continuous": 1,
                    "cls": cls_name
                }
            else:
                record = self.track_records[track_id]
                record["last_seen"] = current_time
                record["continuous"] += 1
                
                # 更新位置队列（优化内存管理）
                if len(record["positions"]) >= 5:
                    record["positions"] = record["positions"][1:] + [self._get_center(obj["bbox"])]
                else:
                    record["positions"].append(self._get_center(obj["bbox"]))

            # 实时计数条件检查（降低阈值）
            record = self.track_records[track_id]
            if not track_id in self.history_ids[cls_name]:
                time_ok = (current_time - record["first_seen"]) > \
                         self.config["counting_rules"]["min_persist_sec"]
                
                frame_ok = record["continuous"] >= \
                          self.config["counting_rules"]["min_continuous_frames"]
                
                move_ok = False
                if len(record["positions"]) > 1:
                    dx = record["positions"][-1][0] - record["positions"][0][0]
                    dy = record["positions"][-1][1] - record["positions"][0][1]
                    move_ok = (dx**2 + dy**2) > \
                             (self.config["counting_rules"]["min_move_distance"] ** 2)
                
                if time_ok or (frame_ok and move_ok):  # 使用OR逻辑提高灵敏度
                    self.history_ids[cls_name].add(track_id)

        # 智能清理策略（仅清理未计数的轨迹）
        expired = [
            tid for tid, rec in self.track_records.items()
            if (current_time - rec["last_seen"]) > self.config["counting_rules"]["max_disappear_sec"]
            and tid not in self.history_ids[rec["cls"]]
        ]
        for tid in expired:
            del self.track_records[tid]

        # 第二步：current计数逻辑（保持原有更新机制）
        for cls in ["pedestrian", "vehicle"]:
            # 移除消失的ID
            disappeared = self.current_ids[cls] - current_frame_ids[cls]
            self.current_ids[cls] -= disappeared
            
            # 定期刷新当前ID集合
            if current_time - self.last_update >= self.interval:
                self.current_ids[cls] = current_frame_ids[cls].copy()

        if current_time - self.last_update >= self.interval:
            self.last_update = current_time

        self.last_detections = raw_detections
        return self._format_counts()
    
    def _get_center(self, bbox):
        """计算包围盒中心点"""
        return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)

    def _format_counts(self):
        """生成统计数据"""
        return {
            "total": {
                cls: len(ids) 
                for cls, ids in self.history_ids.items()
            },
            "current": {
                cls: len(ids)
                for cls, ids in self.current_ids.items()
            }
        }

    def draw_visuals(self, frame):
        """可视化界面绘制"""
        counts = self._format_counts()
        y = self.text_position[1]
        
        # 标题显示
        cv2.putText(frame, "REALTIME TRAFFIC COUNTER", 
                   (self.text_position[0], y-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # 分类计数显示
        for cls in ["pedestrian", "vehicle"]:
            color = tuple(self.config["display_colors"][cls])
            total = counts["total"].get(cls, 0)
            current = counts["current"].get(cls, 0)
            
            text = f"{cls.upper()}: Total={total} Current={current}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # 文字背景
            cv2.rectangle(frame,
                         (self.text_position[0]-5, y-th-5),
                         (self.text_position[0]+tw+5, y+5),
                         (40,40,40), -1)
            
            # 文字内容
            cv2.putText(frame, text,
                       (self.text_position[0], y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 40

        # 绘制检测框
        for obj in self.last_detections:
            cls_name = self.config["class_alias"].get(
                obj["class_name"], obj["class_name"]
            )
            if cls_name not in self.config["display_colors"]:
                continue
                
            x1, y1, x2, y2 = map(int, obj["bbox"])
            color = tuple(self.config["display_colors"][cls_name])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 在标题下方添加参数提示
        param_text = f"Thresholds: {self.config['counting_rules']}"
        cv2.putText(frame, param_text, 
                   (self.text_position[0], self.text_position[1]+100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        return frame
    