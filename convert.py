import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image

# 配置参数
VISDRONE_ROOT = Path(r"E:\Programming\python\Object Detection\YOLOv8\data\VisDrone2019")
YOLO_DATASET = Path("data/YOLO_Dataset")  # 最终输出目录
CATEGORY_MAP = {
    1: 0,   # pedestrian
    2: 0,   # people
    3: 1,   # bicycle
    4: 2,   # car
    5: 3,   # van
    6: 4,   # truck
    7: 5,   # tricycle
    8: 6,   # awning-tricycle
    9: 7,   # bus
    10: 8,  # motor
    11: -1  # ignore
}

# 核心转换函数
def process_split(split_type: str):
    """处理单个数据分割（train/val/test）"""
    # 路径配置
    src_ann_dir = VISDRONE_ROOT / f"VisDrone2019-VID-{split_type}" / "annotations"
    src_img_root = VISDRONE_ROOT / f"VisDrone2019-VID-{split_type}" / "sequences"
    
    # 输出目录
    img_output_dir = YOLO_DATASET / "images" / split_type
    label_output_dir = YOLO_DATASET / "labels" / split_type
    img_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历所有标注文件
    for ann_file in src_ann_dir.glob("*.txt"):
        seq_name = ann_file.stem
        print(f"\nProcessing {split_type} sequence: {seq_name}")
        
        # 处理单个视频序列
        process_sequence(
            ann_file=ann_file,
            src_img_dir=src_img_root / seq_name,
            img_output_dir=img_output_dir,
            label_output_dir=label_output_dir,
            seq_name=seq_name
        )

def process_sequence(ann_file: Path, src_img_dir: Path, 
                    img_output_dir: Path, label_output_dir: Path,
                    seq_name: str):
    """处理单个视频序列"""
    # 校验图像目录
    if not src_img_dir.exists():
        print(f"Image directory missing: {src_img_dir}")
        return

    # 预加载图像尺寸信息
    img_sizes = {}
    for img_path in src_img_dir.glob("*.jpg"):
        with Image.open(img_path) as img:
            img_sizes[img_path.name] = img.size  # (width, height)

    # 处理标注文件
    with open(ann_file, 'r') as f:
        frame_annotations = defaultdict(list)
        
        for line_num, line in enumerate(f, 1):
            if line.startswith('%') or not line.strip():
                continue

            data = line.strip().split(',')
            if len(data) < 10:
                continue

            try:
                # 解析关键字段
                frame_idx = int(data[0])
                orig_img_name = f"{frame_idx:07d}.jpg"
                new_img_name = f"{seq_name}_{orig_img_name}"
                
                # 检查对应图像
                if orig_img_name not in img_sizes:
                    continue

                # 坐标信息
                img_w, img_h = img_sizes[orig_img_name]
                bbox_left = int(data[2])
                bbox_top = int(data[3])
                bbox_w = int(data[4])
                bbox_h = int(data[5])
                
                # 过滤条件
                category_id = int(data[7])
                truncation = int(data[8])
                occlusion = int(data[9])
                if truncation > 0 or occlusion > 1:
                    continue

                # 类别映射
                yolo_class = CATEGORY_MAP.get(category_id, -1)
                if yolo_class == -1:
                    continue

                # 坐标转换
                x_center = (bbox_left + bbox_w/2) / img_w
                y_center = (bbox_top + bbox_h/2) / img_h
                width = bbox_w / img_w
                height = bbox_h / img_h

                # 边界检查
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    continue

                # 生成YOLO行
                yolo_line = f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                frame_annotations[new_img_name].append(yolo_line)

            except Exception as e:
                print(f"Error line {line_num}: {str(e)}")
                continue

        # 保存结果
        save_results(
            frame_annotations=frame_annotations,
            src_img_dir=src_img_dir,
            img_output_dir=img_output_dir,
            label_output_dir=label_output_dir,
            seq_name=seq_name
        )

def save_results(frame_annotations: dict, src_img_dir: Path, 
                img_output_dir: Path, label_output_dir: Path,
                seq_name: str):
    """保存转换结果"""
    # 复制图像并生成标签
    copied_images = set()
    
    for img_name in frame_annotations.keys():
        # 源图像路径
        orig_frame_id = img_name.split('_')[-1]
        src_img_path = src_img_dir / orig_frame_id
        
        # 目标路径
        dst_img_path = img_output_dir / img_name
        label_path = label_output_dir / f"{Path(img_name).stem}.txt"

        # 复制图像（避免重复复制）
        if not dst_img_path.exists():
            shutil.copy(src_img_path, dst_img_path)
            copied_images.add(img_name)

        # 写入标签
        with open(label_path, 'w') as f:
            f.writelines(frame_annotations[img_name])

    print(f"Generated {len(frame_annotations)} labels | Copied {len(copied_images)} images")

# 主程序
def main():
    # 清理旧数据
    shutil.rmtree(YOLO_DATASET, ignore_errors=True)
    
    # 处理所有数据分割
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*30} Processing {split} set {'='*30}")
        process_split(split)
    
    print("\nConversion completed! Final dataset structure:")
    print(f"""
    {YOLO_DATASET}
    ├── images
    │   ├── train
    │   ├── val
    │   └── test
    └── labels
        ├── train
        ├── val
        └── test
    """)

if __name__ == "__main__":
    main()
