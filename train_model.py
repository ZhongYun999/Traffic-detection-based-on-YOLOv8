import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
from ultralytics.utils import SETTINGS, emojis

# 数据验证模块
def validate_and_create_dataset():
    """目录创建与验证"""
    with open('data/visdrone.yaml', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
        custom_path = Path(yaml_config['path']).resolve()
    
    datasets_path = custom_path
    datasets_path.mkdir(parents=True, exist_ok=True)
    print(f"数据集根目录已确认: {datasets_path}")

    # 子目录结构
    required_dirs = [
        "images/train", "labels/train",
        "images/val", "labels/val",
        "images/test", "labels/test"  # 可选
    ]
    
    # 自动创建目录
    created = []
    for rel_path in required_dirs:
        full_path = datasets_path / rel_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            created.append(str(full_path))
    
    if created:
        print(emojis(f"自动创建以下目录:\n" + "\n".join(f" - {p}" for p in created)))
    
    # 二次验证
    missing = [p for p in required_dirs if not (datasets_path / p).exists()]
    if missing:
        raise FileNotFoundError(emojis(f"目录创建失败: {missing}"))

# 训练执行模块
def main():
    # 数据验证与目录创建
    validate_and_create_dataset()
    
    # 模型初始化
    model = YOLO('yolov8s.yaml').load('yolov8s.pt').to('cuda:0')
    
    # 训练配置
    model.train(
        data='data/visdrone.yaml',
        
        # ===== 核心训练参数优化 =====
        epochs=50,             # 大幅增加训练轮次
        batch=16,              # 大幅增加批次大小
        imgsz=640,             # 输入分辨率
        workers=4,             # 线程
        
        # ===== 新增关键优化参数 =====
        warmup_epochs=3.0,      # 学习率预热（防止初期震荡）
        warmup_momentum=0.8,    # 预热期动量
        hsv_s=0.7,              # 饱和度增强（补充hsv_h）
        hsv_v=0.4,              # 明度增强
        mixup=0.15,             # MixUp数据增强
        copy_paste=0.3,         # 复制粘贴增强
        erasing=0.4,            # 随机擦除增强
        degrees=10.0,           # 随机旋转角度
        translate=0.1,          # 图像平移范围
        perspective=0.0001,     # 透视变换
        dfl=1.5,                # 分布焦点损失权重
        dropout=0.2,            # 防止过拟合
        
        # ===== 保留原有良好参数 =====
        device=0,
        amp=True,
        half=True,
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.05,
        cos_lr=True,
        hsv_h=0.015,
        fliplr=0.5,
        box=7.5,
        cls=0.5,
        label_smoothing=0.1,
        mosaic=0.0,
        close_mosaic=10,
        overlap_mask=False,
        single_cls=False,
        verbose=True,
        deterministic=False,
        exist_ok=True,
        
        # ===== 新增训练管理参数 =====
        cache='ram',            # 内存缓存加速数据加载
        save_period=10,         # 每10轮保存一次检查点
        patience=25,            # 早停机制耐心值
        project='runs/train',   # 项目目录
        name='visdrone_v8s',    # 实验名称
        pretrained=True,        # 确保使用预训练权重
        resume=False,           # 明确不从上次中断处恢复
    )

if __name__ == "__main__":
    main()
