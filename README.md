# YOLOv3 Object Detection and Tracking in Video

A comprehensive implementation of YOLOv3 (You Only Look Once v3) object detection and tracking system using PyTorch. This project provides real-time object detection capabilities for both images and video streams, with support for 80 COCO classes and custom model configurations.

![Detection Example](https://i.imgur.com/m2jwnen.png)

## 🚀 Features

- **Real-time Object Detection**: Detect objects in images and video streams with high accuracy
- **Video Processing**: Support for video files and webcam input
- **GPU Acceleration**: CUDA support for faster inference
- **Multiple Input Formats**: Process single images, image directories, or video files
- **Configurable Parameters**: Adjustable confidence thresholds, NMS settings, and input resolution
- **COCO Dataset Support**: Pre-trained on 80 COCO classes
- **Custom Model Support**: Load custom YOLO configurations and weights
- **Batch Processing**: Process multiple images simultaneously

## 📋 Requirements

- Python 3.6+
- PyTorch 1.0+
- OpenCV 4.0+
- NumPy
- CUDA (optional, for GPU acceleration)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Object-detection-and-tracking-in-video-using-YOLOv3.git
   cd Object-detection-and-tracking-in-video-using-YOLOv3
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision
   pip install opencv-python
   pip install numpy
   ```

3. **Download YOLOv3 weights**:
   ```bash
   wget https://pjreddie.com/media/files/yolov3.weights
   ```

## 🎯 Quick Start

### Image Detection

Detect objects in a single image:
```bash
python detect.py --images path/to/image.jpg --weights yolov3.weights
```

Process all images in a directory:
```bash
python detect.py --images imgs/ --weights yolov3.weights
```

### Video Detection

Process a video file:
```bash
python video.py --video path/to/video.mp4 --weights yolov3.weights
```

Use webcam:
```bash
python video.py --video 0 --weights yolov3.weights
```

## 📖 Usage Examples

### Basic Image Detection
```bash
# Detect objects in a single image with default settings
python detect.py --images dog-cycle-car.png --weights yolov3.weights

# Custom confidence threshold and NMS
python detect.py --images imgs/ --confidence 0.7 --nms_thresh 0.3 --weights yolov3.weights

# Higher resolution for better accuracy
python detect.py --images imgs/ --reso 608 --weights yolov3.weights
```

### Video Processing
```bash
# Process video file
python video.py --video sample_video.mp4 --weights yolov3.weights

# Real-time webcam detection
python video.py --video 0 --weights yolov3.weights

# Custom settings for video
python video.py --video sample_video.mp4 --confidence 0.6 --reso 512 --weights yolov3.weights
```

## ⚙️ Configuration Options

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--images` | Image file or directory | `imgs` |
| `--video` | Video file or webcam index | `video.avi` |
| `--weights` | Path to weights file | `yolov3.weights` |
| `--cfg` | Path to config file | `cfg/yolov3.cfg` |
| `--confidence` | Object confidence threshold | `0.5` |
| `--nms_thresh` | NMS threshold | `0.4` |
| `--reso` | Input resolution | `416` |
| `--bs` | Batch size | `1` |
| `--det` | Detection output directory | `det` |

### Performance Tuning

- **Higher Resolution**: Increase `--reso` for better accuracy (416, 512, 608)
- **Confidence Threshold**: Adjust `--confidence` to filter detections (0.1-1.0)
- **NMS Threshold**: Modify `--nms_thresh` for overlap handling (0.1-0.9)
- **Batch Size**: Increase `--bs` for faster processing of multiple images

## 📁 Project Structure

```
Object-detection-and-tracking-in-video-using-YOLOv3/
├── detect.py              # Image detection script
├── video.py               # Video detection script
├── darknet.py             # YOLOv3 network architecture
├── util.py                # Utility functions
├── cfg/
│   └── yolov3.cfg         # YOLOv3 configuration file
├── data/
│   ├── coco.names         # COCO class names
│   └── voc.names          # VOC class names
├── imgs/                  # Sample images
├── pallete                # Color palette for visualization
└── yolov3.weights         # Pre-trained weights (download separately)
```

## 🔧 Customization

### Using Custom Models

1. **Custom Configuration**:
   ```bash
   python detect.py --cfg path/to/custom.cfg --weights path/to/custom.weights
   ```

2. **Custom Class Names**:
   - Modify `data/coco.names` or create your own class file
   - Update the class loading in the scripts

### Training Your Own Model

This repository focuses on inference. For training:
- Use the original Darknet framework
- Or refer to PyTorch YOLO training implementations
- Convert trained weights to PyTorch format

## 📊 Performance

- **Speed**: ~30 FPS on GPU (GTX 1080 Ti)
- **Accuracy**: mAP@0.5 = 57.9% on COCO dataset
- **Memory**: ~2GB GPU memory for 416x416 input
- **CPU**: ~3-5 FPS on modern CPU

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original YOLOv3 paper: [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- Darknet framework by Joseph Redmon
- PyTorch implementation based on tutorial series

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in `docs.md`
- Refer to the original tutorial series

---

**Note**: This implementation is for educational and research purposes. For production use, consider using more recent YOLO versions (YOLOv5, YOLOv8) or other state-of-the-art object detection models.

