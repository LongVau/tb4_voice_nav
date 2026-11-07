# `tb4_voice_nav` – Phân vùng bản đồ và điều hướng robot bằng giọng nói (Bài thực hành 7)

Package ROS 2 Humble độc lập thực hiện phân vùng bản đồ 2D Occupancy Grid (`maze`), gắn nhãn ngữ nghĩa các khu vực, nhận dạng câu lệnh giọng nói tiếng Việt qua microphone bằng **VinAI PhoWhisper** (`vinai/PhoWhisper-small`), và điều hướng tự động TurtleBot 4 trên Nav2.

## 1. Các khu vực đã phân vùng (`config/regions.yaml` & `maps/map_segmented.png`)

| Mã | Tên khu vực | Nhãn ngữ nghĩa | Tọa độ đích `(x, y, yaw)` | Khoảng cách an toàn tới tường |
|---|---|---|---|---|
| `A` | Khu vực A | Phòng học | `(-3.12, 3.02, 0.0)` | `2.09 m` |
| `B` | Khu vực B | Phòng thí nghiệm | `(-6.03, -3.83, 0.0)` | `2.00 m` |
| `C` | Khu vực C | Hành lang | `(-7.58, 7.53, 0.0)` | `2.45 m` |
| `D` | Khu vực D | Văn phòng | `(6.22, -2.08, 0.0)` | `2.00 m` |

## 2. Hướng dẫn Build & Chạy

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select tb4_voice_nav
source install/setup.bash
```

### Bước 1: Chạy lại phân vùng bản đồ (tùy chọn – file đã được sinh sẵn)
```bash
ros2 run tb4_voice_nav map_segmenter
```

### Bước 2: Khởi chạy Gazebo (`maze`), AMCL, Nav2, RViz & Region Visualizer (Terminal 1)
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch tb4_voice_nav voice_nav_sim.launch.py
```
*(Trên RViz có thể thêm display `MarkerArray` với topic `/region_markers` và `Path` với topic `/planned_region_path` để xem trực quan 4 vùng và đường đi dự kiến).*

### Bước 3: Khởi chạy Node điều hướng giọng nói (Terminal 2)
```bash
source ~/ros2_ws/install/setup.bash
ros2 run tb4_voice_nav voice_navigator_node
```

### Bước 4: Khởi chạy Node thu âm Microphone + PhoWhisper ASR (Terminal 3)
```bash
source ~/ros2_ws/install/setup.bash
ros2 run tb4_voice_nav voice_command_node
```
- Nhấn **ENTER** để thu âm giọng nói trực tiếp từ microphone (`4.5s`, qua WSLg PulseAudio `16 kHz` mono) và nhận dạng bằng `vinai/PhoWhisper-small`.
- Các câu lệnh mẫu hỗ trợ:
  - *"Đi đến khu vực A"*
  - *"Đi đến phòng thí nghiệm"*
  - *"Di chuyển từ khu vực A đến khu vực B"* (nếu robot chưa ở A sẽ tự động đi qua A trước khi đến B)
  - *"Đi từ phòng A qua hành lang C rồi đến phòng B"* (đa điểm đến)
  - *"Đi đến khu vực Z"* / *"Đi đến nhà bếp"* (báo lỗi khu vực không tồn tại)

