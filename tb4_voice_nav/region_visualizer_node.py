"""RViz MarkerArray publisher for segmented map regions in tb4_voice_nav."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from visualization_msgs.msg import Marker, MarkerArray

from tb4_voice_nav.map_segmenter import load_regions
from tb4_voice_nav.voice_navigator_node import ensure_fastdds_udp_env


def _default_regions_yaml() -> str:
    try:
        share = get_package_share_directory('tb4_voice_nav')
        candidate = os.path.join(share, 'config', 'regions.yaml')
        if os.path.isfile(candidate):
            return candidate
    except PackageNotFoundError:
        pass
    return str(Path(__file__).resolve().parents[1] / 'config' / 'regions.yaml')


def build_region_marker_array(
    regions: Sequence[dict],
    frame_id: str = 'map',
) -> MarkerArray:
    """Create cylinder + 3D label markers for every segmented region."""
    marker_array = MarkerArray()
    for idx, reg in enumerate(regions):
        gx = float(reg['goal_pose']['x'])
        gy = float(reg['goal_pose']['y'])
        r, g, b = [float(c) / 255.0 for c in reg.get('color_rgb', [66, 135, 245])]

        cyl = Marker()
        cyl.header.frame_id = frame_id
        cyl.ns = 'region_goals'
        cyl.id = idx * 2 + 1
        cyl.type = Marker.CYLINDER
        cyl.action = Marker.ADD
        cyl.pose.position.x = gx
        cyl.pose.position.y = gy
        cyl.pose.position.z = 0.15
        cyl.pose.orientation.w = 1.0
        cyl.scale.x = 0.55
        cyl.scale.y = 0.55
        cyl.scale.z = 0.30
        cyl.color.r = r
        cyl.color.g = g
        cyl.color.b = b
        cyl.color.a = 0.75
        marker_array.markers.append(cyl)

        txt = Marker()
        txt.header.frame_id = frame_id
        txt.ns = 'region_labels'
        txt.id = idx * 2 + 2
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = gx
        txt.pose.position.y = gy
        txt.pose.position.z = 0.65
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.42
        txt.color.r = 1.0
        txt.color.g = 1.0
        txt.color.b = 1.0
        txt.color.a = 1.0
        txt.text = f"{reg['name']} ({reg['semantic_name']})\n[{gx:.1f}, {gy:.1f}]"
        marker_array.markers.append(txt)

    return marker_array


class RegionVisualizerNode(Node):
    """Periodically publish region MarkerArray on /region_markers for RViz."""

    def __init__(self) -> None:
        super().__init__(
            'region_visualizer_node',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
        )
        self.declare_parameter('regions_yaml', _default_regions_yaml())
        self.declare_parameter('frame_id', 'map')

        regions_path = str(self.get_parameter('regions_yaml').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.regions = load_regions(regions_path)

        self.marker_pub = self.create_publisher(MarkerArray, '/region_markers', 10)
        self.timer = self.create_timer(1.0, self._publish_markers)
        self.get_logger().info(
            f'RegionVisualizerNode loaded {len(self.regions)} regions from {regions_path}'
        )
        self._publish_markers()

    def _publish_markers(self) -> None:
        arr = build_region_marker_array(self.regions, frame_id=self.frame_id)
        now = self.get_clock().now().to_msg()
        for m in arr.markers:
            m.header.stamp = now
        self.marker_pub.publish(arr)


def main(args: Sequence[str] | None = None) -> None:
    ensure_fastdds_udp_env()
    rclpy.init(args=args)
    node = RegionVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

