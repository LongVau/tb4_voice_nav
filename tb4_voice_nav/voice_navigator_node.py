"""Voice navigation node converting Vietnamese commands into Nav2 NavigateToPose goals."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

from action_msgs.msg import GoalStatus
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path as NavPath
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from tb4_voice_nav.command_parser import (
    parse_voice_command,
    resolve_navigation_waypoints,
)
from tb4_voice_nav.map_segmenter import load_regions


def _default_regions_yaml() -> str:
    try:
        share = get_package_share_directory('tb4_voice_nav')
        candidate = os.path.join(share, 'config', 'regions.yaml')
        if os.path.isfile(candidate):
            return candidate
    except PackageNotFoundError:
        pass
    return str(Path(__file__).resolve().parents[1] / 'config' / 'regions.yaml')


def _default_fastdds_xml() -> str:
    try:
        share = get_package_share_directory('tb4_voice_nav')
        candidate = os.path.join(share, 'config', 'fastdds_udp.xml')
        if os.path.isfile(candidate):
            return candidate
    except PackageNotFoundError:
        pass
    return str(Path(__file__).resolve().parents[1] / 'config' / 'fastdds_udp.xml')


def ensure_fastdds_udp_env() -> str:
    """Ensure FASTRTPS_DEFAULT_PROFILES_FILE points to UDPv4 profile on WSL2 before rclpy.init."""
    xml_path = _default_fastdds_xml()
    if os.path.isfile(xml_path):
        os.environ.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', xml_path)
    return os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE', xml_path)


def build_navigate_to_pose_goal(
    region: dict,
    frame_id: str = 'map',
) -> NavigateToPose.Goal:
    """Build a Nav2 NavigateToPose.Goal message from a region entry."""
    gp = region['goal_pose']
    yaw = float(gp.get('yaw', 0.0))
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = frame_id
    goal.pose.pose.position.x = float(gp['x'])
    goal.pose.pose.position.y = float(gp['y'])
    goal.pose.pose.position.z = 0.0
    goal.pose.pose.orientation.x = 0.0
    goal.pose.pose.orientation.y = 0.0
    goal.pose.pose.orientation.z = qz
    goal.pose.pose.orientation.w = qw
    return goal


def build_planned_path_msg(
    waypoints: Sequence[dict],
    start_xy: tuple[float, float] | None = None,
    frame_id: str = 'map',
) -> NavPath:
    """Create a nav_msgs/Path connecting the robot's start position and target region poses."""
    path_msg = NavPath()
    path_msg.header.frame_id = frame_id

    if start_xy is not None:
        start_pose = PoseStamped()
        start_pose.header.frame_id = frame_id
        start_pose.pose.position.x = float(start_xy[0])
        start_pose.pose.position.y = float(start_xy[1])
        start_pose.pose.orientation.w = 1.0
        path_msg.poses.append(start_pose)

    for reg in waypoints:
        gp = reg['goal_pose']
        yaw = float(gp.get('yaw', 0.0))
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose.position.x = float(gp['x'])
        ps.pose.position.y = float(gp['y'])
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        path_msg.poses.append(ps)

    return path_msg


class VoiceNavigatorNode(Node):
    """Subscribe to /voice_command_text, parse regions, and send sequential Nav2 goals."""

    def __init__(self) -> None:
        super().__init__(
            'voice_navigator_node',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
        )
        self.declare_parameter('regions_yaml', _default_regions_yaml())
        self.declare_parameter('at_region_radius', 1.5)
        self.declare_parameter('auto_initial_pose', True)
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)

        regions_path = str(self.get_parameter('regions_yaml').value)
        self.at_region_radius = float(self.get_parameter('at_region_radius').value)
        self.auto_initial_pose = bool(self.get_parameter('auto_initial_pose').value)
        self.regions = load_regions(regions_path)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.status_pub = self.create_publisher(String, '/voice_nav_status', 10)
        self.path_pub = self.create_publisher(NavPath, '/planned_region_path', 10)
        self.init_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10
        )

        self.cmd_sub = self.create_subscription(
            String,
            '/voice_command_text',
            self._on_voice_command,
            10,
        )

        self._waypoint_queue: list[dict] = []
        self._active_goal_handle = None
        self._initial_pose_sent_count = 0

        if self.auto_initial_pose:
            self._init_pose_timer = self.create_timer(2.0, self._publish_initial_pose_once)

        valid_str = ', '.join(
            f"{r['code']}:{r['semantic_name']} ({r['goal_pose']['x']:.1f},{r['goal_pose']['y']:.1f})"
            for r in self.regions
        )
        self.get_logger().info(
            f'VoiceNavigatorNode ready with {len(self.regions)} regions: {valid_str}'
        )

    def _publish_initial_pose_once(self) -> None:
        if self._initial_pose_sent_count >= 3:
            self._init_pose_timer.cancel()
            return
        if self._get_robot_xy() is not None:
            self._initial_pose_sent_count = 3
            self._init_pose_timer.cancel()
            return

        ix = float(self.get_parameter('initial_x').value)
        iy = float(self.get_parameter('initial_y').value)
        iyaw = float(self.get_parameter('initial_yaw').value)

        msg = PoseWithCovarianceStamped()
        # Use Time(0) (sec=0, nanosec=0) so AMCL transforms using latest odom->base_link without extrapolation
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = ix
        msg.pose.pose.position.y = iy
        msg.pose.pose.orientation.z = math.sin(iyaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(iyaw / 2.0)
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.0685
        self.init_pose_pub.publish(msg)
        self._initial_pose_sent_count += 1
        self.get_logger().info(
            f'Published initial pose ({ix:.2f}, {iy:.2f}, yaw={iyaw:.2f}) to /initialpose'
        )

    def _get_robot_xy(self) -> tuple[float, float] | None:
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return (
                float(tf.transform.translation.x),
                float(tf.transform.translation.y),
            )
        except TransformException:
            return None

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def _on_voice_command(self, msg: String) -> None:
        raw_text = msg.data
        self.get_logger().info(f"Nhận câu lệnh giọng nói: '{raw_text}'")
        parsed = parse_voice_command(raw_text, self.regions)

        if parsed.status != 'OK':
            self.get_logger().warn(f'[{parsed.status}] {parsed.message}')
            self._publish_status(f'{parsed.status}: {parsed.message}')
            return

        robot_xy = self._get_robot_xy()
        waypoints = resolve_navigation_waypoints(
            parsed,
            robot_xy=robot_xy,
            at_region_radius=self.at_region_radius,
        )

        if parsed.origin is not None and len(parsed.waypoints) >= 2:
            if len(waypoints) < len(parsed.waypoints):
                self.get_logger().info(
                    f"Robot đang ở tại {parsed.origin['name']} -> đi thẳng tới đích tiếp theo."
                )
            else:
                self.get_logger().info(
                    f"Robot chưa ở {parsed.origin['name']} (robot_xy={robot_xy}) "
                    f"-> điều hướng qua {parsed.origin['name']} trước khi đến đích."
                )

        path_msg = build_planned_path_msg(waypoints, start_xy=robot_xy, frame_id='map')
        now = self.get_clock().now().to_msg()
        path_msg.header.stamp = now
        for ps in path_msg.poses:
            ps.header.stamp = now
        self.path_pub.publish(path_msg)

        route_str = ' -> '.join(
            f"{w['name']} ({w['semantic_name']} @ {w['goal_pose']['x']:.2f}, {w['goal_pose']['y']:.2f})"
            for w in waypoints
        )
        self.get_logger().info(f'Lộ trình thực thi: {route_str}')
        self._publish_status(f'EXECUTING: {route_str}')

        self._waypoint_queue = list(waypoints)
        self._send_next_waypoint()

    def _send_next_waypoint(self) -> None:
        if not self._waypoint_queue:
            self.get_logger().info('Đã hoàn thành toàn bộ lộ trình điều hướng bằng giọng nói!')
            self._publish_status('COMPLETED')
            return

        target = self._waypoint_queue.pop(0)
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                'Action server /navigate_to_pose chưa sẵn sàng! Kiểm tra Nav2.'
            )
            self._publish_status('ERROR: NavigateToPose action server unavailable')
            self._waypoint_queue.clear()
            return

        # Keep goal_msg.pose.header.stamp at Time(0) so Nav2 tf_help uses latest map->odom transform
        goal_msg = build_navigate_to_pose_goal(target, frame_id='map')

        gp = target['goal_pose']
        self.get_logger().info(
            f">>> Đang gửi NavigateToPose tới {target['name']} ({target['semantic_name']}): "
            f"x={gp['x']:.2f}, y={gp['y']:.2f}, yaw={gp['yaw']:.2f}"
        )
        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            lambda fut, reg=target: self._on_goal_response(fut, reg)
        )

    def _on_goal_response(self, future, region: dict) -> None:
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error(f"Mục tiêu {region['name']} bị Nav2 từ chối!")
            self._publish_status(f"REJECTED: {region['name']}")
            self._waypoint_queue.clear()
            return

        self._active_goal_handle = goal_handle
        self.get_logger().info(f"Nav2 đã chấp nhận mục tiêu {region['name']}. Đang di chuyển...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda fut, reg=region: self._on_goal_result(fut, reg)
        )

    def _on_goal_result(self, future, region: dict) -> None:
        result = future.result()
        status = result.status if result is not None else GoalStatus.STATUS_UNKNOWN
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                f"Đã đến {region['name']} ({region['semantic_name']}) thành công!"
            )
            self._publish_status(f"REACHED: {region['name']}")
            self._send_next_waypoint()
        else:
            self.get_logger().warn(
                f"Điều hướng tới {region['name']} kết thúc với mã trạng thái={status}."
            )
            self._publish_status(f"FAILED: {region['name']} (status={status})")
            self._waypoint_queue.clear()


def main(args: Sequence[str] | None = None) -> None:
    ensure_fastdds_udp_env()
    rclpy.init(args=args)
    node = VoiceNavigatorNode()
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

