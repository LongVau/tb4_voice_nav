from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


def test_map_files_copied_into_package():
    pgm_path = PKG_ROOT / 'maps' / 'map.pgm'
    yaml_path = PKG_ROOT / 'maps' / 'map.yaml'
    assert pgm_path.is_file()
    assert yaml_path.is_file()


def test_pixel_world_roundtrip():
    from tb4_voice_nav.map_segmenter import pixel_to_world, world_to_pixel

    wx, wy = pixel_to_world(201, 200, height=402, resolution=0.05, origin=(-10.1, -10.1, 0.0))
    assert abs(wx - (-0.025)) < 1e-6
    assert abs(wy - (-0.025)) < 1e-6
    px, py = world_to_pixel(wx, wy, height=402, resolution=0.05, origin=(-10.1, -10.1, 0.0))
    assert (px, py) == (201, 200)


def test_segment_occupancy_map_creates_at_least_3_labeled_regions(tmp_path):
    from tb4_voice_nav.map_segmenter import load_regions, segment_occupancy_map

    pgm_path = PKG_ROOT / 'maps' / 'map.pgm'
    yaml_path = PKG_ROOT / 'maps' / 'map.yaml'
    out_yaml = tmp_path / 'regions.yaml'
    out_img = tmp_path / 'map_segmented.png'

    result = segment_occupancy_map(
        pgm_path=pgm_path,
        yaml_path=yaml_path,
        output_yaml_path=out_yaml,
        output_image_path=out_img,
    )
    assert out_yaml.is_file()
    assert out_img.is_file()
    seg_img = cv2.imread(str(out_img))
    assert seg_img is not None and seg_img.shape[:2] == (402, 403)

    regions = load_regions(out_yaml)
    assert len(regions) >= 3
    codes = {r['code'] for r in regions}
    assert {'A', 'B', 'C'}.issubset(codes)

    raw_map = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    for region in regions:
        assert region['clearance_m'] >= 1.5
        px, py = region['pixel_center']
        assert raw_map[py, px] >= 250
        gx = region['goal_pose']['x']
        gy = region['goal_pose']['y']
        assert -10.1 <= gx <= 10.1
        assert -10.1 <= gy <= 10.1
        assert len(region['aliases']) >= 3
        assert result['regions_count'] == len(regions)


def test_parse_single_destination_commands():
    from tb4_voice_nav.command_parser import parse_voice_command
    from tb4_voice_nav.map_segmenter import load_regions

    regions = load_regions(PKG_ROOT / 'config' / 'regions.yaml')

    c1 = parse_voice_command('Đi đến khu vực A.', regions)
    assert c1.status == 'OK'
    assert c1.mode == 'SINGLE'
    assert [w['code'] for w in c1.waypoints] == ['A']

    c2 = parse_voice_command('Đi đến phòng thí nghiệm', regions)
    assert c2.status == 'OK'
    assert c2.mode == 'SINGLE'
    assert [w['code'] for w in c2.waypoints] == ['B']

    c3 = parse_voice_command('Di chuyển tới hành lang', regions)
    assert c3.status == 'OK'
    assert [w['code'] for w in c3.waypoints] == ['C']


def test_parse_from_to_and_multi_waypoint_commands():
    from tb4_voice_nav.command_parser import parse_voice_command, resolve_navigation_waypoints
    from tb4_voice_nav.map_segmenter import load_regions

    regions = load_regions(PKG_ROOT / 'config' / 'regions.yaml')

    c_ft = parse_voice_command('Di chuyển từ khu vực A đến khu vực B.', regions)
    assert c_ft.status == 'OK'
    assert c_ft.mode == 'FROM_TO'
    assert c_ft.origin is not None and c_ft.origin['code'] == 'A'
    assert [w['code'] for w in c_ft.waypoints] == ['A', 'B']

    # Robot currently at (0, 0) -> far from A (-3.12, 3.02), must visit A first then B
    route_far = resolve_navigation_waypoints(c_ft, robot_xy=(0.0, 0.0), at_region_radius=1.5)
    assert [w['code'] for w in route_far] == ['A', 'B']

    # Robot already inside A (-3.10, 3.00) -> skip A, go directly to B
    route_at_a = resolve_navigation_waypoints(c_ft, robot_xy=(-3.10, 3.00), at_region_radius=1.5)
    assert [w['code'] for w in route_at_a] == ['B']

    c_multi = parse_voice_command('Đi từ phòng A qua hành lang C rồi đến phòng B', regions)
    assert c_multi.status == 'OK'
    assert c_multi.mode == 'MULTI'
    assert [w['code'] for w in c_multi.waypoints] == ['A', 'C', 'B']
    route_multi = resolve_navigation_waypoints(c_multi, robot_xy=(0.0, 0.0), at_region_radius=1.5)
    assert [w['code'] for w in route_multi] == ['A', 'C', 'B']


def test_parse_unknown_region_and_empty_commands():
    from tb4_voice_nav.command_parser import parse_voice_command
    from tb4_voice_nav.map_segmenter import load_regions

    regions = load_regions(PKG_ROOT / 'config' / 'regions.yaml')

    u1 = parse_voice_command('Đi đến khu vực Z', regions)
    assert u1.status == 'UNKNOWN_REGION'
    assert len(u1.waypoints) == 0

    u2 = parse_voice_command('Di chuyển từ khu vực A đến phòng E', regions)
    assert u2.status == 'UNKNOWN_REGION'
    assert len(u2.waypoints) == 0

    u3 = parse_voice_command('Đi đến nhà bếp', regions)
    assert u3.status == 'UNKNOWN_REGION'
    assert len(u3.waypoints) == 0

    empty = parse_voice_command('   ', regions)
    assert empty.status == 'EMPTY'


def test_pulse_recorder_captures_audio_and_saves_wav(tmp_path):
    from tb4_voice_nav.pulse_recorder import load_wav, record_pulse_audio, save_wav

    audio = record_pulse_audio(duration_sec=0.3, sample_rate=16000, warmup_sec=0.1)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.shape == (4800,)
    assert np.all(audio >= -1.0) and np.all(audio <= 1.0)

    wav_file = tmp_path / 'sample.wav'
    save_wav(wav_file, audio, sample_rate=16000)
    assert wav_file.is_file()
    loaded, sr = load_wav(wav_file)
    assert sr == 16000
    assert loaded.shape == (4800,)


def test_phowhisper_transcriber_pipeline_interface():
    from tb4_voice_nav.voice_command_node import PhoWhisperTranscriber

    dummy_calls = []

    def fake_pipe(audio_input, **kwargs):
        dummy_calls.append((audio_input, kwargs))
        return {'text': ' Đi đến khu vực A. '}

    transcriber = PhoWhisperTranscriber(
        model_name='vinai/PhoWhisper-small',
        pipeline_fn=fake_pipe,
    )
    sample = np.zeros(16000, dtype=np.float32)
    text = transcriber.transcribe(sample, sample_rate=16000)
    assert text == 'Đi đến khu vực A.'
    assert len(dummy_calls) == 1


def test_region_visualizer_builds_markers_for_all_regions():
    from tb4_voice_nav.map_segmenter import load_regions
    from tb4_voice_nav.region_visualizer_node import build_region_marker_array

    regions = load_regions(PKG_ROOT / 'config' / 'regions.yaml')
    marker_array = build_region_marker_array(regions)
    # Expect cylinder + text label per region (2 * len(regions))
    assert len(marker_array.markers) == 2 * len(regions)
    texts = [m.text for m in marker_array.markers if m.text]
    assert any('Khu vực A' in t and 'Phòng học' in t for t in texts)
    assert any('Khu vực B' in t and 'Phòng thí nghiệm' in t for t in texts)


def test_voice_navigator_goal_pose_and_planned_path():
    from tb4_voice_nav.map_segmenter import load_regions
    from tb4_voice_nav.voice_navigator_node import (
        build_navigate_to_pose_goal,
        build_planned_path_msg,
    )

    regions = load_regions(PKG_ROOT / 'config' / 'regions.yaml')
    goal_msg = build_navigate_to_pose_goal(regions[0], frame_id='map')
    assert goal_msg.pose.header.frame_id == 'map'
    assert abs(goal_msg.pose.pose.position.x - regions[0]['goal_pose']['x']) < 1e-6
    assert abs(goal_msg.pose.pose.position.y - regions[0]['goal_pose']['y']) < 1e-6
    assert abs(goal_msg.pose.pose.orientation.w - 1.0) < 1e-6

    path_msg = build_planned_path_msg(regions[:3], start_xy=(0.0, 0.0), frame_id='map')
    assert path_msg.header.frame_id == 'map'
    assert len(path_msg.poses) == 4


def test_launch_and_config_files_exist():
    assert (PKG_ROOT / 'config' / 'nav2.yaml').is_file()
    assert (PKG_ROOT / 'config' / 'regions.yaml').is_file()
    assert (PKG_ROOT / 'launch' / 'voice_nav_sim.launch.py').is_file()
    assert (PKG_ROOT / 'launch' / 'voice_nav_nodes.launch.py').is_file()


def test_wsl_tf_stability_configs_and_fastdds_udp():
    import yaml
    from tb4_voice_nav.voice_navigator_node import (
        build_navigate_to_pose_goal,
        ensure_fastdds_udp_env,
    )

    fastdds_path = PKG_ROOT / 'config' / 'fastdds_udp.xml'
    loc_path = PKG_ROOT / 'config' / 'localization.yaml'
    nav2_path = PKG_ROOT / 'config' / 'nav2.yaml'

    assert fastdds_path.is_file()
    xml_text = fastdds_path.read_text(encoding='utf-8')
    assert '<type>UDPv4</type>' in xml_text
    assert '<useBuiltinTransports>false</useBuiltinTransports>' in xml_text

    assert loc_path.is_file()
    loc_cfg = yaml.safe_load(loc_path.read_text(encoding='utf-8'))
    amcl_params = loc_cfg['amcl']['ros__parameters']
    assert float(amcl_params['transform_tolerance']) >= 1.5
    assert int(amcl_params['max_particles']) <= 1000
    assert int(amcl_params['max_beams']) <= 40

    nav_cfg = yaml.safe_load(nav2_path.read_text(encoding='utf-8'))
    ctrl_params = nav_cfg['controller_server']['ros__parameters']
    assert float(ctrl_params['FollowPath']['transform_tolerance']) >= 1.0

    local_cm = nav_cfg['local_costmap']['local_costmap']['ros__parameters']
    assert float(local_cm['transform_tolerance']) >= 1.0
    assert float(local_cm['update_frequency']) <= 3.0
    assert local_cm['voxel_layer']['publish_voxel_map'] is False

    global_cm = nav_cfg['global_costmap']['global_costmap']['ros__parameters']
    assert float(global_cm['transform_tolerance']) >= 1.0

    # Ensure NavigateToPose goal uses Time(0) (sec=0, nanosec=0) so tf_help uses latest transform
    dummy_reg = {'goal_pose': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}}
    goal = build_navigate_to_pose_goal(dummy_reg, frame_id='map')
    assert goal.pose.header.stamp.sec == 0
    assert goal.pose.header.stamp.nanosec == 0

    profile_used = ensure_fastdds_udp_env()
    assert Path(profile_used).is_file()




