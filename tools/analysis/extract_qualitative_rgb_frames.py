import argparse
import sys
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis.render_presentation_video import load_time_index_map


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / 'paper_assets' / 'manuscript_latex' /
    'resfes2026_witidar' / 'figures' / 'qualitative_rgb')


class FrameSpec(NamedTuple):
    sample_name: str
    video_id: str
    frame_id: int


def build_default_frame_specs():
    return [
        FrameSpec('S11_06_319', 'S11_06', 319),
        FrameSpec('S52_40_322', 'S52_40', 322),
        FrameSpec('S23_12_337', 'S23_12', 337),
    ]


def resolve_exact_frame_index(frame_id, time_index_map):
    if frame_id not in time_index_map:
        raise KeyError(f'Frame ID {frame_id} is absent from time_list.txt.')
    return int(time_index_map[frame_id])


def extract_exact_video_frame(video_path, frame_index, output_path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            'OpenCV is required. Install opencv-python in the inference environment.') from exc

    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(f'Missing source video: {video_path}')

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open source video: {video_path}')
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        resolved_index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
    finally:
        capture.release()

    if not ok or frame is None:
        raise RuntimeError(
            f'Failed to decode frame {frame_index} from {video_path}.')
    if resolved_index != int(frame_index):
        raise RuntimeError(
            f'Video seek mismatch for {video_path}: requested {frame_index}, '
            f'decoded {resolved_index}.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f'Failed to write RGB frame: {output_path}')
    return output_path


def extract_default_frames(source_video_root, output_dir=DEFAULT_OUTPUT_DIR):
    source_video_root = Path(source_video_root)
    output_dir = Path(output_dir)
    outputs = []
    for spec in build_default_frame_specs():
        video_dir = source_video_root / spec.video_id
        video_path = video_dir / 'output.mkv'
        time_list_path = video_dir / 'time_list.txt'
        if not time_list_path.exists():
            raise FileNotFoundError(f'Missing time list for {spec.video_id}: {time_list_path}')
        time_index_map = load_time_index_map(time_list_path, video_id=spec.video_id)
        video_index = resolve_exact_frame_index(spec.frame_id, time_index_map)
        output_path = output_dir / f'{spec.sample_name}.png'
        extract_exact_video_frame(video_path, video_index, output_path)
        print(
            f'{spec.sample_name}: video={video_path} '
            f'frame_id={spec.frame_id} video_index={video_index} output={output_path}')
        outputs.append(output_path)
    return outputs


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Extract the three synchronized RGB frames used by manuscript Figure 8.')
    parser.add_argument(
        '--source-video-root',
        required=True,
        help='Directory containing S11_06, S52_40, and S23_12 video folders.')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main():
    args = build_arg_parser().parse_args()
    outputs = extract_default_frames(args.source_video_root, args.output_dir)
    print('Extracted RGB frames:')
    for output in outputs:
        print(output)


if __name__ == '__main__':
    main()
