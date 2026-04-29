import importlib.util
import shutil
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'render_presentation_video.py'


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestRenderPresentationVideo(unittest.TestCase):
    def setUp(self):
        self.work_dir = ROOT / 'tests' / f'tmp_render_video_{uuid.uuid4().hex}'
        self.data_root = self.work_dir / 'wifipose'
        for split in ('train_data', 'test_data'):
            split_root = self.data_root / split
            (split_root / 'csi').mkdir(parents=True, exist_ok=True)
            (split_root / 'keypoint').mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def _write_split(self, split, names):
        split_root = self.data_root / split
        list_path = split_root / f'{split}_list.txt'
        list_path.write_text('\n'.join(names) + '\n', encoding='utf-8')
        for name in names:
            (split_root / 'csi' / f'{name}.mat').write_bytes(b'csi')
            (split_root / 'keypoint' / f'{name}.npy').write_bytes(b'keypoint')

    def test_script_exists(self):
        self.assertTrue(SCRIPT_PATH.exists(), f'Missing script: {SCRIPT_PATH}')

    def test_parse_sample_name_extracts_video_id_and_frame(self):
        module = _load_module('render_presentation_video', SCRIPT_PATH)

        parsed = module.parse_sample_name('S11_01_308')

        self.assertEqual(parsed.video_id, 'S11_01')
        self.assertEqual(parsed.frame_id, 308)
        self.assertEqual(parsed.sample_name, 'S11_01_308')

    def test_collect_sequence_records_merges_train_and_test_sorted_by_frame(self):
        module = _load_module('render_presentation_video', SCRIPT_PATH)
        self._write_split('train_data', ['S11_01_10', 'S11_01_12', 'S12_01_1'])
        self._write_split('test_data', ['S11_01_11', 'S11_01_13'])

        records = module.collect_sequence_records(
            data_root=self.data_root,
            video_id='S11_01',
            splits=['train_data', 'test_data'])

        self.assertEqual([record.sample_name for record in records], [
            'S11_01_10',
            'S11_01_11',
            'S11_01_12',
            'S11_01_13',
        ])
        self.assertEqual([record.split for record in records], [
            'train_data',
            'test_data',
            'train_data',
            'test_data',
        ])

    def test_collect_sequence_records_rejects_duplicate_sample_names(self):
        module = _load_module('render_presentation_video', SCRIPT_PATH)
        self._write_split('train_data', ['S11_01_10'])
        self._write_split('test_data', ['S11_01_10'])

        with self.assertRaisesRegex(ValueError, 'Duplicate sample name'):
            module.collect_sequence_records(
                data_root=self.data_root,
                video_id='S11_01',
                splits=['train_data', 'test_data'])

    def test_select_longest_contiguous_segment_prefers_longest_run(self):
        module = _load_module('render_presentation_video', SCRIPT_PATH)
        self._write_split('train_data', [
            'S11_01_10',
            'S11_01_11',
            'S11_01_20',
            'S11_01_21',
            'S11_01_22',
        ])
        self._write_split('test_data', [])
        records = module.collect_sequence_records(self.data_root, 'S11_01', ['train_data', 'test_data'])

        selected = module.select_contiguous_segment(records, segment='longest')

        self.assertEqual([record.sample_name for record in selected], [
            'S11_01_20',
            'S11_01_21',
            'S11_01_22',
        ])

    def test_select_contiguous_segment_can_use_explicit_frame_bounds(self):
        module = _load_module('render_presentation_video', SCRIPT_PATH)
        self._write_split('train_data', ['S11_01_10', 'S11_01_11', 'S11_01_12', 'S11_01_20'])
        self._write_split('test_data', [])
        records = module.collect_sequence_records(self.data_root, 'S11_01', ['train_data', 'test_data'])

        selected = module.select_contiguous_segment(records, start_frame=11, end_frame=12)

        self.assertEqual([record.sample_name for record in selected], ['S11_01_11', 'S11_01_12'])

    def test_resolve_source_video_from_video_id_folder(self):
        module = _load_module('render_presentation_video', SCRIPT_PATH)
        video_root = self.work_dir / 'videos'
        video_dir = video_root / 'S52_40'
        video_dir.mkdir(parents=True)
        expected = video_dir / 'output.mkv'
        expected.write_bytes(b'video')

        resolved = module.resolve_source_video_path(
            video_id='S52_40',
            source_video=None,
            source_video_root=video_root)

        self.assertEqual(resolved, expected)

    def test_load_time_index_map_accepts_sample_names_and_numeric_rows(self):
        module = _load_module('render_presentation_video', SCRIPT_PATH)
        time_list = self.work_dir / 'time_list.txt'
        time_list.write_text(
            'S52_40_10 0\n'
            '11 1\n'
            'S52_40_12,2\n',
            encoding='utf-8')

        mapping = module.load_time_index_map(time_list, video_id='S52_40')

        self.assertEqual(mapping, {10: 0, 11: 1, 12: 2})

    def test_load_time_index_map_accepts_timestamp_rows_by_source_index(self):
        module = _load_module('render_presentation_video', SCRIPT_PATH)
        time_list = self.work_dir / 'time_list.txt'
        time_list.write_text(
            '0_2023-04-08 13:21:21.195959\n'
            '1_2023-04-08 13:21:21.552977\n'
            '2_2023-04-08 13:21:21.554974\n',
            encoding='utf-8')

        mapping = module.load_time_index_map(time_list, video_id='S52_40')

        self.assertEqual(mapping, {0: 0, 1: 1, 2: 2})

    def test_source_video_frame_index_uses_time_list_when_available(self):
        module = _load_module('render_presentation_video', SCRIPT_PATH)

        frame_index = module.resolve_source_frame_index(frame_id=308, frame_offset=0, time_index_map={308: 17})

        self.assertEqual(frame_index, 17)

    def test_figure_to_rgb_array_uses_buffer_rgba_when_available(self):
        module = _load_module('render_presentation_video', SCRIPT_PATH)
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(1, 1))
        frame = module._figure_to_rgb_array(fig, np)
        plt.close(fig)

        self.assertEqual(frame.shape[2], 3)

    def test_normalize_video_frame_returns_contiguous_uint8_rgb(self):
        module = _load_module('render_presentation_video', SCRIPT_PATH)
        import numpy as np

        frame = np.zeros((11, 13, 4), dtype=np.float32)
        normalized = module._normalize_video_frame(frame)

        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.shape, (11, 13, 3))
        self.assertTrue(normalized.flags['C_CONTIGUOUS'])

    def test_build_video_model_assets_places_baseline_before_target(self):
        module = _load_module('render_presentation_video', SCRIPT_PATH)

        model_assets = module.build_video_model_assets(
            target_asset={'model_id': 'M3'},
            baseline_asset={'model_id': 'M0'})

        self.assertEqual([asset['model_id'] for asset in model_assets], ['M0', 'M3'])
