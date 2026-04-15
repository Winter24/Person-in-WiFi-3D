import csv
import importlib.util
import shutil
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'render_qualitative_figure.py'


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestRenderQualitativeFigure(unittest.TestCase):
    def setUp(self):
        self.work_dir = ROOT / 'tests' / f'tmp_render_qualitative_{uuid.uuid4().hex}'
        self.work_dir.mkdir(parents=True, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def _write_experiment_log(self, rows):
        log_path = self.work_dir / 'experiment_log.csv'
        with log_path.open('w', newline='', encoding='utf-8') as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=['experiment_id', 'config', 'checkpoint', 'notes'])
            writer.writeheader()
            writer.writerows(rows)
        return log_path

    def test_script_exists(self):
        self.assertTrue(
            SCRIPT_PATH.exists(),
            f'Missing script: {SCRIPT_PATH}')

    def test_load_experiment_specs_uses_config_basenames(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)
        log_path = self._write_experiment_log([
            {
                'experiment_id': 'M0',
                'config': '/content/drive/.../M0/petr_wifi.py',
                'checkpoint': '/content/drive/.../M0/latest.pth',
                'notes': 'baseline',
            },
            {
                'experiment_id': 'M4',
                'config': '/content/drive/.../M4/wi_tidir_wifi.py',
                'checkpoint': '/content/drive/.../M4/latest.pth',
                'notes': 'ours',
            },
        ])

        specs = module.load_experiment_specs(log_path)

        self.assertEqual(specs['M0']['config_name'], 'petr_wifi.py')
        self.assertEqual(specs['M4']['config_name'], 'wi_tidir_wifi.py')

    def test_resolve_model_assets_prefers_local_checkpoint_and_csv_config_name(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)
        paper_dir = self.work_dir / 'paper_M1-5'
        model_dir = paper_dir / 'M3'
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / 'epoch_20.pth'
        checkpoint_path.write_bytes(b'checkpoint')
        log_path = self._write_experiment_log([
            {
                'experiment_id': 'M3',
                'config': '/content/drive/.../M3/wi_tidir_wifi_transformer.py',
                'checkpoint': '/content/drive/.../M3/latest.pth',
                'notes': 'flow transformer',
            },
        ])

        resolved = module.resolve_model_assets(
            project_root=ROOT,
            paper_dir=paper_dir,
            model_id='M3',
            specs=module.load_experiment_specs(log_path))

        self.assertEqual(resolved['checkpoint'], checkpoint_path)
        self.assertEqual(resolved['config'], ROOT / 'configs' / 'wifi' / 'wi_tidir_wifi_transformer.py')

    def test_panel_titles_follow_ground_truth_then_model_names(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)

        titles = module.build_panel_titles(['M0', 'M3', 'M4'])

        self.assertEqual(
            titles,
            [
                'Ground Truth',
                'M0: Person-in-WiFi 3D',
                'M3: Draft-to-Refine Rectified Flow with Transformer',
                'M4: Draft-to-Refine Rectified Flow with Mamba',
            ])

    def test_prepare_display_predictions_hides_unmatched_by_default(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)
        pred_keypoints = [
            [[0.0, 0.0, 0.0] for _ in range(14)]
            for _ in range(4)
        ]
        match_details = [
            {'gt_idx': 1, 'pred_idx': 2, 'error_mm': 145.0},
            {'gt_idx': 0, 'pred_idx': 0, 'error_mm': 120.0},
        ]
        gt_colors = ['blue', 'green']
        gt_labels = ['P1', 'P2']

        display = module.prepare_display_predictions(
            pred_keypoints=pred_keypoints,
            match_details=match_details,
            gt_colors=gt_colors,
            gt_labels=gt_labels,
            show_unmatched=False)

        self.assertEqual(display['shown_count'], 2)
        self.assertEqual(display['total_count'], 4)
        self.assertEqual(display['labels'], ['P1', 'P2'])
        self.assertEqual(display['colors'], ['blue', 'green'])
        self.assertEqual(display['hidden_unmatched'], 2)

    def test_prepare_display_predictions_can_append_unmatched(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)
        pred_keypoints = [
            [[0.0, 0.0, 0.0] for _ in range(14)]
            for _ in range(3)
        ]
        match_details = [{'gt_idx': 0, 'pred_idx': 1, 'error_mm': 135.0}]
        gt_colors = ['blue']
        gt_labels = ['P1']

        display = module.prepare_display_predictions(
            pred_keypoints=pred_keypoints,
            match_details=match_details,
            gt_colors=gt_colors,
            gt_labels=gt_labels,
            show_unmatched=True)

        self.assertEqual(display['shown_count'], 3)
        self.assertEqual(display['labels'][0], 'P1')
        self.assertEqual(display['colors'][0], 'blue')
        self.assertEqual(display['labels'][1:], ['U1', 'U3'])
        self.assertEqual(display['colors'][1:], ['#7f7f7f', '#7f7f7f'])

    def test_prepare_display_predictions_grays_poor_match_beyond_threshold(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)
        pred_keypoints = [
            [[0.0, 0.0, 0.0] for _ in range(14)],
        ]
        match_details = [{'gt_idx': 0, 'pred_idx': 0, 'error_mm': 260.0}]

        display = module.prepare_display_predictions(
            pred_keypoints=pred_keypoints,
            match_details=match_details,
            gt_colors=['blue'],
            gt_labels=['P1'],
            show_unmatched=False,
            match_quality_thr_mm=200.0)

        self.assertEqual(display['colors'], ['#7f7f7f'])
        self.assertEqual(display['labels'], ['P1'])
        self.assertEqual(display['poor_match_count'], 1)

    def test_compute_shared_pose_bounds_uses_all_panels_in_row(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)
        pose_sets = [
            [
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            ],
            [
                [[2.0, 3.0, 4.0], [2.2, 3.1, 4.3]],
            ],
        ]

        bounds = module.compute_shared_pose_bounds(pose_sets, min_range=0.35, margin_scale=0.0)

        self.assertEqual(bounds['xlim'], (0.0, 2.2))
        self.assertEqual(bounds['ylim'], (0.0, 3.1))
        self.assertEqual(bounds['zlim'], (4.3, 0.0))

    def test_score_sample_candidate_prefers_crowded_and_difficult_samples(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)
        crowded = {
            'sample_index': 10,
            'gt_count': 3,
            'crowding_score': 2.5,
            'metrics': {
                'M0': {'matched_error_mm': 190.0, 'matched_count': 2, 'false_positives': 2},
                'M3': {'matched_error_mm': 150.0, 'matched_count': 3, 'false_positives': 3},
                'M4': {'matched_error_mm': 158.0, 'matched_count': 3, 'false_positives': 2},
            },
        }
        easy = {
            'sample_index': 11,
            'gt_count': 2,
            'crowding_score': 0.5,
            'metrics': {
                'M0': {'matched_error_mm': 160.0, 'matched_count': 2, 'false_positives': 0},
                'M3': {'matched_error_mm': 162.0, 'matched_count': 2, 'false_positives': 0},
                'M4': {'matched_error_mm': 159.0, 'matched_count': 2, 'false_positives': 0},
            },
        }

        self.assertGreater(
            module.score_sample_candidate(crowded),
            module.score_sample_candidate(easy))

    def test_select_best_sample_indices_sorts_by_candidate_score(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)
        summaries = [
            {
                'sample_index': 3,
                'gt_count': 2,
                'crowding_score': 1.2,
                'metrics': {
                    'M0': {'matched_error_mm': 180.0, 'matched_count': 2, 'false_positives': 1},
                    'M3': {'matched_error_mm': 155.0, 'matched_count': 2, 'false_positives': 1},
                    'M4': {'matched_error_mm': 160.0, 'matched_count': 2, 'false_positives': 1},
                },
            },
            {
                'sample_index': 7,
                'gt_count': 3,
                'crowding_score': 2.8,
                'metrics': {
                    'M0': {'matched_error_mm': 200.0, 'matched_count': 2, 'false_positives': 2},
                    'M3': {'matched_error_mm': 145.0, 'matched_count': 3, 'false_positives': 4},
                    'M4': {'matched_error_mm': 150.0, 'matched_count': 3, 'false_positives': 3},
                },
            },
            {
                'sample_index': 9,
                'gt_count': 2,
                'crowding_score': 0.3,
                'metrics': {
                    'M0': {'matched_error_mm': 155.0, 'matched_count': 2, 'false_positives': 0},
                    'M3': {'matched_error_mm': 165.0, 'matched_count': 2, 'false_positives': 1},
                    'M4': {'matched_error_mm': 170.0, 'matched_count': 2, 'false_positives': 1},
                },
            },
        ]

        chosen = module.select_best_sample_indices(summaries, num_samples=2)

        self.assertEqual(chosen, [7, 3])

    def test_select_best_sample_indices_prefers_one_two_three_person_buckets(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)
        summaries = [
            {
                'sample_index': 101,
                'gt_count': 1,
                'crowding_score': 0.0,
                'metrics': {'M0': {'matched_error_mm': 120.0, 'matched_count': 1, 'false_positives': 0}},
            },
            {
                'sample_index': 202,
                'gt_count': 2,
                'crowding_score': 0.8,
                'metrics': {'M0': {'matched_error_mm': 170.0, 'matched_count': 2, 'false_positives': 1}},
            },
            {
                'sample_index': 303,
                'gt_count': 3,
                'crowding_score': 1.6,
                'metrics': {'M0': {'matched_error_mm': 210.0, 'matched_count': 2, 'false_positives': 2}},
            },
            {
                'sample_index': 304,
                'gt_count': 3,
                'crowding_score': 0.4,
                'metrics': {'M0': {'matched_error_mm': 160.0, 'matched_count': 3, 'false_positives': 0}},
            },
        ]

        chosen = module.select_best_sample_indices(summaries, num_samples=3)

        self.assertEqual(chosen, [101, 202, 303])

    def test_select_best_sample_indices_falls_back_when_bucket_missing(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)
        summaries = [
            {
                'sample_index': 201,
                'gt_count': 2,
                'crowding_score': 0.7,
                'metrics': {'M0': {'matched_error_mm': 170.0, 'matched_count': 2, 'false_positives': 1}},
            },
            {
                'sample_index': 301,
                'gt_count': 3,
                'crowding_score': 1.8,
                'metrics': {'M0': {'matched_error_mm': 220.0, 'matched_count': 2, 'false_positives': 3}},
            },
            {
                'sample_index': 302,
                'gt_count': 3,
                'crowding_score': 1.2,
                'metrics': {'M0': {'matched_error_mm': 180.0, 'matched_count': 3, 'false_positives': 1}},
            },
        ]

        chosen = module.select_best_sample_indices(summaries, num_samples=3)

        self.assertEqual(chosen, [201, 301, 302])

    def test_format_panel_footer_includes_sample_and_matching_metrics(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)

        footer = module.format_panel_footer(
            sample_index=1121,
            img_name='0507-2-00123',
            gt_count=2,
            matched_count=2,
            false_positives=1,
            matched_error_mm=158.43,
            poor_match_count=1)

        self.assertIn('S1121', footer)
        self.assertIn('0507-2-00123', footer)
        self.assertIn('M2/2', footer)
        self.assertIn('FP 1', footer)
        self.assertIn('P1', footer)
        self.assertIn('E158.4', footer)

    def test_validate_dataset_signatures_rejects_mismatch(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)

        with self.assertRaises(ValueError):
            module.validate_dataset_signatures([
                ('M0', ('data/wifipose/test_data', 'test')),
                ('M3', ('data/wifipose/test_data_v2', 'test')),
            ])

    def test_panel_grid_position_uses_four_columns_and_three_rows_for_three_samples(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)

        self.assertEqual(module.get_panel_grid_position(0, 'gt'), (0, 0))
        self.assertEqual(module.get_panel_grid_position(0, 'model_0'), (0, 1))
        self.assertEqual(module.get_panel_grid_position(0, 'model_1'), (0, 2))
        self.assertEqual(module.get_panel_grid_position(0, 'model_2'), (0, 3))

        self.assertEqual(module.get_panel_grid_position(1, 'gt'), (1, 0))
        self.assertEqual(module.get_panel_grid_position(1, 'model_0'), (1, 1))
        self.assertEqual(module.get_panel_grid_position(1, 'model_1'), (1, 2))
        self.assertEqual(module.get_panel_grid_position(1, 'model_2'), (1, 3))

        self.assertEqual(module.get_panel_grid_position(2, 'gt'), (2, 0))
        self.assertEqual(module.get_panel_grid_position(2, 'model_0'), (2, 1))
        self.assertEqual(module.get_panel_grid_position(2, 'model_1'), (2, 2))
        self.assertEqual(module.get_panel_grid_position(2, 'model_2'), (2, 3))

    def test_format_panel_title_wraps_long_model_titles(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)

        wrapped = module.format_panel_title(
            'M3: Draft-to-Refine Rectified Flow with Transformer',
            sample_index=1121,
            top_row=True)

        self.assertIn('\n', wrapped)
        self.assertIn('Transformer', wrapped)

        lower = module.format_panel_title('M3: Draft-to-Refine Rectified Flow with Transformer', sample_index=1121)
        self.assertEqual(lower, 'M3\nS1121')

    def test_render_style_config_prefers_larger_text_and_tighter_zoom(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)

        config = module.get_render_style_config()

        self.assertEqual(config['title_font_size'], 22)
        self.assertEqual(config['footer_font_size'], 14.4)
        self.assertEqual(config['bounds_min_range'], 0.12)
        self.assertEqual(config['bounds_margin_scale'], 0.04)


if __name__ == '__main__':
    unittest.main()
