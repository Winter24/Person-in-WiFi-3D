import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'scripts' / 'run_paper_postprocess.sh'
COLAB_BATCH_SCRIPT_PATH = ROOT / 'scripts' / 'run_colab_variant_batch.sh'


class TestPaperPostprocessScript(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(
            SCRIPT_PATH.exists(),
            f'Missing script: {SCRIPT_PATH}')

    def test_script_exposes_wrapper_modes(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('MODE="${1:-default}"', text)
        self.assertIn('case "$MODE" in', text)
        self.assertIn('default)', text)
        self.assertIn('root-paper)', text)
        self.assertIn('colab)', text)

    def test_script_delegates_root_paper_mode_to_helper(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('python tools/analysis/run_root_paper_postprocess.py "$@"', text)

    def test_script_delegates_colab_mode_to_helper(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('python tools/analysis/run_colab_paper_postprocess.py "$@"', text)

    def test_script_has_checkpoint_overrides(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        for run_id in ('M0', 'M1', 'M2', 'M3', 'M4', 'M5'):
            self.assertIn(f'{run_id}_CKPT=""', text)

    def test_script_has_checkpoint_fallback_logic(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('latest.pth', text)
        self.assertIn('epoch_*.pth', text)

    def test_script_calls_expected_tools(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('python tools/test.py', text)
        self.assertIn('python tools/analysis/benchmark.py', text)
        self.assertIn('python tools/analysis/append_experiment_log.py', text)

    def test_script_generates_teaser_figure(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('python tools/analysis/plot_teaser_figure.py', text)
        self.assertIn('paper_assets/figures/figure1_teaser', text)

    def test_script_targets_expected_runs(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('RUN_IDS=(M0 M1 M2 M3 M4 M5)', text)

    def test_script_skips_missing_run_dirs(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('WARNING: Missing run directory and override for $run_id. Skipping.', text)
        self.assertIn('continue', text)

    def test_script_supports_directory_override(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('if [[ -d "$override" ]]; then', text)
        self.assertIn('resolve_config_dir', text)

    def test_script_can_use_override_when_default_run_dir_is_missing(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('resolve_run_dir', text)
        self.assertIn('run_dir="$(resolve_run_dir "$run_id")"', text)

    def test_script_selects_wimamba_variant_from_run_directory_name(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('select_wimamba_source()', text)
        self.assertIn('wimamba_v1.py', text)
        self.assertIn('wimamba_v2.py', text)
        self.assertIn('wimamba_v3.py', text)
        self.assertIn('wimamba.py', text)
        self.assertIn('*_v1*)', text)
        self.assertIn('*_v2*)', text)
        self.assertIn('*_v3*)', text)

    def test_script_restores_and_applies_selected_wimamba_source(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('WIMAMBA_BACKUP=', text)
        self.assertIn('restore_wimamba_default()', text)
        self.assertIn('trap restore_wimamba_default EXIT', text)
        self.assertIn('apply_wimamba_source "$run_dir"', text)
        self.assertIn('cp "$source_path" "$WIMAMBA_LIVE"', text)

    def test_colab_variant_batch_script_exists(self):
        self.assertTrue(
            COLAB_BATCH_SCRIPT_PATH.exists(),
            f'Missing script: {COLAB_BATCH_SCRIPT_PATH}')

    def test_colab_variant_batch_script_targets_expected_runs(self):
        text = COLAB_BATCH_SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('RUNS=(', text)
        self.assertIn('"M1_v1"', text)
        self.assertIn('"M1_v3"', text)
        self.assertIn('"M5_v1"', text)
        self.assertIn('"M5_v2"', text)
        self.assertIn('"M5_v3"', text)

    def test_colab_variant_batch_script_switches_wimamba_variant_and_restores_files(self):
        text = COLAB_BATCH_SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('INIT_FILE="$BACKBONE_DIR/__init__.py"', text)
        self.assertIn('LIVE_FILE="$BACKBONE_DIR/wimamba_v1.py"', text)
        self.assertIn('restore_backbone_files()', text)
        self.assertIn('trap restore_backbone_files EXIT', text)
        self.assertIn('select_variant_file()', text)
        self.assertIn('*_v1*)', text)
        self.assertIn('*_v2*)', text)
        self.assertIn('*_v3*)', text)
        self.assertIn('cp "$VARIANT_FILE" "$LIVE_FILE"', text)

    def test_colab_variant_batch_script_runs_eval_benchmark_and_csv_append(self):
        text = COLAB_BATCH_SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('python tools/test.py', text)
        self.assertIn('python tools/analysis/benchmark.py', text)
        self.assertIn('python tools/analysis/append_experiment_log.py', text)
        self.assertIn('experiment_log.csv', text)


if __name__ == '__main__':
    unittest.main()
