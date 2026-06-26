import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = (
    ROOT / 'paper_assets' / 'manuscript_latex' / 'resfes2026_witidar'
    / 'main.tex'
)
MANIFEST = ROOT / 'paper_assets' / 'logs' / 'full_alation_20e' / 'experiment_manifest.json'


def table_body(source, label):
    match = re.search(
        rf'\\label\{{{re.escape(label)}\}}(.*?)\\end\{{table\}}',
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f'Missing table {label}')
    return match.group(1)


def plain_table_values(table):
    return re.sub(r'\\textbf\{([^{}]*)\}', r'\1', table)


class TestManuscriptMetricAlignment(unittest.TestCase):
    def setUp(self):
        self.source = MANUSCRIPT.read_text(encoding='utf-8')
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        self.metrics = {
            record['experiment_id']: record['metrics']
            for record in manifest['experiments']
        }

    def test_main_ablation_mpjpe_rows_match_manifest(self):
        table = plain_table_values(table_body(self.source, 'tab:main'))
        for model_id in [
            'M0', 'M1', 'M2', 'M3', 'M4', 'M5',
            'M6', 'M7', 'M8', 'M9', 'M9_RF2',
        ]:
            expected = f"& {self.metrics[model_id]['mpjpe']:.3f} &"
            self.assertIn(expected, table, model_id)

    def test_main_ablation_efficiency_rows_match_manifest(self):
        table = plain_table_values(table_body(self.source, 'tab:main'))
        for model_id in ['M0', 'M6', 'M9', 'M9_RF2']:
            metric = self.metrics[model_id]
            for expected in [
                f"& {metric['fps']:.1f} &",
                f"& {metric['params_m']:.3f} &",
                f"& {metric['peak_memory_allocated_mb']:.2f}",
            ]:
                self.assertIn(expected, table, model_id)

    def test_flow_table_reuses_manifest_values_for_logged_variants(self):
        table = plain_table_values(table_body(self.source, 'tab:flow'))
        for model_id in ['M6', 'M9', 'M9_RF2', 'T_FW2_20e_RF2', 'T_FW2_20e_RF4']:
            expected = f"& {self.metrics[model_id]['mpjpe']:.3f} &"
            self.assertIn(expected, table, model_id)

    def test_training_text_does_not_claim_unrecorded_seed(self):
        self.assertNotIn('Each configuration uses a fixed random seed', self.source)
        self.assertIn('metrics-only provenance', self.source)
        self.assertIn('dataset signature, commit, and provenance status', self.source)

    def test_manifest_table_keeps_shared_reproducibility_anchors_outside_resizebox(self):
        self.assertIn(
            r'Shared test signature: \texttt{0b7c80f119}; commit: '
            r'\texttt{7c3dfd2cfb}.',
            self.source,
        )


if __name__ == '__main__':
    unittest.main()
