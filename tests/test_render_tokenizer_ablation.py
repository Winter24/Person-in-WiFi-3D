import json
import tempfile
import unittest
from pathlib import Path

from tools.analysis.render_tokenizer_ablation import render_component_table


class TokenizerAblationTableTests(unittest.TestCase):
    def _manifest(self, valid=True):
        rows = []
        for index, mode in enumerate((
            'linear', 'linear_ln', 'temporal_residual',
            'spectral_gate_residual', 'spectral',
        )):
            rows.append({
                'run_id': f'T{index}',
                'mode': mode,
                'mpjpe': 172.5 - index,
                'mpjpe_1p': 130.0 - index,
                'mpjpe_2p': 165.0 - index,
                'mpjpe_3p': 195.0 - index,
                'fps': 120.0 + index,
                'params_m': 13.1,
                'peak_memory_allocated_mb': 155.0,
                'latency_ms': 8.3 - index * 0.1,
            })
        return {
            'protocol': {'scope': 'single-seed diagnostic ablation'},
            'endpoint_audit': {'valid': valid},
            'rows': rows,
        }

    def test_table_is_generated_only_from_a_valid_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / 'manifest.json'
            output = root / 'table.tex'
            manifest.write_text(json.dumps(self._manifest()), encoding='utf-8')
            render_component_table(manifest, output)
            table = output.read_text(encoding='utf-8')

        for text in ('T0', 'T1', 'T2', 'T3', 'T4', 'Single-seed diagnostic'):
            self.assertIn(text, table)
        self.assertIn('172.500', table)
        self.assertIn('8.30', table)

    def test_invalid_endpoint_audit_blocks_table_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / 'manifest.json'
            manifest.write_text(
                json.dumps(self._manifest(valid=False)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'endpoint audit'):
                render_component_table(manifest, root / 'table.tex')


if __name__ == '__main__':
    unittest.main()
