import unittest


class TestModelLabels(unittest.TestCase):
    def test_public_labels_are_functional_and_keep_stable_ids_separate(self):
        from tools.analysis.model_labels import (
            PUBLIC_MODEL_LABELS,
            display_label,
            short_label,
        )

        self.assertEqual(display_label('M0'), 'PETR Reference')
        self.assertEqual(display_label('M9_RF2'), 'Mamba-2 Flow (2 steps)')
        self.assertEqual(short_label('M9_RF2'), 'Mamba-2\nFlow 2')
        self.assertNotIn('WiTiDAR', PUBLIC_MODEL_LABELS['M6'])
        self.assertNotIn('RF2', PUBLIC_MODEL_LABELS['M9_RF2'])

    def test_flow_controls_use_public_descriptions(self):
        from tools.analysis.model_labels import flow_setting_label

        self.assertEqual(
            flow_setting_label('M9_no_flow'),
            'Mamba-2 Flow (refinement bypassed)',
        )
        self.assertEqual(
            flow_setting_label('T_FW2_20e_RF4'),
            'Low-flow-loss control (4 steps)',
        )


if __name__ == '__main__':
    unittest.main()
