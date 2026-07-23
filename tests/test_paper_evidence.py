import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = (
    ROOT / 'paper_assets' / 'manuscript_latex' / 'resfes2026_witidar'
    / 'main.tex'
)


class TestPaperEvidence(unittest.TestCase):
    def test_audit_requires_all_display_items_to_be_cited_in_prose(self):
        from tools.analysis.paper_evidence import audit_manuscript

        audit = audit_manuscript(MANUSCRIPT)

        self.assertEqual(audit['undefined_references'], [])
        self.assertEqual(audit['uncited_display_items'], [])

    def test_audit_reconciles_headline_claims_with_canonical_log(self):
        from tools.analysis.paper_evidence import audit_manuscript

        audit = audit_manuscript(MANUSCRIPT)

        self.assertEqual(audit['claim_errors'], [])
        self.assertEqual(audit['canonical_log'], 'full_alation_20e')

    def test_audit_requires_metric_captions_to_state_direction_and_units(self):
        from tools.analysis.paper_evidence import audit_manuscript

        audit = audit_manuscript(MANUSCRIPT)

        self.assertEqual(audit['caption_errors'], [])


if __name__ == '__main__':
    unittest.main()
