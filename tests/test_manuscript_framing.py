import re
import unittest
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = (
    ROOT / 'paper_assets' / 'manuscript_latex' / 'resfes2026_witidar'
)
MANUSCRIPT = PAPER_DIR / 'main.tex'
TABLE_DIR = PAPER_DIR / 'tables'
FLOW_FIGURE = PAPER_DIR / 'figures' / 'fig_flow_refinement_architecture.pdf'


def body_without_related_work(source):
    """Return manuscript source with the Related Work section removed."""
    pattern = re.compile(
        r'\\section\{Related Work\}.*?(?=\\section\{System and Problem Formulation\})',
        flags=re.DOTALL,
    )
    stripped, count = pattern.subn('', source)
    if count != 1:
        raise AssertionError('Could not isolate exactly one Related Work section')
    return stripped


def implementation_details(source):
    """Return the Implementation Details subsection body."""
    pattern = re.compile(
        r'\\subsection\{Implementation Details\}(.*?)'
        r'(?=\\begin\{figure\}|\\section\{|\\subsection\{)',
        flags=re.DOTALL,
    )
    match = pattern.search(source)
    if match is None:
        raise AssertionError('Could not isolate the Implementation Details subsection')
    return match.group(1)


def related_work(source):
    """Return the Related Work section body."""
    pattern = re.compile(
        r'\\section\{Related Work\}(.*?)'
        r'(?=\\section\{System and Problem Formulation\})',
        flags=re.DOTALL,
    )
    match = pattern.search(source)
    if match is None:
        raise AssertionError('Could not isolate the Related Work section')
    return match.group(1)


def system_problem_formulation(source):
    """Return the System and Problem Formulation section body."""
    pattern = re.compile(
        r'\\section\{System and Problem Formulation\}(.*?)'
        r'(?=\\section\{Method\})',
        flags=re.DOTALL,
    )
    match = pattern.search(source)
    if match is None:
        raise AssertionError('Could not isolate System and Problem Formulation')
    return match.group(1)


class TestManuscriptFraming(unittest.TestCase):
    def setUp(self):
        source = MANUSCRIPT.read_text(encoding='utf-8')
        self.source = source
        self.related = related_work(source)
        self.system_problem = system_problem_formulation(source)
        self.body = body_without_related_work(source)
        self.tables = '\n'.join(
            path.read_text(encoding='utf-8')
            for path in sorted(TABLE_DIR.glob('*.tex'))
        )

    def test_prior_project_name_is_confined_to_related_work(self):
        searchable = f'{self.body}\n{self.tables}'
        patterns = [
            r'person\s*-\s*in\s*-\s*wi-?fi',
            r'person\s+in\s+wi-?fi',
            r'person_in_wifi',
        ]
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                self.assertIsNone(
                    re.search(pattern, searchable, flags=re.IGNORECASE),
                    f'Prior-project name found outside Related Work: {pattern}',
                )

    def test_body_uses_architecture_control_framing(self):
        lowered = f'{self.body}\n{self.tables}'.lower()
        for phrase in [
            'baseline-compatible',
            'petr reference',
            'redesign of the person-in-wifi',
        ]:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)

        self.assertIn('transformer-petr control', lowered)
        self.assertIn('public fixed-layout csi--rgb-d benchmark', lowered)

    def test_bibliography_metadata_is_not_part_of_the_body_name_ban(self):
        references = (PAPER_DIR / 'references.bib').read_text(encoding='utf-8')
        self.assertIn('Person-in-WiFi 3D', references)

    def test_implementation_details_use_architecture_terms_not_code_identifiers(self):
        details = implementation_details(self.source)
        lowered = details.lower()
        for phrase in [
            r'\texttt{wifiinputadapter}',
            r'\texttt{velocitymlp}',
            r'\texttt{time\_major}',
            'mean route fusion',
            'project default',
        ]:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)

        for phrase in [
            'spectrally conditioned temporal residual tokenizer',
            'time-major token ordering',
            'conditional mlp velocity field',
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, lowered)

    def test_method_source_and_figure_avoid_internal_identifiers(self):
        source = self.source.lower()
        figure = fitz.open(FLOW_FIGURE)
        try:
            figure_text = '\n'.join(page.get_text() for page in figure).lower()
        finally:
            figure.close()

        for phrase in [
            'wifiinputadapter',
            'velocitymlp',
            'time_major',
            'mean route fusion',
            'project default',
        ]:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, source)
                self.assertNotIn(phrase, figure_text)

    def test_related_work_uses_task_and_architecture_taxonomy(self):
        for heading in [
            'WiFi-based 2D Human Pose Estimation',
            'WiFi-based Single-Person 3D Pose Estimation',
            'WiFi-based Multi-Person 3D Pose Estimation',
            'Frequency-Aware CSI Representation and Feature Recalibration',
            'Mamba and Mamba-2 for Pose and Wireless Sensing',
            'Diffusion and Flow Matching for Pose Refinement',
        ]:
            with self.subTest(heading=heading):
                self.assertIn(rf'\subsection{{{heading}}}', self.related)

        self.assertIn(
            'WiFi-based multi-person 3D pose estimation remains a nascent '
            'area represented by a still-small set of studies.',
            self.related,
        )

    def test_spectral_tokenizer_claims_are_mechanistic_and_scoped(self):
        searchable = f'{self.source}\n{self.tables}'.lower()
        for phrase in [
            'one-sided projected-feature spectral descriptor',
            'spectrally conditioned temporal gate',
            'zero-initialized residual correction',
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, searchable)

        for phrase in [
            'doppler profile',
            'doppler-guided',
            'filters static multipath',
            'preserves 100% phase',
            'same linear representation at initialization',
            'motion-salient timestep localization',
            'first spectral tokenizer',
        ]:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, searchable)

        self.assertIn('tables/tokenizer_component_ablation.tex', self.source)
        self.assertIn('figures/fig_spectral_tokenizer.pdf', self.source)

    def test_flow_matching_novelty_is_scoped_and_not_marketed_as_diffusion(self):
        novelty = (
            'To the best of our knowledge, this is the first WiFi-based '
            'multi-person 3D pose framework to use conditional flow matching '
            'as a learned draft-to-pose correction mechanism.'
        )
        self.assertEqual(self.source.count(novelty), 1)
        novelty_first_uses = re.findall(
            r'\bfirst\b.{0,80}\b(?:framework|use|solution|work)\b',
            self.source,
            flags=re.IGNORECASE,
        )
        self.assertEqual(len(novelty_first_uses), 1)
        self.assertNotIn('one of the first', self.source.lower())
        self.assertNotIn('pioneering', self.source.lower())

        flow_section = re.search(
            r'\\subsection\{Diffusion and Flow Matching for Pose Refinement\}'
            r'(.*?)(?=\\section\{|\\subsection\{)',
            self.source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(flow_section)
        flow_text = flow_section.group(1).lower()
        for phrase in [
            'does not start from gaussian noise',
            'does not generate multiple hypotheses',
            'rectified-flow-inspired conditional flow matching',
            'detached query-conditioned wifi pose draft',
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, flow_text)

    def test_related_work_cites_only_reviewed_or_accepted_pose_sources(self):
        for preprint_key in [
            'chen2025robustwifi',
            'dao2026wiflow',
            'wang2018csinet',
            'wang2019canwifi',
        ]:
            with self.subTest(preprint_key=preprint_key):
                self.assertNotIn(preprint_key, self.related)

        references = (PAPER_DIR / 'references.bib').read_text(encoding='utf-8')
        for reviewed_key in [
            'chen2023easfn',
            'zhou2023metafiplus',
            'gian2024hpeli',
            'huang2025posemamba',
            'huang2025sensemamba',
            'nguyen2026wifimamba',
            'lipman2023flowmatching',
            'wang2026fmpose3d',
            'le2026fmpose',
        ]:
            with self.subTest(reviewed_key=reviewed_key):
                self.assertIn(f'{{{reviewed_key},', references)

    def test_draft_to_refine_algorithm_exposes_shared_training_and_inference_steps(self):
        self.assertIn(
            r'Algorithm~\ref{alg:draft-to-refine} consolidates',
            self.source,
        )
        match = re.search(
            r'\\begin\{algorithm\}\[!t\](.*?)\\end\{algorithm\}',
            self.source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        algorithm = match.group(1)

        for phrase in [
            r'\REQUIRE',
            r'\ENSURE',
            r'\STATE \textbf{Shared forward pass}',
            r'\IF{$m=\mathrm{train}$}',
            'Hungarian',
            r'\mathcal{L}_{\mathrm{flow}}',
            r'\mathcal{L}=2\mathcal{L}_{\mathrm{cls}}',
            r'\STATE \textbf{Inference branch}',
            r'\FOR{$k=0,\ldots,N-1$}',
            r'\Delta t=1/N',
            '14\\times3',
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, algorithm)

        for equation in [
            'eq:token-fusion',
            'eq:mamba-block',
            'eq:flow-start',
            'eq:flow-path',
            'eq:flow-target',
            'eq:euler',
        ]:
            with self.subTest(equation=equation):
                self.assertIn(equation, algorithm)

    def test_system_problem_formulation_follows_signal_to_set_prediction_flow(self):
        section = self.system_problem
        for heading in [
            'CSI Measurement and Physical Meaning',
            'CSI Preprocessing and Phase Calibration',
            'Sensing Layout and Tensor Construction',
            'Multi-Person 3D Pose Formulation',
        ]:
            with self.subTest(heading=heading):
                self.assertIn(rf'\subsection{{{heading}}}', section)

        for phrase in [
            r'H_i=|H_i|e^{j\phi_i}',
            'DWT-based amplitude preprocessing',
            r'1\times3\times3\times30\times20',
            r'3\times3\times20\times60',
            '180 WiFi tokens',
            r'\mathcal P=\{p_m\}_{m=1}^{M}',
            r'\hat{\mathcal P}=\{(\hat p_q,s_q)\}_{q=1}^{Q}',
            'Hungarian',
            r'fig_problem_formulation_gpt_final.png',
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, section)

        self.assertIn(r'\cite{wang2015phasefi}', section)
        self.assertNotIn('discrete-wavelet amplitude denoising', section.lower())
        self.assertRegex(section, r'Kinect.*not.*inference')
        self.assertTrue(section.rstrip().endswith(r'\FloatBarrier'))

        references = (PAPER_DIR / 'references.bib').read_text(encoding='utf-8')
        self.assertIn('{wang2015phasefi,', references)


if __name__ == '__main__':
    unittest.main()
