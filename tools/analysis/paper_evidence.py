"""Validate manuscript display references and headline claims against canonical logs."""

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_LOG_DIR = ROOT / 'paper_assets' / 'logs' / 'full_alation_20e'
CANONICAL_LOG_NAME = 'full_alation_20e'
REQUIRED_DISPLAY_LABELS = (
    'fig:system',
    'tab:notation',
    'fig:flow',
    'tab:dataset-stats',
    'tab:variant-summary',
    'tab:main',
    'fig:main_ablation',
    'tab:flow',
    'fig:flow_ablation',
    'fig:qualitative',
    'tab:hypothesis-verdict',
    'fig:accuracy_efficiency_bubble',
    'tab:per-joint',
    'tab:receiver-ablation',
    'tab:flow-control',
)
CAPTION_REQUIREMENTS = {
    'tab:main': ('lower mpjpe', 'millimeters'),
    'fig:main_ablation': ('lower mpjpe', 'millimeters'),
    'tab:flow': ('lower', 'millimeters'),
    'fig:flow_ablation': ('lower', 'millimeters'),
    'fig:accuracy_efficiency_bubble': ('lower mpjpe', 'millimeters'),
    'tab:per-joint': ('lower', 'millimeters'),
    'tab:receiver-ablation': ('lower', 'units: mm'),
    'tab:flow-control': ('lower', 'millimeters'),
}


def _read_csv_rows():
    with (CANONICAL_LOG_DIR / 'experiment_log.csv').open(
            encoding='utf-8', newline='') as handle:
        return {row['experiment_id']: row for row in csv.DictReader(handle)}


def _read_eval(experiment_id):
    return json.loads((CANONICAL_LOG_DIR / f'{experiment_id}_eval.json').read_text(
        encoding='utf-8'))


def _ratio(numerator, denominator):
    return numerator / denominator


def canonical_claims():
    """Return the rounded headline comparisons derived from canonical artifacts."""
    rows = _read_csv_rows()
    m0, m6, m9, rf2 = (rows[key] for key in ('M0', 'M6', 'M9', 'M9_RF2'))
    bypass_eval, one_step_eval, two_step_eval = (
        _read_eval(key) for key in ('M9_no_flow', 'M9', 'M9_RF2'))

    def csv_value(row, key):
        return float(row[key])

    def eval_value(row, key):
        return float(row[key])

    rf2_mpjpe = csv_value(rf2, 'mpjpe')
    return {
        'main_mpjpe_delta_mm': round(csv_value(m0, 'mpjpe') - rf2_mpjpe, 3),
        'main_mpjpe_reduction_pct': round(
            100 * _ratio(csv_value(m0, 'mpjpe') - rf2_mpjpe,
                         csv_value(m0, 'mpjpe')), 2),
        'main_throughput_ratio': round(
            _ratio(csv_value(rf2, 'fps'), csv_value(m0, 'fps')), 2),
        'main_parameter_reduction_pct': round(
            100 * _ratio(csv_value(m0, 'params_m') - csv_value(rf2, 'params_m'),
                         csv_value(m0, 'params_m')), 2),
        'main_memory_reduction_pct': round(
            100 * _ratio(csv_value(m0, 'peak_memory_allocated_mb') -
                         csv_value(rf2, 'peak_memory_allocated_mb'),
                         csv_value(m0, 'peak_memory_allocated_mb')), 2),
        'draft_to_two_step_mpjpe_delta_mm': round(
            csv_value(m6, 'mpjpe') - rf2_mpjpe, 3),
        'draft_to_two_step_fps_reduction_pct': round(
            100 * _ratio(csv_value(m6, 'fps') - csv_value(rf2, 'fps'),
                         csv_value(m6, 'fps')), 2),
        'bypass_to_two_step_mpjpe_delta_mm': round(
            eval_value(bypass_eval, 'mpjpe') - eval_value(two_step_eval, 'mpjpe'), 3),
        'one_step_to_two_step_mpjpe_delta_mm': round(
            eval_value(one_step_eval, 'mpjpe') - eval_value(two_step_eval, 'mpjpe'), 3),
        'bypass_to_two_step_matched_delta_mm': round(
            _matched_only_mpjpe(bypass_eval) - _matched_only_mpjpe(two_step_eval), 3),
        'one_step_to_two_step_matched_delta_mm': round(
            _matched_only_mpjpe(one_step_eval) - _matched_only_mpjpe(two_step_eval), 3),
        'bypass_missed_persons': int(bypass_eval['missed_persons']),
        'one_step_missed_persons': int(one_step_eval['missed_persons']),
        'two_step_missed_persons': int(two_step_eval['missed_persons']),
    }


def _matched_only_mpjpe(eval_row):
    matched = int(eval_row['matched_persons'])
    total = int(eval_row['total_gt_persons'])
    missed = int(eval_row['missed_persons'])
    penalty = float(eval_row['miss_penalty_mm'])
    return (float(eval_row['mpjpe']) * total - penalty * missed) / matched


def _all_tex_source(manuscript):
    tex_root = manuscript.parent
    source = manuscript.read_text(encoding='utf-8')
    for table_path in sorted((tex_root / 'tables').glob('*.tex')):
        source += '\n' + table_path.read_text(encoding='utf-8')
    return source


def _claim_patterns(claims):
    return {
        'main MPJPE improvement': rf"{claims['main_mpjpe_delta_mm']:.3f} mm \({claims['main_mpjpe_reduction_pct']:.2f}\\%\)",
        'main throughput ratio': rf"{claims['main_throughput_ratio']:.2f}\\times",
        'parameter reduction': rf"{claims['main_parameter_reduction_pct']:.2f}\\%",
        'memory reduction': rf"{claims['main_memory_reduction_pct']:.2f}\\%",
        'draft-to-two-step MPJPE': rf"{claims['draft_to_two_step_mpjpe_delta_mm']:.3f} mm MPJPE",
        'draft-to-two-step throughput': rf"{claims['draft_to_two_step_fps_reduction_pct']:.2f}\\% throughput",
        'bypass-to-two-step MPJPE': rf"{claims['bypass_to_two_step_mpjpe_delta_mm']:.3f} mm MPJPE",
        'one-step-to-two-step MPJPE': rf"{claims['one_step_to_two_step_mpjpe_delta_mm']:.3f} mm MPJPE",
        'bypass-to-two-step matched': rf"{claims['bypass_to_two_step_matched_delta_mm']:.3f} mm",
        'one-step-to-two-step matched': rf"{claims['one_step_to_two_step_matched_delta_mm']:.3f} mm",
        'bypass missed persons': rf"{claims['bypass_missed_persons']}\\rightarrow{claims['two_step_missed_persons']}",
        'one-step missed persons': rf"{claims['one_step_missed_persons']}\\rightarrow{claims['two_step_missed_persons']}",
    }


def _caption_before_label(source, label):
    label_offset = source.find(f'\\label{{{label}}}')
    if label_offset < 0:
        return ''
    caption_offset = source.rfind('\\caption{', 0, label_offset)
    if caption_offset < 0:
        return ''
    return source[caption_offset:label_offset]


def audit_manuscript(manuscript):
    """Return a deterministic audit of display references and headline evidence."""
    manuscript = Path(manuscript)
    source = _all_tex_source(manuscript)
    labels = set(re.findall(r'\\label\{([^}]+)\}', source))
    references = set(re.findall(r'\\(?:ref|autoref)\{([^}]+)\}', source))
    claims = canonical_claims()
    claim_errors = [
        name for name, pattern in _claim_patterns(claims).items()
        if re.search(pattern, source) is None
    ]
    caption_errors = []
    for label, snippets in CAPTION_REQUIREMENTS.items():
        caption = _caption_before_label(source, label).lower()
        if any(snippet not in caption for snippet in snippets):
            caption_errors.append(label)
    return {
        'canonical_log': CANONICAL_LOG_NAME,
        'undefined_references': sorted(references - labels),
        'uncited_display_items': sorted(
            label for label in REQUIRED_DISPLAY_LABELS if label not in references),
        'claim_errors': claim_errors,
        'caption_errors': caption_errors,
    }
