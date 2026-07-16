"""Public-facing labels for stable experiment identifiers.

Experiment IDs remain the keys used by logs and benchmark artifacts.  This
module centralizes the functional names shown in the manuscript and figures.
"""

PUBLIC_MODEL_LABELS = {
    'M0': 'PETR Reference',
    'M1': 'Spectral PETR',
    'M2': 'Mamba PETR',
    'M3': 'Mamba-2 PETR',
    'M4': 'Transformer Draft',
    'M5': 'Mamba Draft',
    'M6': 'Mamba-2 Draft',
    'M7': 'Transformer Flow',
    'M8': 'Mamba Flow',
    'M9': 'Mamba-2 Flow (1 step)',
    'M9_RF2': 'Mamba-2 Flow (2 steps)',
}

SHORT_MODEL_LABELS = {
    'M0': 'PETR\nRef.',
    'M1': 'Spec.\nPETR',
    'M2': 'Mamba\nPETR',
    'M3': 'Mamba-2\nPETR',
    'M4': 'Trans.\nDraft',
    'M5': 'Mamba\nDraft',
    'M6': 'Mamba-2\nDraft',
    'M7': 'Trans.\nFlow',
    'M8': 'Mamba\nFlow',
    'M9': 'Mamba-2\nFlow 1',
    'M9_RF2': 'Mamba-2\nFlow 2',
}

FLOW_SETTING_LABELS = {
    'M6': 'Mamba-2 Draft',
    'M9_no_flow': 'Mamba-2 Flow (refinement bypassed)',
    'M9': 'Mamba-2 Flow (1 step)',
    'M9_RF2': 'Mamba-2 Flow (2 steps)',
    'T_FW2_20e_RF2': 'Low-flow-loss control (2 steps)',
    'T_FW2_20e_RF4': 'Low-flow-loss control (4 steps)',
}


def display_label(experiment_id):
    """Return the full public label for a primary ablation variant."""
    return PUBLIC_MODEL_LABELS[experiment_id]


def short_label(experiment_id):
    """Return a compact two-line label for dense figures."""
    return SHORT_MODEL_LABELS[experiment_id]


def flow_setting_label(experiment_id):
    """Return the public label used by the flow-solver analysis."""
    return FLOW_SETTING_LABELS[experiment_id]
