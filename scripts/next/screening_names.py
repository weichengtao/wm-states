"""Canonical check identifiers, measurement columns, and diagnostic labels."""

MEASUREMENT_COLUMNS = {
    'firing_rate': 'mean_test_firing_rate_hz',
    'presence_ratio': 'presence_ratio',
    'delay_variance': 'delay_to_baseline_variance_ratio',
    'baseline_variance': 'baseline_window_variance_ratio',
    'baseline_drift': 'baseline_drift_r',
    'selectivity': 'mean_selectivity_pev_pct',
    'preferred_cue_drift': 'preferred_cue_drift_r',
}
# Order is shared by screening decisions and per-check diagnostic columns.
CHECK_NAMES = tuple(MEASUREMENT_COLUMNS)
REASONS = {check: f'fail_{check}' for check in CHECK_NAMES}
CHECK_LABELS = {
    'firing_rate': 'Test-period firing rate',
    'presence_ratio': 'Correct-trial presence ratio',
    'delay_variance': 'Delay-to-baseline variance ratio',
    'baseline_variance': 'Baseline window variance ratio',
    'baseline_drift': 'Baseline firing-rate drift',
    'selectivity': 'Cue selectivity (PEV)',
    'preferred_cue_drift': 'Preferred-cue firing-rate drift',
}
UNAVAILABLE_SUFFIX = '_not_applicable'


def reason_label(reason: str) -> str:
    """Translate one canonical reason, rejecting obsolete diagnostic schemas."""
    if reason == 'pass':
        return 'Passed all enabled checks'
    if isinstance(reason, str):
        unavailable = reason.endswith(UNAVAILABLE_SUFFIX)
        base_reason = reason.removesuffix(UNAVAILABLE_SUFFIX)
        for check, code in REASONS.items():
            if base_reason == code:
                return f'{CHECK_LABELS[check]}: {"unavailable" if unavailable else "failed"}'
    raise ValueError(f'Unrecognized screening rejection reason {reason!r}. '
                     'Regenerate diagnostics with scripts/next/cell_screening.py '
                     '--save-extended-diagnostics; older diagnostic names are unsupported.')


def rejection_label(reasons: str, *, show_not_applicable: bool = True) -> str:
    """Format a cell's complete reason list without mistaking unavailability for a pass."""
    if not isinstance(reasons, str):
        return reason_label(reasons)  # Raise the same actionable error for missing CSV values.
    codes = reasons.split('|')
    labels = [(code, reason_label(code)) for code in codes]
    visible = [label for code, label in labels
               if show_not_applicable or not code.endswith(UNAVAILABLE_SUFFIX)]
    return '; '.join(visible) or 'Checks not applicable'
