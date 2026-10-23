"""Label-free deployment monitoring, after "Temporal quality degradation in AI models"
(Scientific Reports 2022).

For every (corruption, severity) group we average signals that need no labels and
compare them with the true error rate. Erel = group error / clean error, and Erel > 2
is the point where the model is treated as no longer accurate, i.e. a retraining
trigger should fire.
"""
import numpy as np
from scipy import stats
from sklearn import metrics

SIGNALS = ['1-msp', 'p_err', 'total_risk', 'epistemic', 'aleatoric']
EREL_THRESHOLD = 2.0


def group_row(signals, correct):
    """signals: name -> per-sample array. Returns per-group means and the error rate."""
    row = {name: float(np.mean(signals[name])) for name in SIGNALS}
    row['n'] = int(len(correct))
    row['error_rate'] = float(1 - np.mean(correct))
    return row


def add_erel(rows, clean_error):
    for row in rows:
        row['erel'] = row['error_rate'] / clean_error if clean_error > 0 else float('nan')
        row['trigger'] = int(row['erel'] > EREL_THRESHOLD)
    return rows


def summarize(rows):
    """Spearman(signal, error rate) and retraining-trigger AUROC over all groups."""
    err = np.array([r['error_rate'] for r in rows])
    trig = np.array([r['trigger'] for r in rows])
    out = []
    for name in SIGNALS:
        x = np.array([r[name] for r in rows])
        rho, p = stats.spearmanr(x, err) if len(rows) > 2 else (float('nan'), float('nan'))
        auroc = metrics.roc_auc_score(trig, x) if 0 < trig.sum() < len(trig) else float('nan')
        out.append({'signal': name, 'spearman': float(rho), 'spearman_p': float(p),
                    'trigger_auroc': 100 * float(auroc), 'n_groups': len(rows),
                    'n_trigger': int(trig.sum())})
    return out
