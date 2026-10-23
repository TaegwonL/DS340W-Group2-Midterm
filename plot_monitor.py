"""Label-free signals vs. true error rate, one point per (corruption, severity) group."""
import argparse
import csv
import glob
import math
import os

import numpy as np
from scipy import stats

import plot_style
import matplotlib.pyplot as plt
from monitor import SIGNALS, EREL_THRESHOLD

SIGNAL_LABELS = {
    '1-msp': 'mean(1 − MSP)', 'p_err': 'mean p_err', 'total_risk': 'mean total risk',
    'epistemic': 'mean epistemic', 'aleatoric': 'mean aleatoric',
}


def read(paths):
    data = {}
    for path in paths:
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        if rows:
            data[rows[0]['tag']] = rows
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='./output_shift', help='folder with monitor_{tag}.csv files')
    p.add_argument('--tags', nargs='*', default=None, help='default: every monitor_{tag}.csv found')
    p.add_argument('--signals', nargs='*', default=SIGNALS, choices=SIGNALS)
    p.add_argument('--out', default=None, help='default {dir}/monitor_signals.png')
    args = p.parse_args()

    if args.tags:
        paths = [os.path.join(args.dir, 'monitor_{}.csv'.format(t)) for t in args.tags]
    else:
        paths = sorted(glob.glob(os.path.join(args.dir, 'monitor_*.csv')))
        paths = [q for q in paths if not q.endswith('monitor_summary.csv')]
    data = read(paths)
    if not data:
        raise SystemExit('no monitor_{tag}.csv in ' + args.dir)
    tags = sorted(data, key=lambda t: (t not in plot_style.TAG_COLORS, t))
    styles = plot_style.tag_style(tags)

    plot_style.apply()
    ncols = min(3, len(args.signals))
    nrows = math.ceil(len(args.signals) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 5.4 * nrows), squeeze=False)
    for ax in axes.flat[len(args.signals):]:
        ax.set_visible(False)

    for ax, sig in zip(axes.flat, args.signals):
        for tag in tags:
            rows = data[tag]
            color, marker, label = styles[tag]
            err = np.array([float(r['error_rate']) for r in rows]) * 100
            y = np.array([float(r[sig]) for r in rows])
            rho = stats.spearmanr(y, err)[0] if len(rows) > 2 else float('nan')
            ax.scatter(err, y, s=64, color=color, marker=marker, alpha=0.85,
                       edgecolors='white', linewidths=1.0, zorder=3,
                       label='{}  (ρ = {:.2f})'.format(label, rho))
            clean = [float(r['error_rate']) for r in rows if r['corruption'] == 'clean']
            if clean and EREL_THRESHOLD * clean[0] < 1:
                ax.axvline(EREL_THRESHOLD * clean[0] * 100, color=color, linestyle=':',
                           linewidth=1.6, zorder=2)
        ax.set_title(SIGNAL_LABELS[sig], loc='left')
        ax.set_xlabel('True error rate (%)')
        ax.legend(frameon=False, loc='best', title='Spearman vs. error', title_fontsize=12)

    fig.suptitle('Label-free signals vs. true error  (one point per corruption × severity; '
                 'dotted line = Erel {:g})'.format(EREL_THRESHOLD), x=0.01, ha='left', fontsize=17)
    fig.tight_layout()
    out = args.out or os.path.join(args.dir, 'monitor_signals.png')
    fig.savefig(out, dpi=plot_style.DPI)
    print('wrote', out)


if __name__ == '__main__':
    main()
