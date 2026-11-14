"""Predicted vs. true Erel for the label-free trigger, one point per shifted group.

Reads trigger_groups.csv written by label_free_trigger.py.
"""
import argparse
import os

import numpy as np
import pandas as pd

import plot_style
import matplotlib.pyplot as plt

METHODS = [('ratio:1-msp', 'Ratio rule (1 − MSP)', '#7a7975', 'o'),
           ('iso_loco:1-msp', 'Isotonic LOCO (1 − MSP)', '#1baf7a', 's')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='./output_shift')
    p.add_argument('--tags', nargs='+', default=['baseline', 'fmfp'])
    p.add_argument('--taus', nargs='+', type=float, default=[2.0, 4.0])
    p.add_argument('--out', default=None)
    args = p.parse_args()

    df = pd.read_csv(os.path.join(args.dir, 'trigger_groups.csv'))
    plot_style.apply()
    fig, axes = plt.subplots(1, len(args.tags), figsize=(6.2 * len(args.tags), 5.6), squeeze=False)
    for ax, tag in zip(axes.flat, args.tags):
        d = df[df['tag'] == tag]
        for col, label, color, marker in METHODS:
            ax.scatter(d['erel'], d[col], s=56, color=color, marker=marker, alpha=0.85,
                       edgecolors='white', linewidths=0.8, zorder=3, label=label)
        lim = [0.8, max(d['erel'].max(), d[[m[0] for m in METHODS]].max().max()) * 1.2]
        ax.plot(lim, lim, color=plot_style.INK_2, linewidth=1.2, zorder=2)
        for tau in args.taus:
            ax.axvline(tau, color=plot_style.INK_2, linestyle=':', linewidth=1.2)
            ax.axhline(tau, color=plot_style.INK_2, linestyle=':', linewidth=1.2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ticks = [1, 2, 4, 8, 16]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.set_title(plot_style.TAG_LABELS.get(tag, tag), loc='left')
        ax.set_xlabel('True Erel')
        ax.set_ylabel('Predicted Erel (no labels from this group)')
        ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    out = args.out or os.path.join(args.dir, 'trigger_pred_vs_true.png')
    fig.savefig(out, dpi=plot_style.DPI)
    print('wrote', out)


if __name__ == '__main__':
    main()
