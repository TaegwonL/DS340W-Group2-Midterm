import argparse
import csv
import os
from collections import defaultdict

import plot_style
import matplotlib.pyplot as plt

METRICS = {
    'auroc': 'AUROC (%)  ↑', 'aupr_err': 'AUPR-Error (%)  ↑', 'aurc': 'AURC (×10³)  ↓',
    'eaurc': 'E-AURC (×10³)  ↓', 'fpr95': 'FPR@95%TPR (%)  ↓', 'acc': 'Accuracy (%)',
}
# scorer -> (line style, marker fill, label); the tag supplies color and marker shape
SCORER_STYLE = {
    'msp': ('--', 'white', 'MSP'),
    'ra_single': (':', 'white', 'Risk Advisor (single)'),
    'ra_ens_p_err': ('-', None, 'RA ensemble, p_err'),
    'ra_ens_total': ('-.', None, 'RA ensemble, total risk'),
    'risk_advisor': ('-', None, 'Risk Advisor'),  # rows from the earlier results.csv format
}


def read(path, corruption):
    series = defaultdict(dict)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('corruption', 'all') != corruption:
                continue
            series[(row['tag'], row['scorer'])][int(row['severity'])] = row
    return series


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='./output_shift/results.csv')
    p.add_argument('--metric', default='auroc', choices=list(METRICS))
    p.add_argument('--corruption', default='all',
                   help="'all' = pooled over corruptions, or one name such as fog")
    p.add_argument('--tags', nargs='*', default=None, help='default: every tag in the csv')
    p.add_argument('--scorers', nargs='*', default=None, help='default: every scorer in the csv')
    p.add_argument('--out', default=None,
                   help='default ./output_shift/{metric}_vs_severity[_{corruption}].png')
    args = p.parse_args()

    series = read(args.csv, args.corruption)
    if not series:
        raise SystemExit('no rows with corruption={} in {}'.format(args.corruption, args.csv))
    tags = [t for t in dict.fromkeys(k[0] for k in sorted(series)) if not args.tags or t in args.tags]
    scorers = [s for s in SCORER_STYLE if any(k[1] == s for k in series)]
    scorers += sorted({k[1] for k in series} - set(scorers))
    if args.scorers:
        scorers = [s for s in scorers if s in args.scorers]
    styles = plot_style.tag_style(tags)

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for tag in tags:
        color, marker, tag_label = styles[tag]
        for scorer in scorers:
            points = series.get((tag, scorer))
            if not points:
                continue
            sev = sorted(points)
            vals = [float(points[s][args.metric]) for s in sev]
            ls, fill, s_label = SCORER_STYLE.get(scorer, ('-', None, scorer))
            ax.plot(sev, vals, color=color, linestyle=ls, marker=marker,
                    markerfacecolor=fill or color, markeredgecolor=color, markeredgewidth=1.8,
                    label='{} + {}'.format(tag_label, s_label))

    where = 'all corruptions' if args.corruption == 'all' else args.corruption
    ax.set_title('{} vs. corruption severity ({})'.format(METRICS[args.metric].split('  ')[0], where), loc='left')
    ax.set_xlabel('Corruption severity  (0 = clean CIFAR-10)')
    ax.set_ylabel(METRICS[args.metric])
    ax.set_xticks(range(6))
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(args.csv) or '.', '{}_vs_severity{}.png'.format(
        args.metric, '' if args.corruption == 'all' else '_' + args.corruption))
    fig.savefig(out, dpi=plot_style.DPI)
    print('wrote', out)


if __name__ == '__main__':
    main()
