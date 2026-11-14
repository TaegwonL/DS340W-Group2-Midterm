"""Label-free retraining trigger (final-project extension of monitor.py).

monitor.py only checks how well each label-free signal correlates with the true error
rate, and its trigger test (Erel > 2) turned out to be too easy: most groups already
pass that line. Here every method has to make an actual yes/no call per group, without
labels from that group, and we score it at stricter thresholds too.

Methods (all of them see only the group's average signal):
  ratio       Srel = signal / clean signal, fire when Srel > tau. No fitting at all.
  iso_loco    isotonic map signal -> error rate, fit on the other 14 corruption types
              plus clean data, applied to the held-out corruption (leave-one-corruption-out).
  lin_loco    same as iso_loco with a straight line instead of an isotonic curve.
  multi_loco  ridge regression on all five signals, leave-one-corruption-out.
For the fitted methods, predicted Erel = predicted error rate / clean error rate.
The clean error rate comes from the test set the model was validated on, so it is known
before deployment.

Usage:
    python label_free_trigger.py
    python label_free_trigger.py --tags baseline fmfp --taus 2 3 4
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from monitor import SIGNALS


def load(path):
    df = pd.read_csv(path)
    clean = df[df['corruption'] == 'clean'].iloc[0]
    shifted = df[df['corruption'] != 'clean'].reset_index(drop=True)
    return clean, shifted


def ratio_rule(clean, shifted, signal):
    """Srel = group signal / clean signal. Uses no labels from any shifted group."""
    return shifted[signal].values / clean[signal]


def _loco(clean, shifted, fit_predict):
    """Leave-one-corruption-out: fit on every other corruption plus the clean group."""
    pred_err = np.zeros(len(shifted))
    for name in shifted['corruption'].unique():
        test = (shifted['corruption'] == name).values
        train = pd.concat([shifted[~test], clean.to_frame().T], ignore_index=True)
        pred_err[test] = fit_predict(train, shifted[test])
    pred_err = np.clip(pred_err, 0.0, 1.0)
    return pred_err / clean['error_rate']


def iso_loco(clean, shifted, signal):
    def fp(train, test):
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
        iso.fit(train[signal].astype(float), train['error_rate'].astype(float))
        return iso.predict(test[signal].astype(float))
    return _loco(clean, shifted, fp)


def lin_loco(clean, shifted, signal):
    def fp(train, test):
        lr = LinearRegression().fit(train[[signal]].astype(float), train['error_rate'].astype(float))
        return lr.predict(test[[signal]].astype(float))
    return _loco(clean, shifted, fp)


def multi_loco(clean, shifted, alpha=1.0):
    def fp(train, test):
        sc = StandardScaler().fit(train[SIGNALS].astype(float))
        rg = Ridge(alpha=alpha).fit(sc.transform(train[SIGNALS].astype(float)),
                                    train['error_rate'].astype(float))
        return rg.predict(sc.transform(test[SIGNALS].astype(float)))
    return _loco(clean, shifted, fp)


def score(pred_erel, true_erel, clean_err, tau):
    fire = pred_erel > tau
    pos = true_erel > tau
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    miss = float((~fire & pos).sum() / n_pos) if n_pos else float('nan')
    false_alarm = float((fire & ~pos).sum() / n_neg) if n_neg else float('nan')
    return {
        'tau': tau, 'n_pos': n_pos, 'n_neg': n_neg,
        'false_alarm': 100 * false_alarm, 'miss': 100 * miss,
        'balanced_acc': 100 * (1 - (miss + false_alarm) / 2),
        'mae_err_pp': 100 * float(np.mean(np.abs(pred_erel - true_erel))) * clean_err,
    }


def run(path, tag, taus):
    clean, shifted = load(path)
    true_erel = shifted['erel'].values
    preds = {}
    for s in SIGNALS:
        preds[('ratio', s)] = ratio_rule(clean, shifted, s)
        preds[('iso_loco', s)] = iso_loco(clean, shifted, s)
        preds[('lin_loco', s)] = lin_loco(clean, shifted, s)
    preds[('multi_loco', 'all')] = multi_loco(clean, shifted)

    rows, per_group = [], shifted[['corruption', 'severity', 'error_rate', 'erel']].copy()
    for (method, signal), p in preds.items():
        per_group[f'{method}:{signal}'] = p
        for tau in taus:
            r = {'tag': tag, 'method': method, 'signal': signal}
            r.update(score(p, true_erel, float(clean['error_rate']), tau))
            rows.append(r)
    per_group.insert(0, 'tag', tag)
    return pd.DataFrame(rows), per_group


# rows shown in Table II of the paper
PAPER_ROWS = [('ratio', '1-msp'), ('ratio', 'p_err'), ('lin_loco', '1-msp'),
              ('iso_loco', '1-msp'), ('iso_loco', 'p_err'), ('multi_loco', 'all')]


def paper_table(summary, taus=(2.0, 4.0)):
    rows = []
    for method, signal in PAPER_ROWS:
        r = {'method': method, 'signal': signal}
        for tag in ['baseline', 'fmfp']:
            sub = summary[(summary['method'] == method) & (summary['signal'] == signal)
                          & (summary['tag'] == tag)]
            if sub.empty:
                continue
            r[f'{tag}_mae_pp'] = sub['mae_err_pp'].iloc[0]
            for tau in taus:
                hit = sub[sub['tau'] == tau]
                if not hit.empty:
                    r[f'{tag}_ba_tau{tau:g}'] = hit['balanced_acc'].iloc[0]
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='./output_shift')
    ap.add_argument('--tags', nargs='+', default=['baseline', 'fmfp'])
    ap.add_argument('--taus', nargs='+', type=float, default=[2.0, 3.0, 4.0])
    args = ap.parse_args()

    summaries, groups = [], []
    for tag in args.tags:
        s, g = run(os.path.join(args.dir, f'monitor_{tag}.csv'), tag, args.taus)
        summaries.append(s)
        groups.append(g)
    summary = pd.concat(summaries, ignore_index=True).round(2)
    summary.to_csv(os.path.join(args.dir, 'trigger_summary.csv'), index=False)
    pd.concat(groups, ignore_index=True).round(4).to_csv(
        os.path.join(args.dir, 'trigger_groups.csv'), index=False)

    paper_table(summary).round(1).to_csv(os.path.join(args.dir, 'trigger_table.csv'), index=False)

    show = summary[summary['tau'] == args.taus[0]].sort_values(['tag', 'balanced_acc'],
                                                               ascending=[True, False])
    print(show.to_string(index=False))
    print('wrote', os.path.join(args.dir, 'trigger_summary.csv'))


if __name__ == '__main__':
    main()
