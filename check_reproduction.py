"""Compare a fresh run against the committed results and print the paper's tables.

Run after eval_shift.py (both tags) and label_free_trigger.py have written to --dir.
The committed reference files in ./output_shift/ are what the paper reports.

Usage:
    python check_reproduction.py                      # compares ./demo_output with ./output_shift
    python check_reproduction.py --dir demo_output --ref output_shift
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

FILES = {
    'results.csv': ['tag', 'corruption', 'severity', 'scorer'],
    'monitor_baseline.csv': ['tag', 'corruption', 'severity'],
    'monitor_fmfp.csv': ['tag', 'corruption', 'severity'],
    'monitor_summary.csv': ['tag', 'signal'],
    'trigger_summary.csv': ['tag', 'method', 'signal', 'tau'],
    'trigger_table.csv': ['method', 'signal'],
}


def same(a, b, keys):
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if not np.allclose(a[c].values, b[c].values, rtol=0, atol=1e-6, equal_nan=True):
                return False
        elif not (a[c].astype(str).values == b[c].astype(str).values).all():
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='./demo_output', help='folder written by this run')
    p.add_argument('--ref', default='./output_shift', help='committed results used in the paper')
    args = p.parse_args()

    print('== Fresh run ({}) vs. committed results ({}) =='.format(args.dir, args.ref))
    ok = True
    for name, keys in FILES.items():
        new, ref = os.path.join(args.dir, name), os.path.join(args.ref, name)
        if not os.path.exists(new):
            print('  MISSING  ', name)
            ok = False
            continue
        match = same(pd.read_csv(new), pd.read_csv(ref), keys)
        ok &= match
        print('  {:9s}  {}'.format('MATCH' if match else 'DIFFERENT', name))

    r = pd.read_csv(os.path.join(args.dir, 'results.csv'))
    pooled = r[r.corruption.isin(['all', 'clean'])]
    t3 = pooled.pivot_table(index='severity', columns=['tag', 'scorer'], values='auroc')
    t3 = t3[[('baseline', 'msp'), ('baseline', 'ra_ens_p_err'), ('fmfp', 'msp'), ('fmfp', 'ra_ens_p_err')]]
    t3.columns = ['Base MSP', 'Base RA', 'FMFP MSP', 'FMFP RA']
    t3.index = ['Clean' if s == 0 else str(s) for s in t3.index]
    print('\n== Paper Table III: pooled AUROC (%) by severity ==')
    print(t3.to_string(float_format='{:.2f}'.format))

    print('\n== Paper Table II: label-free trigger, 75 shifted groups per model ==')
    print(pd.read_csv(os.path.join(args.dir, 'trigger_table.csv')).to_string(index=False))

    m = pd.read_csv(os.path.join(args.dir, 'monitor_summary.csv'))
    print('\n== Paper Section IV.C: Spearman with the true error rate (76 groups) ==')
    print(m.pivot(index='signal', columns='tag', values='spearman').to_string(float_format='{:.3f}'.format))

    print('\nRESULT:', 'all files match the paper' if ok else 'some files differ (see above)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
