"""Smoke tests that need neither a GPU nor CIFAR-10-C.

Run with either:
    python -m pytest tests
    python tests/test_smoke.py
"""
import csv
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cifar10c  # noqa: E402
import eval_shift  # noqa: E402
import monitor  # noqa: E402
import plot_monitor  # noqa: E402
import plot_shift  # noqa: E402
from fp_metrics import evaluate, msp  # noqa: E402
from risk_advisor import RiskAdvisor  # noqa: E402

FAST = dict(n_estimators=20)


def fake_outputs(n, noise=0.0, seed=0, emb_dim=64):
    """Random logits/embeddings whose accuracy drops as noise grows."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, n)
    logits = rng.normal(0, 1, (n, 10))
    logits[np.arange(n), y] += 4.0 / (1 + noise)
    correct = (logits.argmax(1) == y).astype(int)
    emb = rng.normal(size=(n, emb_dim)).astype(np.float32)
    emb[:, 0] += 1.5 * (1 - correct)
    return logits, emb, correct


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def test_fp_metrics():
    _, _, correct = fake_outputs(500)
    perfect = evaluate(correct + 0.01 * np.random.default_rng(0).random(500), correct)
    # the E-AURC optimum r + (1-r)ln(1-r) is a continuous approximation, so ~0 not exactly 0
    assert perfect['auroc'] == 100.0 and abs(perfect['eaurc']) < 0.1
    r = evaluate(np.random.default_rng(1).random(500), correct)
    assert 30 < r['auroc'] < 70
    single = evaluate(np.ones(10), np.ones(10))
    assert single['acc'] == 100.0 and np.isnan(single['auroc'])


def test_risk_advisor_single_and_ensemble():
    lv, ev, cv = fake_outputs(800, seed=1)
    lt, et, ct = fake_outputs(400, noise=1.0, seed=2)
    for feature_set in ['logits', 'logits+emb']:
        single = RiskAdvisor(n_members=1, feature_set=feature_set, pca_dim=8, **FAST).fit(lv, cv, ev)
        r1 = single.predict_risk(lt, et)
        assert np.allclose(r1['epistemic'], 0)
        assert np.allclose(single.score(lt, et), 1 - r1['p_err'])

        ens = RiskAdvisor(n_members=4, feature_set=feature_set, pca_dim=8, **FAST).fit(lv, cv, ev)
        r = ens.predict_risk(lt, et)
        assert set(r) == {'p_err', 'aleatoric', 'epistemic', 'total_risk'}
        for k, v in r.items():
            assert v.shape == (400,) and np.all(np.isfinite(v)), k
        assert np.all((0 <= r['p_err']) & (r['p_err'] <= 1))
        assert np.all((0 <= r['aleatoric']) & (r['aleatoric'] <= 1))
        assert np.all(r['epistemic'] >= 0) and r['epistemic'].max() > 0
        assert np.allclose(r['total_risk'], r['p_err'] + r['aleatoric'] + r['epistemic'])
        assert evaluate(1 - r['p_err'], ct)['auroc'] > 60
    assert ens.pca.n_components_ == 8


def test_monitor():
    rows = []
    for i, err in enumerate([0.05, 0.06, 0.08, 0.15, 0.3, 0.5]):
        correct = np.r_[np.zeros(int(100 * err)), np.ones(100 - int(100 * err))]
        sig = {name: np.full(100, err + 0.01 * j) for j, name in enumerate(monitor.SIGNALS)}
        rows.append(monitor.group_row(sig, correct))
    monitor.add_erel(rows, rows[0]['error_rate'])
    assert rows[0]['erel'] == 1.0 and [r['trigger'] for r in rows] == [0, 0, 0, 1, 1, 1]
    for s in monitor.summarize(rows):
        assert abs(s['spearman'] - 1) < 1e-9 and s['trigger_auroc'] == 100.0 and s['n_trigger'] == 3


def test_cifar10c_mmap_loader():
    old = cifar10c.PER_SEVERITY
    cifar10c.PER_SEVERITY = 6  # tiny fake file: 5 severities x 6 images
    try:
        with tempfile.TemporaryDirectory() as d:
            imgs = np.arange(30 * 32 * 32 * 3, dtype=np.uint32).astype(np.uint8).reshape(30, 32, 32, 3)
            np.save(os.path.join(d, 'fog.npy'), imgs)
            np.save(os.path.join(d, 'labels.npy'), np.tile(np.arange(6), 5))
            mm = cifar10c.open_corruption(d, 'fog')
            assert isinstance(mm, np.memmap)
            x, y = cifar10c.severity_slice(mm, cifar10c.load_labels(d), 3, limit=4)
            assert len(x) == 4 and np.array_equal(np.asarray(x[0]), imgs[12])
            batch = next(iter(cifar10c.get_loader(x, y, batch_size=4)))
            assert tuple(batch[0].shape) == (4, 3, 32, 32)
            del mm, x
    finally:
        cifar10c.PER_SEVERITY = old


def write_fake_cache(path, corruptions, per_group=60, seed=0):
    lv, ev, cv = fake_outputs(600, seed=seed)
    parts = [('clean', 0, fake_outputs(per_group, 0.0, seed + 1))]
    for i, name in enumerate(corruptions):
        for s in range(1, 6):
            parts.append((name, s, fake_outputs(per_group, 0.6 * s, seed + 10 * i + s)))
    n = [len(p[2][2]) for p in parts]
    np.savez(path, valid_logits=lv, valid_emb=ev.astype(np.float16), valid_correct=cv,
             logits=np.concatenate([p[2][0] for p in parts]),
             emb=np.concatenate([p[2][1] for p in parts]).astype(np.float16),
             correct=np.concatenate([p[2][2] for p in parts]),
             corruption=np.repeat([p[0] for p in parts], n),
             severity=np.repeat([p[1] for p in parts], n),
             limit=np.array(per_group))


def run_main(module, argv):
    old = sys.argv
    sys.argv = [module.__name__ + '.py'] + argv
    try:
        module.main()
    finally:
        sys.argv = old


def test_eval_shift_from_cache_and_plots():
    corruptions = ['gaussian_noise', 'fog']
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'results.csv')
        for i, tag in enumerate(['baseline', 'fmfp']):
            write_fake_cache(os.path.join(d, 'cache_{}.npz'.format(tag)), corruptions, seed=100 * i)
            args = ['--tag', tag, '--use_cache', '--limit', '60', '--corruptions'] + corruptions + \
                   ['--ra_members', '2', '--ra_features', 'logits+emb', '--pca_dim', '8', '--out', out]
            run_main(eval_shift, args)
        run_main(eval_shift, args)  # rerunning a tag replaces its rows

        rows = read_csv(out)
        groups = 1 + 5 * len(corruptions) + 6  # clean + corruption x severity + pooled 'all'
        assert len(rows) == 2 * groups * len(eval_shift.SCORERS)
        assert {r['scorer'] for r in rows} == set(eval_shift.SCORERS)
        assert {r['corruption'] for r in rows} == {'clean', 'all'} | set(corruptions)

        mon = read_csv(os.path.join(d, 'monitor_fmfp.csv'))
        assert len(mon) == 1 + 5 * len(corruptions) and float(mon[0]['erel']) == 1.0
        summary = read_csv(os.path.join(d, 'monitor_summary.csv'))
        assert len(summary) == 2 * len(monitor.SIGNALS)
        assert float([s for s in summary if s['signal'] == '1-msp'][0]['spearman']) > 0.5

        for metric in ['auroc', 'aurc']:
            run_main(plot_shift, ['--csv', out, '--metric', metric])
            assert os.path.getsize(os.path.join(d, '{}_vs_severity.png'.format(metric))) > 10000
        run_main(plot_shift, ['--csv', out, '--corruption', 'fog'])
        assert os.path.exists(os.path.join(d, 'auroc_vs_severity_fog.png'))
        run_main(plot_monitor, ['--dir', d])
        assert os.path.getsize(os.path.join(d, 'monitor_signals.png')) > 10000
        import matplotlib.pyplot as plt
        plt.close('all')


def test_msp():
    p = msp(np.array([[0.0, 0.0], [10.0, 0.0]]))
    assert np.allclose(p, [0.5, 1.0], atol=1e-4)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print('ok  ', t.__name__)
    print('{} passed'.format(len(tests)))
