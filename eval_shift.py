import argparse
import csv
import os
import shutil

import numpy as np
import torch
import torch.nn as nn

from model import resnet, resnet18, densenet_BC, vgg, mobilenet, efficientnet, wrn, convmixer
from risk_advisor import RiskAdvisor, FEATURE_SETS
import cifar10c
import monitor
from utils import data as dataset
from fp_metrics import evaluate, msp

FIELDS = ['tag', 'corruption', 'severity', 'scorer', 'acc', 'auroc', 'aupr_err', 'aurc', 'eaurc', 'fpr95']
MONITOR_FIELDS = ['tag', 'corruption', 'severity', 'n'] + monitor.SIGNALS + ['error_rate', 'erel', 'trigger']
SUMMARY_FIELDS = ['tag', 'signal', 'spearman', 'spearman_p', 'trigger_auroc', 'n_groups', 'n_trigger']
SCORERS = ['msp', 'ra_single', 'ra_ens_p_err', 'ra_ens_total']


def build_model(name, num_classes):
    if name == 'resnet18':
        return resnet18.ResNet18(num_classes=num_classes)
    if name == 'res110':
        return resnet.resnet110(num_classes=num_classes)
    if name == 'dense':
        return densenet_BC.DenseNet3(depth=100, num_classes=num_classes, growth_rate=12,
                                     reduction=0.5, bottleneck=True, dropRate=0.0)
    if name == 'vgg':
        return vgg.vgg16(num_classes=num_classes)
    if name == 'wrn':
        return wrn.WideResNet(28, num_classes, 10)
    if name == 'efficientnet':
        return efficientnet.efficientnet(num_classes=num_classes)
    if name == 'mobilenet':
        return mobilenet.mobilenet(num_classes=num_classes)
    if name == 'cmixer':
        return convmixer.ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=num_classes)
    raise ValueError(name)


def load_weights(model, path):
    sd = torch.load(path, map_location='cpu')
    sd = {k[len('module.'):] if k.startswith('module.') else k: v
          for k, v in sd.items() if k != 'n_averaged'}
    model.load_state_dict(sd)
    return model


def attach_embedding_hook(model):
    """Record the input of the last nn.Linear, i.e. the penultimate-layer embedding
    (512-d after global average pooling for ResNet-18)."""
    head = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    store = {}
    head.register_forward_hook(lambda mod, inp, out: store.__setitem__('emb', inp[0].detach()))
    return store


@torch.no_grad()
def collect(model, store, loader, device):
    model.eval()
    logits, embs, correct = [], [], []
    for x, y, _ in loader:
        out = model(x.to(device))
        pred = out.argmax(1).cpu()
        logits.append(out.cpu().numpy())
        # float16 keeps the cache for 760k images under 1 GB
        embs.append(store['emb'].cpu().numpy().astype(np.float16))
        correct.append(pred.eq(torch.as_tensor(y)).numpy().astype(int))
    return np.concatenate(logits), np.concatenate(embs), np.concatenate(correct)


def run_inference(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 100 if args.data == 'cifar100' else 10

    model = build_model(args.model, num_classes)
    load_weights(model, args.ckpt)
    model.to(device)
    store = attach_embedding_hook(model)

    _, valid_loader, test_loader, _, _ = dataset.get_loader(
        args.data, args.data_path, args.batch_size, args)

    v_logits, v_emb, v_correct = collect(model, store, valid_loader, device)
    print('valid acc {:.2f}  errors {}'.format(100 * v_correct.mean(), int((1 - v_correct).sum())))

    parts = []
    if args.limit:
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(test_loader.dataset, range(min(args.limit, len(test_loader.dataset)))),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    parts.append(('clean', 0) + collect(model, store, test_loader, device))
    print('clean  acc {:.2f}'.format(100 * parts[-1][4].mean()))

    labels = cifar10c.load_labels(args.cifar10c_path)
    for name in args.corruptions:
        images = cifar10c.open_corruption(args.cifar10c_path, name)
        for s in range(1, 6):
            x, y = cifar10c.severity_slice(images, labels, s, args.limit)
            loader = cifar10c.get_loader(x, y, args.batch_size, args.num_workers)
            parts.append((name, s) + collect(model, store, loader, device))
            print('{:<18} sev {}  acc {:.2f}'.format(name, s, 100 * parts[-1][4].mean()))
        del images

    n = [len(p[2]) for p in parts]
    return dict(
        valid_logits=v_logits, valid_emb=v_emb, valid_correct=v_correct,
        logits=np.concatenate([p[2] for p in parts]),
        emb=np.concatenate([p[3] for p in parts]),
        correct=np.concatenate([p[4] for p in parts]),
        corruption=np.repeat([p[0] for p in parts], n),
        severity=np.repeat([p[1] for p in parts], n),
        limit=np.array(args.limit or 0),
    )


def load_cache(path, args):
    c = dict(np.load(path, allow_pickle=False))
    if int(c['limit']) != (args.limit or 0):
        raise SystemExit('cache {} was built with --limit {}, not {}; drop --use_cache to rebuild'
                         .format(path, int(c['limit']), args.limit or 0))
    missing = sorted(set(args.corruptions) - set(c['corruption']))
    if missing:
        raise SystemExit('cache {} has no {}; drop --use_cache to rebuild'.format(path, missing))
    keep = np.isin(c['corruption'], ['clean'] + list(args.corruptions))
    for k in ['logits', 'emb', 'correct', 'corruption', 'severity']:
        c[k] = c[k][keep]
    return c


def upsert_csv(path, fields, rows, tag):
    """Replace this tag's rows, keep other tags'. A file with another header is moved to .bak."""
    old = []
    if os.path.exists(path):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            same = reader.fieldnames == fields
            if same:
                old = [r for r in reader if r['tag'] != tag]
        if not same:
            shutil.move(path, path + '.bak')
            print('moved old {} (different columns) to .bak'.format(path))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(old + rows)


def rounded(d, nd=4):
    # p-values keep 3 significant digits instead of collapsing to 0.0
    return {k: float('{:.3g}'.format(v)) if k.endswith('_p') else round(v, nd) if isinstance(v, float) else v
            for k, v in d.items()}


def main():
    p = argparse.ArgumentParser(description='Failure prediction under corruption severity')
    p.add_argument('--ckpt', help='path to model.pth (not needed with --use_cache)')
    p.add_argument('--tag', required=True, help='label for this checkpoint, e.g. baseline or fmfp')
    p.add_argument('--model', default='resnet18')
    p.add_argument('--data', default='cifar10')
    p.add_argument('--data_path', default='./data/')
    p.add_argument('--cifar10c_path', default='./data/CIFAR-10-C/')
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--num_workers', default=0, type=int)
    p.add_argument('--corruptions', nargs='*', default=None,
                   help='subset of corruption names, default all 15')
    p.add_argument('--limit', default=None, type=int,
                   help='use only the first N images per corruption/severity (and of the clean test set)')
    p.add_argument('--ra_members', default=10, type=int, help='Risk Advisor ensemble size')
    p.add_argument('--ra_features', default='logits', choices=FEATURE_SETS)
    p.add_argument('--pca_dim', default=32, type=int, help='embedding PCA size for logits+emb')
    p.add_argument('--use_cache', action='store_true',
                   help='reuse output_shift/cache_{tag}.npz and skip GPU inference')
    p.add_argument('--out', default='./output_shift/results.csv')
    p.add_argument('--gpu', default='0')
    args = p.parse_args()
    args.corruptions = args.corruptions or cifar10c.CORRUPTIONS
    unknown = sorted(set(args.corruptions) - set(cifar10c.CORRUPTIONS))
    if unknown:
        p.error('unknown corruptions {}'.format(unknown))

    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, 'cache_{}.npz'.format(args.tag))

    if args.use_cache and os.path.exists(cache_path):
        print('using cache', cache_path)
        c = load_cache(cache_path, args)
    else:
        if not args.ckpt:
            p.error('--ckpt is required unless --use_cache finds {}'.format(cache_path))
        c = run_inference(args)
        np.savez(cache_path, **c)
        print('wrote', cache_path)

    # Risk Advisors are fit on the validation split only (see risk_advisor.py)
    print('fitting Risk Advisors ({}, ensemble of {})'.format(args.ra_features, args.ra_members))
    v_emb = c['valid_emb'].astype(np.float32)
    single = RiskAdvisor(n_members=1, feature_set=args.ra_features, pca_dim=args.pca_dim)
    single.fit(c['valid_logits'], c['valid_correct'], v_emb)
    ens = RiskAdvisor(n_members=args.ra_members, feature_set=args.ra_features, pca_dim=args.pca_dim)
    ens.fit(c['valid_logits'], c['valid_correct'], v_emb)

    emb = c['emb']  # float16; the advisor converts it chunk by chunk
    conf = msp(c['logits'])
    risk = ens.predict_risk(c['logits'], emb)
    scores = {
        'msp': conf,
        'ra_single': single.score(c['logits'], emb),
        'ra_ens_p_err': 1 - risk['p_err'],
        'ra_ens_total': -risk['total_risk'],
    }
    signals = {'1-msp': 1 - conf, 'p_err': risk['p_err'], 'total_risk': risk['total_risk'],
               'epistemic': risk['epistemic'], 'aleatoric': risk['aleatoric']}
    correct, corr, sev = c['correct'], c['corruption'], c['severity']

    # 76 groups (clean + corruption x severity), then pooled per severity as corruption='all'
    groups = [('clean', 0)] + [(n, s) for n in args.corruptions for s in range(1, 6)]
    pooled = [('all', s) for s in range(6)]
    rows, mon = [], []
    for name, s in groups + pooled:
        if name == 'all':
            mask = sev == s
        else:
            mask = (corr == name) & (sev == s)
        for scorer in SCORERS:
            row = {'tag': args.tag, 'corruption': name, 'severity': s, 'scorer': scorer}
            row.update({k: round(v, 2) for k, v in evaluate(scores[scorer][mask], correct[mask]).items()})
            rows.append(row)
        if name != 'all':
            m = {'tag': args.tag, 'corruption': name, 'severity': s}
            m.update(monitor.group_row({k: v[mask] for k, v in signals.items()}, correct[mask]))
            mon.append(m)

    for row in rows:
        if row['corruption'] == 'all':
            print('sev {}  {:<13} acc {acc:.2f}  auroc {auroc:.2f}  '
                  'aurc {aurc:.2f}  fpr95 {fpr95:.2f}'.format(row['severity'], row['scorer'], **row))

    monitor.add_erel(mon, mon[0]['error_rate'])
    summary = [dict(tag=args.tag, **s) for s in monitor.summarize(mon)]
    print('label-free monitoring over {} groups ({} with Erel > {:g}):'.format(
        len(mon), summary[0]['n_trigger'], monitor.EREL_THRESHOLD))
    for s in summary:
        print('  {:<11} spearman {:+.3f}  trigger auroc {:.2f}'.format(
            s['signal'], s['spearman'], s['trigger_auroc']))

    upsert_csv(args.out, FIELDS, rows, args.tag)
    print('wrote', args.out)
    mon_path = os.path.join(out_dir, 'monitor_{}.csv'.format(args.tag))
    with open(mon_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MONITOR_FIELDS)
        w.writeheader()
        w.writerows([rounded(m, 6) for m in mon])
    print('wrote', mon_path)
    sum_path = os.path.join(out_dir, 'monitor_summary.csv')
    upsert_csv(sum_path, SUMMARY_FIELDS, [rounded(s) for s in summary], args.tag)
    print('wrote', sum_path)


if __name__ == '__main__':
    main()
