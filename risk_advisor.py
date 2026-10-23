"""Risk Advisor meta-model, after Lahoti et al., "Responsible Model Deployment via
Model-Agnostic Uncertainty Learning".

An ensemble of stochastic gradient boosted trees predicts whether the base model's
prediction is wrong (error = 1). Each member is fit on the correct/incorrect label
and p(error) = 1 - p(correct); this is the same model as fitting on the error label,
but keeps n_members=1 bit-identical to the earlier single-GBT advisor. The members differ only in random_state; with
subsample < 1 each member sees a different row sample, which gives the ensemble
the diversity needed for the uncertainty decomposition:

    p_err      mean member error probability            (model uncertainty)
    aleatoric  mean of member binary entropies          (expected entropy)
    epistemic  entropy of p_err - aleatoric             (mutual information)
    total_risk p_err + aleatoric + epistemic

Entropies are in bits, so all three terms lie in [0, 1].

Deviation from Paper 2: the original fits the advisor on the base model's training
errors. We fit it on the held-out validation split (5,000 CIFAR-10 train images that
the base model never trained on), because a ResNet-18 trained for 201 epochs makes
almost no errors on its own training set.
"""
import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier

FEATURE_SETS = ('logits', 'logits+emb')


def features(logits):
    logits = np.asarray(logits, dtype=np.float64)
    p = softmax(logits, axis=1)
    ps = np.sort(p, axis=1)[:, ::-1]
    ls = np.sort(logits, axis=1)[:, ::-1]
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)
    return np.column_stack([
        ps[:, :3],
        ps[:, 0] - ps[:, 1],
        entropy,
        ls[:, 0],
        ls[:, 0] - ls[:, 1],
        logits.std(axis=1),
        logsumexp(logits, axis=1),
    ])


def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class RiskAdvisor:
    def __init__(self, n_members=10, feature_set='logits', pca_dim=32, random_state=0, **kwargs):
        if feature_set not in FEATURE_SETS:
            raise ValueError('feature_set must be one of {}'.format(FEATURE_SETS))
        self.params = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8)
        self.params.update(kwargs)
        if n_members > 1 and self.params['subsample'] >= 1:
            raise ValueError('an ensemble needs subsample < 1, otherwise all members are identical')
        self.n_members = n_members
        self.feature_set = feature_set
        self.pca_dim = pca_dim
        self.random_state = random_state
        self.pca = None
        self.members = []

    def _design(self, logits, emb):
        x = features(logits)
        if self.feature_set == 'logits+emb':
            if emb is None:
                raise ValueError("feature_set 'logits+emb' needs the penultimate embedding")
            x = np.column_stack([x, self.pca.transform(np.asarray(emb, dtype=np.float64))])
        return x

    def fit(self, logits, correct, emb=None):
        """Fit on validation-set logits (and embeddings); see the module docstring."""
        correct = np.asarray(correct).astype(int)
        if len(np.unique(correct)) < 2:
            raise ValueError('need both correct and incorrect samples to fit')
        if self.feature_set == 'logits+emb':
            if emb is None:
                raise ValueError("feature_set 'logits+emb' needs the penultimate embedding")
            emb = np.asarray(emb, dtype=np.float64)
            dim = min(self.pca_dim, emb.shape[0], emb.shape[1])
            self.pca = PCA(n_components=dim, random_state=self.random_state).fit(emb)
        x = self._design(logits, emb)
        self.members = [
            GradientBoostingClassifier(random_state=self.random_state + m, **self.params).fit(x, correct)
            for m in range(self.n_members)
        ]
        return self

    def member_probs(self, logits, emb=None, chunk=50000):
        """(n_members, n_samples) error probabilities, in chunks to bound memory."""
        out = []
        for i in range(0, len(logits), chunk):
            x = self._design(logits[i:i + chunk], None if emb is None else emb[i:i + chunk])
            out.append(np.stack([1 - m.predict_proba(x)[:, list(m.classes_).index(1)]
                                 for m in self.members]))
        return np.concatenate(out, axis=1)

    def predict_risk(self, logits, emb=None):
        probs = self.member_probs(logits, emb)
        p_err = probs.mean(axis=0)
        aleatoric = binary_entropy(probs).mean(axis=0)
        epistemic = np.maximum(binary_entropy(p_err) - aleatoric, 0.0)
        return {
            'p_err': p_err,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'total_risk': p_err + aleatoric + epistemic,
        }

    def score(self, logits, emb=None):
        """Confidence for fp_metrics.evaluate (higher = more likely correct): 1 - p_err."""
        return 1 - self.predict_risk(logits, emb)['p_err']
