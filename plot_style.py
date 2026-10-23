"""Shared figure style for slide-ready plots: white background, large text, dpi 200."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DPI = 200

# categorical slots in fixed order; a tag keeps its color across every figure
TAG_COLORS = {'baseline': '#2a78d6', 'fmfp': '#eb6834'}
EXTRA_COLORS = ['#1baf7a', '#eda100', '#6250d6', '#e34948']
TAG_MARKERS = {'baseline': 'o', 'fmfp': 's'}
EXTRA_MARKERS = ['^', 'D', 'v', 'P']
TAG_LABELS = {'baseline': 'Baseline', 'fmfp': 'FMFP'}

INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e4e3df'


def apply():
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 16,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK_2,
        'xtick.color': INK_2, 'ytick.color': INK_2,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'lines.linewidth': 2.2, 'lines.markersize': 8,
    })


def tag_style(tags):
    """tag -> (color, marker, label); known tags first, others take the next free slot."""
    out, extra = {}, 0
    for t in tags:
        if t in TAG_COLORS:
            out[t] = (TAG_COLORS[t], TAG_MARKERS[t], TAG_LABELS[t])
        else:
            i = extra % len(EXTRA_COLORS)
            out[t] = (EXTRA_COLORS[i], EXTRA_MARKERS[i], t)
            extra += 1
    return out
