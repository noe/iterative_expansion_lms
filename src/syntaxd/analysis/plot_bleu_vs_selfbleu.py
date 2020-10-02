import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pathlib
import re
import seaborn as sns
from typing import List, Tuple


class Point:
    def __init__(self, temp: float, self_bleu: float, test_bleu: float):
        self.temp = temp
        self.self_bleu = self_bleu
        self.test_bleu = test_bleu


def read_float(path: pathlib.Path):
    with path.open(encoding='utf-8') as f:
        return float(f.read())


def read_temp_points(model_dir: pathlib.Path, order: int) -> List[Point]:
    template = re.compile('temp_(\d\.\d+)(_.+)?')
    temps = sorted(set(float(m[1])
                       for m in (template.match(f.name) for f in model_dir.iterdir())
                       if bool(m)))
    points = []
    for temp in temps:
        test_bleu = read_float(model_dir / "temp_{}_testbleu{}".format(temp, order))
        self_bleu = read_float(model_dir / "temp_{}_selfbleu{}".format(temp, order))
        points.append(Point(temp, self_bleu, test_bleu))
    return points


def read_single_point(model_dir: pathlib.Path, order: int) -> Tuple[float, float]:
    model_name = model_dir.name
    test_bleu = read_float(model_dir / "{}_testbleu{}".format(model_name, order))
    self_bleu = read_float(model_dir / "{}_selfbleu{}".format(model_name, order))
    return test_bleu, self_bleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', type=int, required=True)
    parser.add_argument('--limits', type=float, nargs='+', required=False)
    parser.add_argument('dir', type=str)
    args = parser.parse_args()

    dir = pathlib.Path(args.dir)

    line_models = [('itex_w', 'Iterative Expansion LM (w)', 'green', '-', 4),
                   ('itex_sw', 'Iterative Expansion LM (sw)', 'red', '-', 3),
                   ('awdlstm_w', 'AWD-LSTM (w)', 'purple', '--', 2),
                   ('awdlstm_sw', 'AWD-LSTM (sw)', 'gray', '--', 1),
                   ]

    point_models = [('train_sample', 'train sample', -70., -8.),
                    ('valid_sample', 'valid sample', 5., 0.),
                    #('srilm', 'SRILM', 2., 0.),
                    ]

    temps = [(0.7, 'v'),
             (0.8, 's'),
             (0.9, 'D'),
             (1.0, '^'),
             (1.2, 's')
             ]

    #sns.set(font_scale=1.9)
    sns.set_style("whitegrid")

    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax = fig.add_subplot(111)

    lines = []
    for model, label, color, line_style, z_order in line_models:
        model_dir = dir / model
        points = read_temp_points(model_dir, args.order)
        x = [-p.test_bleu for p in points]
        y = [p.self_bleu for p in points]
        line, = ax.plot(x, y, color=color, alpha=1.0, label=label,
                        zorder=z_order, linewidth=1, linestyle=line_style)
        lines.append(line)
        for t, symbol in temps:
            t_x = [-p.test_bleu for p in points if p.temp == t]
            t_y = [p.self_bleu for p in points if p.temp == t]
            plt.scatter(t_x, t_y, c=color, edgecolors='k', s=30,
                        zorder=len(line_models) + z_order, marker=symbol, linewidth=0.8)

    for model, label, x_offset, y_offset in point_models:
        model_dir = dir / model
        test_bleu, self_bleu = read_single_point(model_dir, args.order)
        x = -test_bleu
        y = self_bleu
        ax.plot(x, y, 'o', color='k', markersize=3, zorder=2 * len(line_models))
        ax.annotate(label, xy=(x, y), textcoords='offset points', xytext=(x_offset, y_offset), size=12)

    ax.set_xlabel('Negative BLEU-{}'.format(args.order))
    ax.set_ylabel('self BLEU-{}'.format(args.order))
    if args.limits:
        plt.axis(args.limits)
    line_legend = plt.legend(handles=lines, markerfirst=False, frameon=True, loc='upper right')
    plt.gca().add_artist(line_legend)
    markers = [mlines.Line2D([], [], color='w', markeredgecolor='k', marker=m, linestyle='None',
                             linewidth=0.8, markersize=5, label='$\\tau={}$'.format(t))
               for t, m in temps]
    plt.legend(handles=markers, frameon=True, loc='center right')
    plt.savefig('bleu_sbleu_{}.pdf'.format(args.order), bbox_inches='tight')


if __name__ == '__main__':
    main()
