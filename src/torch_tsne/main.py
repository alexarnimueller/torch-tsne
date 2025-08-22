import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_tsne import tsne

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x",
        "--xfile",
        type=str,
        default=f"{PKG_ROOT}/data/mnist2500_X.txt",
        help="file name of feature stored",
    )
    parser.add_argument(
        "-y",
        "--yfile",
        type=str,
        default=f"{PKG_ROOT}/data/mnist2500_labels.txt",
        help="file name of label stored",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"{PKG_ROOT}/examples/pytorch.png",
        help="output image path",
    )
    parser.add_argument(
        "-i", "--iter", type=int, help="number of iterations", default=1000
    )
    parser.add_argument("-p", "--perplex", type=float, help="perplexity", default=30.0)
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=500.0)
    parser.add_argument(
        "-e", "--exager", type=float, help="early exageration", default=4.0
    )
    parser.add_argument("-s", "--seed", type=int, help="random seed", default=42)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose output"
    )

    args = parser.parse_args()

    X = torch.Tensor(np.loadtxt(args.xfile))
    labels = np.loadtxt(args.yfile).tolist()

    # confirm that x file get same number point than label file
    assert len(X) == len(labels), "different number of datapoints and labels"

    with torch.no_grad():
        Y = tsne(
            X,
            max_iter=args.iter,
            iter_explore=args.iter // 10,
            perplexity=args.perplex,
            lr=args.lr,
            early_exager=args.exager,
            seed=args.seed,
            verbose=args.verbose,
        )

    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(args.output, dpi=300)
    if args.verbose:
        print(f"Saved tSNE output to {args.output}")
