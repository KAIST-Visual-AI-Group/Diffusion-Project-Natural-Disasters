"""
compute_geodesic.py

This script computes the geodesic distance between two points on the Earth's surface.
"""

from dataclasses import dataclass
from pathlib import Path

from geographiclib.geodesic import Geodesic
from jaxtyping import jaxtyped, Shaped
import numpy as np
from typeguard import typechecked
import pandas as pd
from tqdm import tqdm
import tyro

from scripts.utils import VPTree


GT_PTS_PATH: Path = Path("./data/volcano.tsv")


@dataclass
class Args:
   
    gen_pts_path: Path
    """Path to the generated points file."""

def geoddist(p1, p2):
    """
    p1: [lon1, lat1] in degrees
    p2: [lon2, lat2] in degrees
    """
    return Geodesic.WGS84.Inverse(p1[1], p1[0], p2[1], p2[0])['s12']


@jaxtyped(typechecker=typechecked)
def main(args: Args) -> None:

    assert args.gen_pts_path.exists(), f"File {args.gen_pts_path} does not exist."
    assert GT_PTS_PATH.exists(), f"File {GT_PTS_PATH} does not exist."

    # Load the generated points
    if args.gen_pts_path.suffix in [".tsv", ".csv"]:
        gen_pts = pd.read_csv(args.gen_pts_path, sep="\t").to_numpy()
    elif args.gen_pts_path.suffix == ".npy":
        gen_pts = np.load(args.gen_pts_path)
    else:
        raise ValueError(f"Unsupported file format: {args.gen_pts_path.suffix}")

    # Load the ground truth points
    if GT_PTS_PATH.suffix in [".tsv", ".csv"]:
        gt_pts = pd.read_csv(GT_PTS_PATH, sep="\t").to_numpy()
    elif GT_PTS_PATH.suffix == ".npy":
        gt_pts = np.load(GT_PTS_PATH)
    else:
        raise ValueError(f"Unsupported file format: {GT_PTS_PATH.suffix}")

    # Make sure that points are unique in each set
    gt_pts = np.unique(gt_pts, axis=0)
    gen_pts = np.unique(gen_pts, axis=0)

    # Compute coverage.
    # Coverage: Fraction of ground truth points that are the nearest neighbors of at least one generated point.
    cov = compute_cov(gt_pts, gen_pts)

    # Compute miminum matching distance.
    # MMD: Average distance from each ground truth point to its nearest neighbor in the generated points.
    mmd = compute_mmd(gt_pts, gen_pts)

    print(f"COV: {cov:.5f} | MMD: {mmd:.5f}")

@jaxtyped(typechecker=typechecked)
def compute_cov(
    gt_pts: Shaped[np.ndarray, "N 2"],
    gen_pts: Shaped[np.ndarray, "N 2"],
) -> float:
    gt_nns = []

    tree = VPTree(gt_pts, geoddist, point_ids=np.arange(len(gt_pts)))
    for gen_pt in tqdm(gen_pts):
        result = tree.get_nearest_neighbor(gen_pt)
        _, _, point_id = result
        gt_nns.append(point_id)
    gt_nns = set(gt_nns)

    cov = len(gt_nns) / len(gt_pts)
    return cov

@jaxtyped(typechecker=typechecked)
def compute_mmd(
    gt_pts: Shaped[np.ndarray, "N 2"],
    gen_pts: Shaped[np.ndarray, "N 2"],
) -> float:
    mmd = 0.0

    tree = VPTree(gen_pts, geoddist, point_ids=np.arange(len(gen_pts)))
    for gt_pt in tqdm(gt_pts):
        dist, _, _ = tree.get_nearest_neighbor(gt_pt)
        mmd += dist
    mmd /= len(gt_pts)

    return mmd

if __name__ == "__main__":
    main(
        tyro.cli(Args)   
    )
