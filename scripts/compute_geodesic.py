"""
compute_geodesic.py

This script computes the geodesic distance between two points on the Earth's surface.
"""

from dataclasses import dataclass
from pathlib import Path

from geographiclib.geodesic import Geodesic
from jaxtyping import jaxtyped
import numpy as np
from typeguard import typechecked
import pandas as pd
from tqdm import tqdm
import tyro
import vptree


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

    
    # Geodesic distance from each generated point to the nearest ground truth point
    d_gen2gt = 0.0
    tree = vptree.VPTree(gt_pts, geoddist)
    for gen_pt in tqdm(gen_pts):
        result = tree.get_nearest_neighbor(gen_pt)
        d = result[0]
        d_gen2gt += d
    d_gen2gt /= len(gen_pts)

    # Geodesic distance from each ground truth point to the nearest generated point
    d_gt2gen = 0.0
    tree = vptree.VPTree(gen_pts, geoddist)
    for gt_pt in tqdm(gt_pts):
        result = tree.get_nearest_neighbor(gt_pt)
        d = result[0]
        d_gt2gen += d
    d_gt2gen /= len(gt_pts)

    d_total = d_gen2gt + d_gt2gen

    print(f"Average geodesic distance: {d_total / len(gen_pts)}")


if __name__ == "__main__":
    main(
        tyro.cli(Args)   
    )
