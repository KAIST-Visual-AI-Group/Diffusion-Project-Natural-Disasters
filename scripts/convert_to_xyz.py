
from pathlib import Path

import igl
import numpy as np
import pandas as pd


def spherical_to_xyz(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)

def main() -> None:

    # Read training data
    gt_pts_path: Path = Path("./data/volcano.tsv")
    assert gt_pts_path.exists(), f"File {gt_pts_path} does not exist."

    # Convert spherical to cartesian coordinates
    gt_theta_phi = pd.read_csv(gt_pts_path, sep="\t").to_numpy()
    gt_xyz = spherical_to_xyz(
        r=1.0,
        theta=gt_theta_phi[:, 0] * np.pi / 180,
        phi=gt_theta_phi[:, 1] * np.pi / 180,
    )

    # Write to .off file
    igl.write_off(
        "data/volcano.off",
        gt_xyz,
        np.zeros((1, 3), dtype=int),
        np.ones_like(gt_xyz),
    )


if __name__ == "__main__":
    main()
