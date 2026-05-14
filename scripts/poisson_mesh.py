#!/usr/bin/env python3
"""
Convert a colored point cloud (PLY) into a triangle mesh via Poisson
surface reconstruction. Designed to run in the DA3 venv where Open3D
lives (~/.venvs/da3/bin/python).

Usage:
    poisson_mesh.py <input.ply> <output.ply> [--depth 9] [--keep-fraction 0.05]

The default `--depth 9` is a good middle ground for room-scale scenes
captured from a drone (~3-5 GB point clouds). Lower for smaller objects,
higher for very dense clouds. `--keep-fraction` drops the lowest-density
N% of vertices — Poisson interpolates aggressively at edges and those
extrapolated regions look like floating sheets.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input',  type=Path, help='Input colored PLY')
    ap.add_argument('output', type=Path, help='Output mesh PLY')
    ap.add_argument('--depth', type=int, default=9,
                    help='Octree depth — higher = more detail but slower')
    ap.add_argument('--keep-fraction', type=float, default=0.05,
                    help='Drop the lowest-density N%% of vertices (sheet artefacts)')
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f'Input not found: {args.input}', file=sys.stderr)
        return 1

    print(f'[poisson] loading {args.input}', flush=True)
    pcd = o3d.io.read_point_cloud(str(args.input))
    n_in = len(pcd.points)
    if n_in == 0:
        print('Input point cloud is empty', file=sys.stderr)
        return 2
    print(f'[poisson] {n_in:,} points loaded', flush=True)

    # Normals are required for Poisson. Estimate from the local
    # neighborhood (KNN), then orient consistently along view direction.
    print('[poisson] estimating normals…', flush=True)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    print(f'[poisson] reconstructing (depth={args.depth})…', flush=True)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth)

    # Trim low-density vertices — Poisson extrapolates aggressively and
    # those bits manifest as floating sheets. Cutting the lowest 5%
    # cleans them up without affecting the well-supported surfaces.
    densities = np.asarray(densities)
    if 0.0 < args.keep_fraction < 1.0 and densities.size > 0:
        cutoff = np.quantile(densities, args.keep_fraction)
        mask = densities < cutoff
        mesh.remove_vertices_by_mask(mask)
        print(f'[poisson] trimmed {mask.sum():,} low-density verts '
              f'(cutoff={cutoff:.4f})', flush=True)

    mesh.compute_vertex_normals()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.output), mesh,
                               write_vertex_colors=True,
                               write_vertex_normals=True)
    print(f'[poisson] wrote {len(mesh.vertices):,} verts, '
          f'{len(mesh.triangles):,} tris → {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
