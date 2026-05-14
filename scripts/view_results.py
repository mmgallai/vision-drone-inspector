#!/usr/bin/env python3
"""
Pop open a PLY file in an Open3D window. Auto-detects whether the file
is a triangle mesh or a point cloud.

Designed to be launched as a non-blocking subprocess from the recon
pipeline so the user sees their results immediately when reconstruction
finishes. One window per file; close the window or Ctrl+C to dismiss.

Usage:
    view_results.py <path/to/file.ply> [<window title>]

Runs in the DA3 venv where Open3D lives (~/.venvs/da3/bin/python).
"""

import sys
from pathlib import Path

import open3d as o3d


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print('Usage: view_results.py <file.ply> [<title>]', file=sys.stderr)
        return 1

    path = Path(argv[0])
    title = argv[1] if len(argv) > 1 else path.name

    if not path.exists():
        print(f'File not found: {path}', file=sys.stderr)
        return 2

    # Try as a triangle mesh first. Most generic PLYs without faces
    # come back with 0 triangles — those we treat as point clouds.
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) > 0:
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        n_v, n_t = len(mesh.vertices), len(mesh.triangles)
        print(f'[viewer] {path.name}: mesh, {n_v:,} verts / {n_t:,} tris',
              flush=True)
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=f'{title} — mesh',
            mesh_show_back_face=True)
        return 0

    pcd = o3d.io.read_point_cloud(str(path))
    n = len(pcd.points)
    if n == 0:
        print(f'[viewer] {path.name}: empty file', file=sys.stderr)
        return 3
    print(f'[viewer] {path.name}: point cloud, {n:,} points', flush=True)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f'{title} — point cloud',
        point_show_normal=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
