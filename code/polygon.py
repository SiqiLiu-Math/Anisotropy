"""Signed polygonal interfacial tensor used by the manuscript.

Vertices must be ordered counterclockwise. The supplied center is the mechanical
generator, not an automatically fitted circle center or geometric centroid.
The implementation does not take absolute values of signed edge distances.
All functions accept an arbitrary batch shape before the final (n, 2) axes.
"""

import numpy as np


def _geometry(vertices, center=None):
    v = np.asarray(vertices, dtype=float)
    if v.ndim < 2 or v.shape[-1] != 2 or v.shape[-2] < 3:
        raise ValueError("vertices must have shape (..., n >= 3, 2)")
    if center is not None:
        v = v - np.asarray(center, dtype=float)[..., None, :]
    nxt = np.roll(v, -1, axis=-2)
    edge = nxt - v
    length2 = np.sum(edge * edge, axis=-1)
    signed_dL = v[..., 0] * nxt[..., 1] - v[..., 1] * nxt[..., 0]
    return v, edge, length2, signed_dL


def polygon_tensor(vertices, center=None, k=1.0):
    """Return E_s = k sum_i signed_d_i L_i t_i tensor t_i."""
    v, edge, length2, signed_dL = _geometry(vertices, center)
    if not np.isscalar(k) or not np.isfinite(k) or k <= 0:
        raise ValueError("k must be a finite positive scalar")
    if not np.all(np.isfinite(v)):
        raise ValueError("Vertices and mechanical centers must be finite")
    if np.any(length2 <= 0):
        raise ValueError("A polygon contains a zero-length edge")
    if np.any(np.sum(signed_dL, axis=-1) <= 0):
        raise ValueError("Polygon boundaries must have positive area and CCW orientation")
    return k * np.einsum("...n,...ni,...nj->...ij", signed_dL / length2, edge, edge)


def polygon_metrics(vertices, center=None, k=1.0):
    """Return tensor, complex deviator W, gap, signed area and normalized gap.

The normalization is the interfacial tensor trace, not the trace after adding
the (possibly cancelling) isotropic pressure term.
"""
    tensor = polygon_tensor(vertices, center, k)
    _, _, _, signed_dL = _geometry(vertices, center)
    W = tensor[..., 0, 0] - tensor[..., 1, 1] + 2j * tensor[..., 0, 1]
    trace = np.trace(tensor, axis1=-2, axis2=-1)
    gap = np.abs(W)
    with np.errstate(divide="ignore", invalid="ignore"):
        anisotropy = gap / trace
    return dict(tensor=tensor, W=W, delta=gap, trace=trace,
                area=np.sum(signed_dL, axis=-1) / 2, anisotropy=anisotropy)


def polygon_validity(vertices, center=None, tol=1e-12):
    """Classify convex CCW polygons containing the fixed mechanical center.

The stricter geometric domain is appropriate for the model benchmarks here.
The signed algebra itself is defined more widely. No resampling is performed.
"""
    v, edge, length2, signed_dL = _geometry(vertices, center)
    scale2 = np.maximum(np.max(np.sum(v * v, axis=-1), axis=-1), 1.0)
    nondegenerate = np.all(length2 > tol * scale2[..., None], axis=-1)
    finite = np.all(np.isfinite(v), axis=(-2, -1))
    # Every nonincident vertex must lie strictly left of each directed edge.
    # Local turn signs alone would incorrectly admit some self-crossing stars.
    differences = v[..., None, :, :] - v[..., :, None, :]
    halfplanes = (edge[..., :, None, 0] * differences[..., 1]
                  - edge[..., :, None, 1] * differences[..., 0])
    n = v.shape[-2]
    incident = np.eye(n, dtype=bool) | np.roll(np.eye(n, dtype=bool), 1, axis=1)
    convex_ccw = np.all(incident | (halfplanes > tol * scale2[..., None, None]),
                        axis=(-2, -1))
    center_inside = np.all(signed_dL > tol * scale2[..., None], axis=-1)
    valid = finite & nondegenerate & convex_ccw & center_inside
    return dict(nondegenerate=nondegenerate, finite=finite,
                convex_ccw=convex_ccw, center_inside=center_inside, valid=valid)


def regular_polygon(n, radius=1.0, phase=0.0):
    theta = phase + 2 * np.pi * np.arange(n) / n
    return radius * np.column_stack((np.cos(theta), np.sin(theta)))


def single_vertex_exact(epsilon, n, radius=1.0, k=1.0):
    """Exact complex deviator for a radial shift at angle zero (real-valued).

epsilon is an absolute radial displacement. The unperturbed radius is radius.
The magnitude is the eigenvalue gap; its sign fixes the principal-axis branch.
"""
    alpha = 2 * np.pi / n
    r = 1 + np.asarray(epsilon) / radius
    return (k * radius**2 * 2 * np.sin(alpha) * (r*r - 1)
            * (r - np.cos(alpha)) / (r*r - 2*r*np.cos(alpha) + 1))
