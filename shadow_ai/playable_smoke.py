"""Smoke perception and navigable escape choices for the playable runtimes."""
import numpy as np


def cloud_data(cloud):
    if isinstance(cloud, dict):
        return cloud["pos"], cloud["radius"], cloud["ttl"]
    return np.asarray(cloud.center), cloud.radius, cloud.ttl


def smoke_density(pos, clouds):
    return max((1.0 for cloud in clouds
                for center, radius, ttl in [cloud_data(cloud)]
                if ttl > 0 and np.linalg.norm(pos - center) <= radius), default=0.)


def smoke_visibility(a, b, clouds):
    delta = b - a
    length = float(np.dot(delta, delta))
    for cloud in clouds:
        center, radius, ttl = cloud_data(cloud)
        if ttl <= 0:
            continue
        t = np.clip(np.dot(center - a, delta) / max(length, 1e-7), 0., 1.)
        if np.linalg.norm(a + delta * t - center) <= radius:
            return .05
    return 1.


def smoke_obscures(a, b, clouds, vision_range=11.5):
    return np.linalg.norm(b - a) > vision_range * smoke_visibility(a, b, clouds)


def smoke_escape(navigator, pos, clouds):
    """Choose the shortest reachable exit using only geometry and smoke."""
    if not smoke_density(pos, clouds):
        return None
    candidates = []
    x, y = navigator.cell(pos)
    # Expand locally; no hidden player position enters this decision.
    for radius in range(1, 9):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius or not navigator.free(x + dx, y + dy):
                    continue
                point = np.array([x + dx, y + dy], dtype=np.float32)
                if smoke_density(point, clouds):
                    continue
                distance = navigator.path_length(pos, point)
                if np.isfinite(distance):
                    candidates.append((distance, point))
        if candidates:
            return min(candidates, key=lambda candidate: candidate[0])[1]
    return None
