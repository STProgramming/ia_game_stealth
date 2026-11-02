"""Radial grenade impulses swept through the playable collision geometry."""
import math

import numpy as np

from .soliton_security import line_clear


BLAST_RADIUS = 3.
MAX_BLAST_PUSH = 2.1


def blast_impulse(grid, center, pos, fallback=None):
    """Return a collision-safe destination and normalized blast strength.

    This deliberately never snaps to a nearest free cell or follows a path
    around cover: an impulse stops at the first obstructed point on its ray.
    """
    center = np.asarray(center, dtype=np.float32)
    pos = np.asarray(pos, dtype=np.float32)
    delta = pos - center
    distance = float(np.linalg.norm(delta))
    if distance >= BLAST_RADIUS or not line_clear(grid, center, pos):
        return pos.copy(), 0.
    strength = max(0., 1. - distance / BLAST_RADIUS)
    if distance > 1e-6:
        direction = delta / distance
    else:
        direction = np.asarray([1., 0.] if fallback is None else fallback, dtype=np.float32)
        length = float(np.linalg.norm(direction))
        direction = direction / length if length > 1e-6 else np.array([1., 0.], dtype=np.float32)
    travel = MAX_BLAST_PUSH * strength
    steps = max(1, math.ceil(travel / .05))
    height, width = grid.shape
    goal = pos.copy()
    for amount in np.linspace(travel / steps, travel, steps):
        candidate = pos + direction * amount
        if not (0 <= candidate[0] <= width - 1 and 0 <= candidate[1] <= height - 1):
            break
        if not line_clear(grid, pos, candidate):
            break
        goal = candidate
    return goal.astype(np.float32), strength


def knockback_player(runtime, center, grenade_origin):
    world, grid = runtime._combat_world()
    goal, strength = blast_impulse(grid, center, world.player_pos, center - grenade_origin)
    if strength <= 0:
        return
    world.player_pos = goal
    world.player_velocity[:] = 0.
    if hasattr(world, 'player_current_speed'):
        world.player_current_speed = 0.
    if strength >= .4:
        # The existing fall/get-up timing is reused without adding melee damage.
        runtime.player_actions.knock_down(runtime.player_vitals)
        runtime.takedown = None
    else:
        runtime.player_vitals.hit_stun_remaining = max(
            runtime.player_vitals.hit_stun_remaining, .15 + .6 * strength)


def knockback_actor(runtime, uid, actor, vitals, center, grenade_origin):
    _, grid = runtime._combat_world()
    goal, strength = blast_impulse(grid, center, actor.pos, center - grenade_origin)
    if strength <= 0:
        return
    actor.pos = goal
    actor.aim_timer = 0.
    vitals.hit_stun_remaining = max(vitals.hit_stun_remaining, .15 + .85 * strength)
    if hasattr(actor, 'fire_goal'):
        actor.fire_goal = None
    if hasattr(runtime, 'env') and uid == 'G0':
        runtime.env.guard_pos = goal.copy()
    runtime.navigator.paths.pop(uid, None)
    motion = getattr(runtime, '_motions', {}).get(uid)
    if motion is not None:
        motion.speed = 0.
    attack = runtime._melee_attacks.get(uid)
    if attack is not None:
        attack.windup = 0.
    if runtime.takedown and runtime.takedown.get('guard_id') == uid:
        runtime.takedown = None
    if hasattr(runtime, '_cancel_guard_care'):
        runtime._cancel_guard_care(uid)
    else:
        runtime._context_medical_jobs.pop(uid, None)
