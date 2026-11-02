"""Distribute sentries around known sensitive sites instead of on their centre."""
import math

import numpy as np

from .guard_medical import CARE_STATES
from .soliton_security import line_clear


class RuntimeSecurityDistribution:
    @staticmethod
    def _security_heading(post):
        direction = post['anchor'] - post['goal']
        return math.atan2(float(direction[1]), float(direction[0]))

    def _reset_security_distribution(self):
        self._security_roster = None
        self._security_posts = {}
        self._swat_security_posts = {}
        self._security_geometry = None
        self._security_phase = None
        self._security_layouts = {}
        self._security_perimeters = {}

    def _refresh_security_geometry(self):
        grid = self.env.grid
        geometry = (id(self.navigator), grid.shape, grid.tobytes())
        phase = int(self.env.steps * self.game_dt / 8.) % 4
        changed = geometry != getattr(self, '_security_geometry', None)
        if changed or phase != getattr(self, '_security_phase', None):
            self._security_layouts = {}
            self._security_perimeters = {}
            self._security_geometry, self._security_phase = geometry, phase
        if changed:
            # A door or wall can also make a previously unreachable site
            # reachable, even without a roster change or a new game tick.
            self._security_roster = None
            self._security_posts = {}

    def _security_sites(self):
        """Return radio knowledge, never the hidden player's inventory/health."""
        env = self.env
        sites = [(f'INTEL:{index}', 'PROTECT INTEL', point.copy())
                 for index, point in enumerate(env._intel)
                 if env.episode_has_intel[index] and env.known_intel_present[index]]
        watch = self.medical_watch
        if watch.wounded_by:
            sites.extend((f'MEDKIT:{item["id"]}', 'WATCH MEDKIT', item['pos'].copy())
                         for item in self.pickups
                         if item.get('fixed') and item['kind'] == 'MEDKIT'
                         and item['id'] not in watch.consumed
                         and not item.get('reserved_by'))
        active = np.asarray(env.episode_has_intel, dtype=bool)
        if active.any() and not np.asarray(env.known_intel_present)[active].any():
            sites.append(('EXFIL', 'SECURE EXFIL', env._exit.copy()))
        return sites

    def _security_regulars(self):
        return sorted((guard for guard in self._active_guards()
                       if guard.guard_id.startswith('G')
                       and guard.state not in CARE_STATES
                       and guard.state not in {'LEAVING', 'EXITED', 'WITHDRAW'}),
                      key=lambda guard: guard.guard_id)

    def _security_allocations(self, sites, guards):
        self._refresh_security_geometry()
        # A site/roster change rebalances the whole watch. Ordinary movement and
        # different update orders cannot make actors continually swap posts.
        signature = (tuple((key, tuple(point)) for key, _, point in sites),
                     tuple(guard.guard_id for guard in guards))
        if signature != getattr(self, '_security_roster', None):
            self._security_layouts = {}
            self._security_perimeters = {}
            assignments, loads = {}, {key: 0 for key, _, _ in sites}
            for guard in guards:
                reachable = [(loads[key], self.navigator.path_length(guard.pos, point), key)
                             for key, _, point in sites]
                reachable = [entry for entry in reachable if np.isfinite(entry[1])]
                if reachable:
                    key = min(reachable)[2]
                    assignments[guard.guard_id] = key
                    loads[key] += 1
            self._security_roster, self._security_posts = signature, assignments
        return self._security_posts

    def _security_perimeter(self, anchor, rank, count, occupied=()):
        # Patrol around the object, leaving the object itself free. A slow,
        # shared phase moves the perimeter watch without collapsing its slots.
        self._refresh_security_geometry()
        phase = self._security_phase
        cache_key = (tuple(anchor), rank, count, tuple(tuple(point) for point in occupied))
        if cache_key in self._security_perimeters:
            cached = self._security_perimeters[cache_key]
            return None if cached is None else cached.copy()
        angle = math.tau * rank / max(count, 1) + phase * math.pi / 2.
        radius = max(2., min(4., count * .34))
        candidates = []
        for radius_offset in (0., .8, -.6, 1.6):
            for offset in (0., .3, -.3, .65, -.65, 1., -1., math.pi):
                heading = angle + offset
                raw = anchor + np.array([math.cos(heading), math.sin(heading)], dtype=np.float32) * (radius + radius_offset)
                point = self.navigator.goal(raw)
                if (np.linalg.norm(point - anchor) < .9
                        or not line_clear(self.env.grid, anchor, point)
                        or any(np.linalg.norm(point - other) < 1.1 for other in occupied)):
                    continue
                candidates.append((abs(offset) + abs(radius_offset) * .3, point))
        goal = min(candidates, key=lambda item: item[0])[1] if candidates else None
        self._security_perimeters[cache_key] = None if goal is None else goal.copy()
        return None if goal is None else goal.copy()

    def _security_assignment(self, guard):
        """A regular guard's stable site and its current perimeter patrol slot."""
        sites, guards = self._security_sites(), self._security_regulars()
        assignments = self._security_allocations(sites, guards)
        key = assignments.get(guard.guard_id)
        site = next((site for site in sites if site[0] == key), None)
        if site is None:
            return None
        _, state, anchor = site
        peers = [other for other in guards if assignments.get(other.guard_id) == key]
        layout_key = (key, tuple(anchor), tuple(peer.guard_id for peer in peers))
        if layout_key not in self._security_layouts:
            occupied, layout = [], {}
            for rank, peer in enumerate(peers):
                goal = self._security_perimeter(anchor, rank, len(peers), occupied)
                # Cache missing slots too: a cramped room should not trigger a
                # full geometric search every time an excess sentry updates.
                layout[peer.guard_id] = goal
                if goal is not None:
                    occupied.append(goal)
            self._security_layouts[layout_key] = layout
        goal = self._security_layouts[layout_key].get(guard.guard_id)
        return None if goal is None else dict(key=key, state=state, anchor=anchor.copy(), goal=goal.copy())

    def _swat_security_assignment(self, guard):
        """At most one SWAT covers an intel site lacking a regular sentry."""
        sites = self._security_sites()
        guards = self._security_regulars()
        occupied = {post['key'] for other in guards
                    if not other.seeing_player
                    and other.state not in {'CHASE', 'REPOSITION', 'FLANK'}
                    and (post := self._security_assignment(other)) is not None}
        candidates = [site for site in sites if site[0].startswith('INTEL:')
                      and site[0] not in occupied]
        claims = getattr(self, '_swat_security_posts', {})
        active = {other.guard_id for other in self._active_guards()
                  if other.guard_id.startswith('S')
                  and other.state not in {'WITHDRAW', 'LEAVING', 'EXITED'}}
        valid = {key for key, _, _ in candidates}
        claims = {uid: key for uid, key in claims.items() if uid in active and key in valid}
        self._swat_security_posts = claims
        key = claims.get(guard.guard_id)
        if key is None:
            reachable = [(self.navigator.path_length(guard.pos, point), candidate_key)
                         for candidate_key, _, point in candidates
                         if candidate_key not in claims.values()]
            reachable = [entry for entry in reachable if np.isfinite(entry[0])]
            if not reachable:
                return None
            key = min(reachable)[1]
            claims[guard.guard_id] = key
        _, _, anchor = next(site for site in candidates if site[0] == key)
        goal = self._security_perimeter(anchor, 0, 1)
        if goal is None:
            claims.pop(guard.guard_id, None)
            return None
        return dict(key=key, state='SUPPORT INTEL', anchor=anchor.copy(), goal=goal.copy())

    def _withdraw_security_support(self, guard):
        getattr(self, '_swat_security_posts', {}).pop(guard.guard_id, None)
        guard.state = 'WITHDRAW'
        guard.target = (guard.exit_pos.copy() if guard.exit_pos is not None
                        else np.array([self.env.cfg.width - 1., 10.], dtype=np.float32))
        guard._security_support_remaining = 0.

    def _finish_swat_security_search(self, guard):
        """After clearing the search, briefly cover an otherwise unguarded intel."""
        post = None if getattr(guard, '_security_support_done', False) else self._swat_security_assignment(guard)
        if post is None:
            self._withdraw_security_support(guard)
            return
        guard._security_support_done = True
        guard._security_support_remaining = 12.
        guard.pause_angle = self._security_heading(post)
        guard.state, guard.target = post['state'], post['goal'].copy()

    def _step_swat_security_support(self, guard):
        if guard.state != 'SUPPORT INTEL':
            return False
        guard._security_support_remaining = max(0., getattr(guard, '_security_support_remaining', 0.) - self.game_dt)
        post = self._swat_security_assignment(guard)
        if post is None or guard._security_support_remaining <= 1e-7:
            self._withdraw_security_support(guard)
        else:
            guard.target = post['goal'].copy()
            guard.pause_angle = self._security_heading(post)
        return True
