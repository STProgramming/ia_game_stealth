"""Radio handover of a reported room from regular guards to the entry team."""
import math
import numpy as np
from .facility_map import FacilityMap
from .playable_navigation import PlayableNavigation
from .soliton_security import turn_towards


def inside_room(pos, room):
    x0, y0, x1, y1 = room
    return x0 - .5 <= pos[0] <= x1 + .5 and y0 - .5 <= pos[1] <= y1 + .5


class RuntimeRoomHandover:
    def _reset_room_handover(self):
        self._handover_room = None
        self._room_orders = {}
        self._outside_room_navigation = None

    def _room_doors(self, room):
        x0, y0, x1, y1 = room
        return [(np.array([x, y], dtype=np.float32), np.array(normal, dtype=np.float32))
                for x, y, normal in
                ([(x, y0, (0, 1)) for x in range(x0 + 1, x1)] +
                 [(x, y1, (0, -1)) for x in range(x0 + 1, x1)] +
                 [(x0, y, (1, 0)) for y in range(y0 + 1, y1)] +
                 [(x1, y, (-1, 0)) for y in range(y0 + 1, y1)]) if self.env._free(x, y)]

    def _ensure_room_handover(self, room):
        if room != self._handover_room:
            for guard in self._active_guards():
                if guard.guard_id in self._room_orders and guard.state in {'EVACUATE ROOM', 'REAR GUARD', 'PERIMETER PATROL', 'PROTECT INTEL'}:
                    guard.state, guard.target = 'PATROL', guard.patrol_target()
            self._reset_room_handover()
            self._handover_room = room
            world = FacilityMap.__new__(FacilityMap)
            world.width, world.height = self.env.cfg.width, self.env.cfg.height
            world.grid = self.env.grid.copy()
            x0, y0, x1, y1 = room
            world.grid[y0:y1 + 1, x0:x1 + 1] = 1
            self._outside_room_navigation = PlayableNavigation(world.grid, world.cell, world.free, world.astar, world.nearest_free_cell)
        regulars = [g for g in self._active_guards() if not g.guard_id.startswith('S')]
        for guard in regulars:
            if guard.guard_id in self._room_orders or not inside_room(guard.pos, room):
                continue
            doors = self._room_doors(room)
            if not doors:
                continue
            door, inward = min(doors, key=lambda p: self.navigator.path_length(guard.pos, p[0] - p[1] * 1.5))
            outside = self._navigable_goal(door - inward * 1.6)
            if inside_room(outside, room):
                continue
            index = int(guard.guard_id[1:]) % 3
            assignment = ('REAR GUARD', 'PERIMETER PATROL', 'PROTECT INTEL')[index]
            nav = self._outside_room_navigation
            candidates = []
            if assignment == 'PROTECT INTEL':
                candidates = [p for i, p in enumerate(self.env._intel)
                              if self.env.episode_has_intel[i] and self.env.known_intel_present[i]] + [self.env._exit]
            elif assignment == 'PERIMETER PATROL':
                candidates = [np.array(p, dtype=np.float32) for p in guard.patrol]
                if not any(not inside_room(p, room) for p in candidates):
                    side = np.array([-inward[1], inward[0]], dtype=np.float32)
                    candidates = [outside, outside - inward * 2.5 + side * 2., outside - inward * 2.5 - side * 2.]
            candidates = [nav.goal(p) for p in candidates if not inside_room(p, room)]
            candidates = [p for p in candidates if np.isfinite(nav.path_length(outside, p))]
            post = min(candidates, key=lambda p: nav.path_length(outside, p)) if candidates else outside.copy()
            self._room_orders[guard.guard_id] = dict(room=room, exit=outside, goal=post,
                assignment=assignment, evacuated=False, patrol=candidates, patrol_index=0)
            guard.state, guard.target = 'EVACUATE ROOM', outside.copy()
            self._searches.pop(guard.guard_id, None)
            self.navigator.paths.pop(guard.guard_id, None)
            task = {'REAR GUARD': 'coprici le spalle', 'PERIMETER PATROL': 'perlustra il perimetro',
                    'PROTECT INTEL': 'sorveglia i punti sensibili'}[assignment]
            self.guard_dialogue[guard.guard_id] = f'SWAT: dobbiamo entrare. {guard.guard_id}, esci e {task}. Attendiamo tutti fuori.'
            self._dialogue_timer = 8.
            self.radio_updates += 1

    def _room_blockers(self, room):
        return [g for g in self._active_guards() if not g.guard_id.startswith('S') and inside_room(g.pos, room)]

    def _step_guard_room_order(self, guard):
        order = self._room_orders.get(guard.guard_id)
        if order is None:
            return False
        active = any(unit.active and unit.state not in {'WITHDRAW', 'LEAVING', 'EXITED'} for _, unit in self.swat_units)
        reported = self._reported_room(self.team_contact) if self.team_contact is not None else None
        if not active or reported != order['room']:
            self._room_orders.pop(guard.guard_id, None)
            guard.state, guard.target = 'PATROL', guard.patrol_target()
            return False
        if (guard.seeing_player and guard.visual_confidence >= 1.
                and not inside_room(guard.pos, order['room'])):
            # An evacuated sentry engages a visible intruder outside the entry room.
            return False
        if not order['evacuated']:
            guard.state, guard.target = 'EVACUATE ROOM', order['exit'].copy()
            if not inside_room(guard.pos, order['room']):
                order['evacuated'] = True
                self.navigator.paths.pop(guard.guard_id, None)
                self.guard_dialogue[guard.guard_id] = 'Radio: sono fuori dalla stanza. Eseguo il nuovo incarico.'
                self._dialogue_timer = 7.
        if order['evacuated']:
            guard.state, guard.target = order['assignment'], order['goal'].copy()
            nav = self._outside_room_navigation
            if guard.state == 'PERIMETER PATROL' and order['patrol'] and np.linalg.norm(guard.pos - guard.target) < .3:
                order['patrol_index'] = (order['patrol_index'] + 1) % len(order['patrol'])
                order['goal'] = order['patrol'][order['patrol_index']].copy()
                guard.target = order['goal'].copy()
        else:
            nav = self.navigator
        original_nav = self.navigator
        try:
            self.navigator = nav
            guard.pos, guard.angle = self._guard_move(guard, guard.target,
                float(self.env.cfg.guard_speed) / self.game_dt * 1.35, 3.)
        finally:
            self.navigator = original_nav
        if order['evacuated'] and np.linalg.norm(guard.pos - guard.target) < .3:
            heading = math.atan2(float(guard.pos[1] - self.team_contact[1]), float(guard.pos[0] - self.team_contact[0]))
            guard.angle = turn_towards(guard.angle, heading, 2. * self.game_dt)
        return True

    def _room_handover_snapshot(self):
        if self._handover_room is None:
            return None
        return dict(room=list(self._handover_room), waiting_for=[g.guard_id for g in self._room_blockers(self._handover_room)],
                    orders={uid: dict(assignment=o['assignment'], evacuated=o['evacuated'])
                            for uid, o in self._room_orders.items()})
