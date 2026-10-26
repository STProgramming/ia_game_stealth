"""Physical blood deposition and shared, visibility-limited evidence tracking."""
from .soliton_security import BloodMark, can_see


class BloodTrail:
    def __init__(self):
        self.pending = 0.0

    def deposit(self, marks, old, pos, loss, new_id):
        # More lost blood produces more drops, including while standing still.
        self.pending += loss
        count = min(32, int(self.pending / .0008))
        if not count:
            return
        self.pending -= count * .0008
        for index in range(count):
            point = old + (pos - old) * ((index + 1) / count)
            marks.append(BloodMark(new_id(), point.copy()))


def observe_blood(marks, grid, pos, angle, observer, after=-1, obscured=None):
    # A radio-shared discovery is known to the whole squad. Never use the
    # player's position or a hidden newer drop to choose the next waypoint.
    visible = [m for m in marks if not m.discovered_by
               and can_see(grid, pos, angle, m.pos, 6.5, 105.0)
               and (obscured is None or not obscured(pos, m.pos))]
    candidates = [m for m in visible if m.mark_id > after]
    for mark in visible:
        mark.discovered_by.add(observer)
    return max(candidates, key=lambda m: m.mark_id) if candidates else None
