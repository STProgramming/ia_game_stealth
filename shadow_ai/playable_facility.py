"""Runtime-only architectural layouts; training arenas stay unchanged."""
import numpy as np


ROOMS = [
    (1, 1, 9, 10, "DIREZIONE", [(9, 5), (9, 6), (6, 10)]),
    (21, 3, 26, 8, "ARCHIVIO", [(23, 3), (24, 3), (23, 8), (24, 8)]),
    (30, 1, 38, 10, "LAB CHIMICO", [(34, 10), (35, 10), (38, 6), (38, 7)]),
    (40, 1, 46, 12, "SICUREZZA", [(40, 8), (40, 9), (43, 12), (44, 12)]),
    (6, 18, 15, 25, "SERVIZI", [(10, 18), (11, 18), (15, 22)]),
    (29, 17, 37, 26, "INFERMERIA", [(29, 20), (29, 21), (33, 17)]),
    (41, 18, 46, 25, "CAMPIONI", [(43, 18), (44, 18), (43, 25), (44, 25)]),
]


def build_research_grid(width=48, height=30, compact=False):
    grid = np.zeros((height, width), dtype=np.uint8)
    if compact:
        rooms = [(7, 1, 13, 7, "SECURITY", [(10, 7)]),
                 (18, 1, 26, 7, "LAB", [(21, 7), (22, 7)]),
                 (16, 13, 24, 18, "MED", [(20, 13)]),
                 (6, 13, 12, 18, "SERVICE", [(9, 13)])]
        if width >= 48 and height >= 30:
            rooms += [(34,1,46,12,"OPERATIONS",[(34,6),(34,7),(40,12)]),
                      (30,19,46,28,"STORAGE",[(35,19),(36,19),(30,24)]),
                      (6,22,24,28,"OFFICE",[(12,22),(13,22),(24,25)])]
    else:
        rooms = ROOMS
    for x0, y0, x1, y1, _, doors in rooms:
        grid[y0:y1+1, x0] = grid[y0:y1+1, x1] = 1
        grid[y0, x0:x1+1] = grid[y1, x0:x1+1] = 1
        for x, y in doors:
            grid[y, x] = 0
    if compact:
        grid[10, 13:17] = 1  # Reception counter, open atrium around it.
        grid[4, 0:6] = 1
        grid[1:5, 5] = 1
    else:
        grid[16, 17:23] = 1  # Reception counter.
        for x, y in [(13, 12), (26, 12), (13, 16), (26, 16)]:
            grid[y, x] = 1  # Structural columns with space on all sides.
        grid[25, 0:6] = 1
        grid[25:29, 5] = 1
    return grid


def facility_zones(compact=False):
    if compact:
        return [{"name": name, "x": x, "y": y} for name, x, y in
                [("ATRIO", 9, 9), ("RECEPTION", 12, 12), ("LAB CHIMICO", 18, 3),
                 ("SICUREZZA", 7, 3), ("INFERMERIA", 17, 15), ("SERVIZI", 7, 15)]]
    return [{"name": name, "x": x0 + 1, "y": y0 + 2} for x0, y0, _, _, name, _ in ROOMS] + [
        {"name": "ATRIO / RICERCA", "x": 15, "y": 11},
        {"name": "RECEPTION", "x": 17, "y": 18},
        {"name": "ACCESSO RISERVATO", "x": 30, "y": 14},
        {"name": "USCITA SICUREZZA", "x": 38, "y": 28},
    ]
