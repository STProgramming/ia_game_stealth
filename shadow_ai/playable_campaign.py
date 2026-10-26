"""Three fictional missions and their gameplay-only safety requirements."""
import numpy as np
from .playable_facility import ROOMS, build_research_grid

MAP_WIDTH, MAP_HEIGHT = 64, 40
ARCHIVE_DOORS = ((23,3), (24,3), (23,8), (24,8))


def annex_rooms(level):
    names = {1: ('SALA RIUNIONI', 'LOGISTICA', 'ARCHIVIO PERSONALE'),
             2: ('SVILUPPO R&D', 'SALA SERVER', 'SALA PROGETTI'),
             3: ('LAB ANALISI', 'DEPOSITO REAGENTI', 'CONTROLLO QUALITÀ')}[level]
    return [(50,1,62,12,names[0],[(50,6),(50,7),(55,12),(56,12)]),
            (50,19,62,34,names[1],[(50,24),(50,25),(55,34),(56,34)]),
            (18,30,39,38,names[2],[(23,30),(24,30),(39,34),(39,35)])]

MISSIONS = {
    1: dict(title="ATRIO E RECEPTION", objective="Trova la scheda di sicurezza nell’ufficio e raggiungi l’uscita R&D.",
            names=["Scheda di sicurezza"], points=[(4,4)], indices=[0]),
    2: dict(title="UFFICI R&D", objective="Apri l’archivio con la tessera della prima missione. Recupera dossier e prove digitali, poi raggiungi i laboratori.",
            names=["Dossier R&D", "Prove digitali"], points=[(24,6),(34,5)], indices=[0,1]),
    3: dict(title="LABORATORI", objective="Indossa la protezione, trova la provetta sterile e preleva il campione. Poi esfiltra.",
            names=["Campione biologico"], points=[(44,22)], indices=[2]),
}

def mission_room_bounds(level):
    if level == 1:
        rooms = [(1,1,9,10),(21,3,26,8),(30,1,46,12),(6,18,15,25),(29,17,37,26)]
    else:
        rooms = [(x0,y0,x1,y1) for x0,y0,x1,y1,_,_ in ROOMS]
    return rooms + [room[:4] for room in annex_rooms(level)]

def mission_layout(level):
    grid = (build_research_grid(MAP_WIDTH, MAP_HEIGHT) if level == 3
            else np.zeros((MAP_HEIGHT,MAP_WIDTH),dtype=np.uint8))
    rooms = ([(1,1,9,10,"UFFICI",[(9,5),(9,6),(6,10)]),
              (21,3,26,8,"AMMINISTRAZIONE",[(23,8),(24,8)]),
              (30,1,46,12,"SALA CONGRESSI",[(34,12),(35,12),(46,9),(46,10)]),
              (6,18,15,25,"SERVIZI",[(10,18),(11,18),(15,22)]),
              (29,17,37,26,"PRIMO SOCCORSO",[(29,20),(29,21),(33,17)])]
             if level == 1 else
             [(1,1,9,10,"SICUREZZA",[(9,5),(9,6),(6,10)]),
              (21,3,26,8,"ARCHIVIO R&D",[(23,3),(24,3),(23,8),(24,8)]),
              (30,1,38,10,"UFFICI R&D",[(34,10),(35,10),(38,6),(38,7)]),
              (40,1,46,12,"DIREZIONE R&D",[(40,8),(40,9),(43,12),(44,12)]),
              (6,18,15,25,"SERVIZI",[(10,18),(11,18),(15,22)]),
              (29,17,37,26,"PRIMO SOCCORSO",[(29,20),(29,21),(33,17)]),
              (41,18,46,25,"ACCESSO LAB",[(43,18),(44,18),(43,25),(44,25)])])
    if level == 3:
        rooms = ROOMS
    rooms = rooms + annex_rooms(level)
    for x0,y0,x1,y1,_,doors in rooms:
        grid[y0:y1+1,x0] = grid[y0:y1+1,x1] = 1
        grid[y0,x0:x1+1] = grid[y1,x0:x1+1] = 1
        for x,y in doors:
            grid[y,x] = 0
    grid[16,17:23] = 1
    grid[25,0:6] = 1
    grid[25:29,5] = 1
    if level == 1:
        for y in (4,7):
            for x in (33,36,39,42):
                grid[y,x] = 1  # Congress seating blocks, with aisles between.
    else:
        grid[19,19:25] = 1
        grid[23,19:25] = 1  # Research workstations.
    zones = [(name,x0+1,y0+2) for x0,y0,_,_,name,_ in rooms]
    return grid, zones + [("ATRIO" if level == 1 else "CORRIDOIO R&D",12,12),("RECEPTION" if level == 1 else "POSTAZIONI R&D",17,18)]

def campaign_supplies(navigator,level):
    placements = [("MAGAZINE",(13,24),1),("BANDAGE",(7,21),2),
                  ("MEDKIT",(32,20),1),("SMOKE",(42,9),1),("SNACK",(7,22),2),
                  ("MAGAZINE",(56,6),1),("BANDAGE",(55,27),1),("SNACK",(30,34),1)]
    if level == 3:
        placements += [("HAZMAT",(32,23),1),("SAMPLE_VIAL",(39,17),1)]
    return [dict(id=f'supply-{i}',kind=kind,pos=navigator.goal(np.array(point,dtype=np.float32)),
                 amount=amount,source='MAP',fixed=kind=='MEDKIT',zone='INFERMERIA' if kind=='MEDKIT' else 'PREPARAZIONE' if kind in {'HAZMAT','SAMPLE_VIAL'} else 'SERVIZI')
            for i,(kind,point,amount) in enumerate(placements)]
