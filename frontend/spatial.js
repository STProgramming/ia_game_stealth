// The server's X/Y plane becomes X/Z in the 3D scene. Cells are centred on
// integer coordinates, exactly like the authoritative collision grid.
export function wallCells(map) {
  const cells = [];
  for (let y = 0; y < (map.grid?.length || 0); y++) {
    for (let x = 0; x < map.grid[y].length; x++) {
      if (map.grid[y][x]) cells.push({x, y});
    }
  }
  return cells;
}

export function clippedVision(agent, grid, doors = [], rays = 56) {
  const range = agent.vision_range ?? 8;
  const half = (agent.fov ?? 104) * Math.PI / 360;
  const locked = new Set(doors.filter(door => door.locked).map(door => `${door.x},${door.y}`));
  const points = [];
  for (let i = 0; i <= rays; i++) {
    const angle = (agent.angle || 0) - half + half * 2 * i / rays;
    const dx = Math.cos(angle), dy = Math.sin(angle);
    let distance = 0;
    for (let step = .08; step <= range; step += .08) {
      const x = Math.round(agent.x + dx * step), y = Math.round(agent.y + dy * step);
      if (!grid[y] || x < 0 || x >= grid[y].length || grid[y][x] || locked.has(`${x},${y}`)) break;
      distance = step;
    }
    points.push({x: agent.x + dx * distance, y: agent.y + dy * distance});
  }
  return points;
}

export function overviewHeight(width, height, aspect, fov = 45) {
  // The north-up camera is inclined by atan(1 / .72). Account for depth at
  // the far edge as well as the reduced vertical projection of the floor.
  const slope = .72, tilt = Math.atan(slope), tan = Math.tan(fov * Math.PI / 360);
  const distance = Math.max(width / (2 * tan * Math.max(.2, aspect)) + height * Math.sin(tilt) / 2,
    height * (Math.cos(tilt) / tan + Math.sin(tilt)) / 2);
  return Math.max(12, distance * Math.cos(tilt) * 1.15);
}

export function interactionHint(state, names = {}) {
  const player = state?.player;
  if (!player || state.status !== "RUNNING") return "";
  if (state.takedown) return `TIENI F · STORDIMENTO ${Math.round(Math.min(1, (state.takedown.progress || 0) / (state.takedown.duration || 3)) * 100)}%`;
  if (player.treatment) return `${player.station_id ? "TIENI E · " : ""}CURA IN CORSO`;
  if (player.prone) return "SPAZIO · RIALZATI";
  if (state.takedown_candidate) return state.takedown_candidate.can_lethal === false ? "TIENI F · STORDISCI" : "F · COLTELLO    TIENI F · STORDISCI";
  const near = point => Math.hypot(point.x - player.x, point.y - player.y);
  const door = (state.map?.doors || []).find(door => door.locked && near(door) < 2);
  if (door) return "E · APRI ARCHIVIO CON LA TESSERA";
  const intel = (state.map?.intel || []).find(item => !item.stolen && near(item) <= 1.2);
  if (intel) return `E · ${intel.name || "RECUPERA INTEL"}`;
  const pickup = [...(state.pickups || [])].sort((a, b) => near(a) - near(b)).find(item => near(item) <= 1.2);
  if (pickup) return pickup.fixed ? "TIENI E · POSTAZIONE MEDICA" : `E · RACCOGLI ${names[pickup.kind] || pickup.kind}`;
  return "";
}
