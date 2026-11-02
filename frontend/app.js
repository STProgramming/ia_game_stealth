const canvas = document.getElementById("game");
const radarCanvas = document.getElementById("radar") || canvas;
const ctx = radarCanvas.getContext("2d");

const ui = {
  banner: document.getElementById("banner"),
  vision: {checked: true},
  semantic: {checked: false},
  memory: {checked: false},
  hud: document.getElementById("alertHud"),
  title: document.getElementById("alertTitle"),
  detail: document.getElementById("alertDetail"),
};

let ws = null;
let state = null;
let lastStaticMap = null;
let predictedPlayer = null;
let predictionKey = "";
let lastRenderTime = performance.now();
let lastRadarTime = -Infinity;
let radarPhase = "";
let radarPhaseSince = 0;
let radarJammedSince = null;
let radarClockSession = "";
let radarFrozenAt = null;
const keys = new Set();
let interactPulse = false;
let craftPulse = false;
let throwPulse = false;
let lethalPulse = false;
let flashPulse = false;
let smokePulse = false;
let bandagePulse = false;
let medkitPulse = false;
let divePulse = false;
let reloadPulse = false;
let fastReloadPulse = false;
let dropItemPulse = false;
let lastReloadPress = null;
let crouchToggled = false;
let selectedItem = "";
let ePressedAt = null;
let fPressedAt = null;
let fireHeld = false;
let facingAngle = 0;
let movementPointer = null;
let dragDirection = {x: 0, y: 0};
const renderedAgents = new Map();
const visionPalette = {
  NORMAL: {rgb: "146,216,248", color: "#92d8f8"},
  SUSPICIOUS: {rgb: "255,212,106", color: "#ffd46a"},
  ALERT: {rgb: "255,89,103", color: "#ff5967"},
};

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${location.host}/ws`);

  ws.onopen = () => {
    document.body.dataset.connected = "true";
    document.getElementById("connectionLabel").textContent = "SEGNALE ATTIVO";
    ui.banner.classList.add("hidden");
    canvas.focus({preventScroll: true});
    const options = new URLSearchParams(location.search);
    const mode = options.get("mode");
    if (mode === "guard" || mode === "swat") send({type: "mode", mode});
    if (options.has("scenario")) send({type: "scenario", scenario: options.get("scenario")});
  };

  ws.onclose = () => {
    document.body.dataset.connected = "false";
    document.getElementById("connectionLabel").textContent = "SEGNALE PERSO";
    keys.clear();
    state = null;
    predictedPlayer = null;
    ui.banner.textContent = "Riconnessione...";
    ui.banner.classList.remove("hidden");
    setTimeout(connect, 900);
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "fatal") {
      ui.banner.textContent = msg.message;
      ui.banner.classList.remove("hidden");
      return;
    }
    if (msg.type === "state") {
      const incoming = msg.state;

      if (incoming.map?.grid) {
        lastStaticMap = {
          width: incoming.map.width,
          height: incoming.map.height,
          grid: incoming.map.grid,
          exit: incoming.map.exit,
          intel: incoming.map.intel || [],
          semantic: incoming.map.semantic || [],
          zones: incoming.map.zones || [],
          hazards: incoming.map.hazards || [],
        };
      }

      if (lastStaticMap && incoming.map) {
        incoming.map = {
          ...lastStaticMap,
          ...incoming.map,
          grid: incoming.map.grid || lastStaticMap.grid,
          semantic: incoming.map.semantic || lastStaticMap.semantic,
          intel: incoming.map.intel || lastStaticMap.intel,
          zones: incoming.map.zones || lastStaticMap.zones,
          hazards: incoming.map.hazards || lastStaticMap.hazards,
        };
      }

      const newPredictionKey =
        `${incoming.revision}:${incoming.mode}:${incoming.scenario}:${incoming.status}`;

      if (!predictedPlayer || newPredictionKey !== predictionKey) {
        lastReloadPress = null;
        reloadPulse = fastReloadPulse = false;
        crouchToggled = false;
        selectedItem = "";
        renderedAgents.clear();
        predictedPlayer = incoming.player
          ? {x: incoming.player.x, y: incoming.player.y}
          : null;
        predictionKey = newPredictionKey;
      } else if (incoming.player && predictedPlayer) {
        // Gentle reconciliation avoids visible network snaps while keeping
        // the local player close to authoritative collision state.
        const dx = incoming.player.x - predictedPlayer.x;
        const dy = incoming.player.y - predictedPlayer.y;
        const err = Math.hypot(dx, dy);
        const gain = err > 0.8 ? 0.45 : 0.14;
        predictedPlayer.x += dx * gain;
        predictedPlayer.y += dy * gain;
      }

      state = incoming;
      window.shadowAudio?.update(incoming);
      refreshUI();
    }
  };
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function inputPacket() {
  const direction=keyDirection();
  if (direction.x || direction.y) facingAngle=Math.atan2(direction.y,direction.x);
  const origin=state?.player || {x:0,y:0};
  return {
    type: "input",
    up: keys.has("KeyW") || keys.has("ArrowUp") || dragDirection.y < -0.2,
    down: keys.has("KeyS") || keys.has("ArrowDown") || dragDirection.y > 0.2,
    left: keys.has("KeyA") || keys.has("ArrowLeft") || dragDirection.x < -0.2,
    right: keys.has("KeyD") || keys.has("ArrowRight") || dragDirection.x > 0.2,
    sprint: keys.has("ShiftLeft") || keys.has("ShiftRight"),
    crouch: crouchToggled,
    compliant: false,
    interact: interactPulse,
    interact_held: keys.has("KeyE"),
    attack_held: keys.has("KeyF"),
    dive: divePulse,
    selected_item: selectedItem,
    craft: false,
    throw: throwPulse,
    lethal: lethalPulse,
    choke: false,
    fire: keys.has("KeyH") || fireHeld,
    reload: reloadPulse,
    fast_reload: fastReloadPulse,
    drop_item: dropItemPulse,
    use_flash: false,
    use_smoke: false,
    use_bandage: false,
    use_medkit: false,
    aim_x: origin.x+Math.cos(facingAngle)*8,
    aim_y: origin.y+Math.sin(facingAngle)*8,
  };
}

setInterval(() => {
  send(inputPacket());
  selectedItem = "";
  interactPulse = false;
  craftPulse = false;
  throwPulse = false;
  lethalPulse = false;
  flashPulse = false;
  smokePulse = false;
  bandagePulse = false;
  medkitPulse = false;
  divePulse = false;
  reloadPulse = false;
  fastReloadPulse = false;
  dropItemPulse = false;
}, 16);

function inputCode(e) {
  if (e.code) return e.code;
  if (/^[a-z]$/i.test(e.key)) return `Key${e.key.toUpperCase()}`;
  if (e.key === " ") return "Space";
  if (e.key === "Shift") return "ShiftLeft";
  return e.key;
}
window.addEventListener("keydown", (e) => {
  const code = inputCode(e);
  if ([
    "KeyW","KeyA","KeyS","KeyD","ArrowUp","ArrowDown","ArrowLeft","ArrowRight",
    "ShiftLeft","ShiftRight","KeyC","Space","KeyE",
    "KeyQ","KeyF","KeyH","KeyR","KeyG"
  ].includes(code)) e.preventDefault();



  if (!e.repeat && code === "KeyE") ePressedAt = performance.now();
  if (!e.repeat && code === "KeyF") fPressedAt = performance.now();
  if (!e.repeat && code === "KeyQ") throwPulse = true;
  if (!e.repeat && code === "KeyG") dropItemPulse = true;
  if (!e.repeat && code === "KeyC" && !keys.has(code)) crouchToggled = !crouchToggled;
  if (!e.repeat && code === "Space") divePulse = true;





  if (!e.repeat && code === "KeyR") {
    const now = performance.now();
    fastReloadPulse = lastReloadPress !== null && now - lastReloadPress <= 280;
    reloadPulse = true;
    lastReloadPress = fastReloadPulse ? null : now;
  }
  if (!e.repeat && code === "Backspace") {
    e.preventDefault(); send({type:"reset", scenario: state?.scenario});
  }
  if (!e.repeat && code === "KeyV") toggleVision();
  if (!e.repeat && code === "KeyT") window.shadowScene?.toggleOverview();
  if (!e.repeat && code === "KeyM") ui.semantic.checked = !ui.semantic.checked;
  if (!e.repeat && code === "Enter") {
    if (["CAPTURED", "NEUTRALIZED"].includes(state?.status) || (state?.status === "ESCAPED" && state.level >= 3)) {
      e.preventDefault(); send({type:"reset", scenario:state.scenario});
    } else if (state?.status === "ESCAPED") {
      e.preventDefault(); send({type:"next_level"});
    }
  }
  if (!e.repeat && code === "F11") {
    e.preventDefault();
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(() => {});
  }
  keys.add(code);
  send(inputPacket());
}, true);

window.addEventListener("keyup", (e) => {
  const code=inputCode(e);
  if (code === "KeyE") { if (ePressedAt !== null && performance.now()-ePressedAt<250) interactPulse=true; ePressedAt=null; }
  if (code === "KeyF") { if (fPressedAt !== null && performance.now()-fPressedAt<220) lethalPulse=true; fPressedAt=null; }
  keys.delete(code); send(inputPacket());
}, true);
window.addEventListener("blur", () => { keys.clear(); lastReloadPress = null; stopDrag(); send(inputPacket()); });

canvas.addEventListener("pointermove", (e) => {
  if (movementPointer && movementPointer.id === e.pointerId) {
    const dx = e.clientX - movementPointer.x;
    const dy = e.clientY - movementPointer.y;
    const mag = Math.hypot(dx,dy);
    dragDirection = mag > 8 ? {x: dx/mag, y: dy/mag} : {x:0,y:0};
    send(inputPacket());
  }

});

function toggleFullscreen() {
  if (document.fullscreenElement) return document.exitFullscreen();
  return document.documentElement.requestFullscreen().catch(() => {});
}
document.getElementById("fullscreenBtn").onclick = toggleFullscreen;
function toggleVision() {
  ui.vision.checked = !ui.vision.checked;
  const button = document.getElementById("visionBtn");
  button.setAttribute("aria-pressed", String(ui.vision.checked));
  button.querySelector("span").textContent = ui.vision.checked ? "ON" : "OFF";
}
document.getElementById("visionBtn").onclick = () => { toggleVision(); canvas.focus({preventScroll: true}); };
canvas.addEventListener("pointerdown", (e) => {
  canvas.focus({preventScroll: true});
  if (e.button === 2) { fireHeld = true; canvas.setPointerCapture(e.pointerId); send(inputPacket()); return; }
  if (e.button !== 0) return;
  movementPointer = {id: e.pointerId, x: e.clientX, y: e.clientY};
  canvas.setPointerCapture(e.pointerId);
});
function stopDrag() {
  fireHeld = false;
  movementPointer = null;
  dragDirection = {x:0,y:0};
}
canvas.addEventListener("pointerup", () => { stopDrag(); send(inputPacket()); });
canvas.addEventListener("pointercancel", () => { stopDrag(); send(inputPacket()); });
canvas.addEventListener("contextmenu", e => e.preventDefault());

function resize() {
  const rect = radarCanvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  radarCanvas.width = Math.max(1, Math.floor(rect.width * dpr));
  radarCanvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
window.addEventListener("resize", resize);
new ResizeObserver(resize).observe(radarCanvas);
setTimeout(resize, 0);

function renderLoop(now) {
  const dt = Math.min(0.033, Math.max(0, (now - lastRenderTime) / 1000));
  lastRenderTime = now;
  updatePredictedPlayer(dt);
  updateRenderedAgents(dt);
  updateRadarStatus(now);
  if (window.shadowScene && !window.shadowSceneError) {
    window.shadowScene.render(state, {
      player: predictedPlayer, facing: facingAngle, displayed: displayedAgent,
      colorFor: agentPalette, vision: ui.vision.checked, now, itemNames,
    });
  }
  if (now - lastRadarTime > 66) { draw(); lastRadarTime = now; }
  requestAnimationFrame(renderLoop);
}
requestAnimationFrame(renderLoop);

function keyDirection() {
  let x = dragDirection.x, y = dragDirection.y;
  if (keys.has("KeyA") || keys.has("ArrowLeft")) x -= 1;
  if (keys.has("KeyD") || keys.has("ArrowRight")) x += 1;
  if (keys.has("KeyW") || keys.has("ArrowUp")) y -= 1;
  if (keys.has("KeyS") || keys.has("ArrowDown")) y += 1;
  const mag = Math.hypot(x,y);
  return mag > 0 ? {x:x/mag, y:y/mag} : {x:0,y:0};
}

function predictedCellFree(x,y) {
  const grid = state?.map?.grid;
  if (!grid?.length) return true;
  const cx = Math.round(x);
  const cy = Math.round(y);
  return (
    cy >= 0 && cy < grid.length &&
    cx >= 0 && cx < grid[cy].length &&
    !grid[cy][cx] &&
    !(state.map.doors || []).some(door => door.locked && door.x === cx && door.y === cy)
  );
}

function updatePredictedPlayer(dt) {
  if (!state?.player || !predictedPlayer || state.status !== "RUNNING") return;
  if (state.takedown || state.player.prone || state.player.rising || state.player.diving || divePulse || state.player.fear) return;


  const dir = keyDirection();
  if (!dir.x && !dir.y) return;

  const gp = state.gameplay || {};
  let speed = gp.walk_speed || 1.5;
  if (crouchToggled && state.player.can_crouch !== false) {
    speed = gp.crouch_speed || speed * 0.55;
  } else if ((keys.has("ShiftLeft") || keys.has("ShiftRight")) && !state.player.exhausted && state.player.can_sprint !== false && !state.player.injured) {
    speed = gp.sprint_speed || speed * 1.6;
    if (state.player.boost_remaining > 0) speed *= 1.65;
  }
  speed *= state.player.speed_multiplier ?? 1;
  speed *= state.player.weapon_speed_multiplier ?? 1;
  speed *= state.player.backpack_speed_multiplier ?? 1;
  if (state.player.flashed) speed *= .28;

  const dx = dir.x * speed * dt;
  const dy = dir.y * speed * dt;

  const nx = predictedPlayer.x + dx;
  const ny = predictedPlayer.y + dy;

  if (predictedCellFree(nx, ny)) {
    predictedPlayer.x = nx;
    predictedPlayer.y = ny;
    return;
  }

  if (predictedCellFree(predictedPlayer.x + dx, predictedPlayer.y)) {
    predictedPlayer.x += dx;
  }
  if (predictedCellFree(predictedPlayer.x, predictedPlayer.y + dy)) {
    predictedPlayer.y += dy;
  }
}

function alertPresentation(current) {
  if (current.gameplay?.spawn_protected || (current.gameplay?.spawn_grace || 0) > 0) return {title: "NORMALE", detail: "Punto d'infiltrazione sicuro", alarm: false};
  const security = current.security || {};
  if (security.countdown) {
    const phase = security.countdown.phase;
    if (phase === "CONTACT") return {title:"ALLERTA", detail:"Intruso segnalato · Perdi la visuale nemica per accelerare il conto alla rovescia", alarm:true};
    if (phase === "SEARCH") return {title:"EVASIONE", detail:"Ricerca sull’ultimo punto noto", alarm:false};
    if (phase === "SUSPICIOUS") return {title:"CAUTELA", detail:"Sorveglianza rafforzata", alarm:false};
    return {title:"NORMALE",detail:"Nessun contatto confermato",alarm:false};
  }
  const dialogue = Object.values(security.dialogue || {}).filter(Boolean).slice(-2).join(" · ");
  if (current.mode === "swat" && !security.alert) return {title: "ALLERTA", detail: "Intervento della SWAT", alarm: true};
  if (security.source === "GUNFIRE" && ["CONTACT", "SEARCH"].includes(security.alert)) return {title:"ALLERTA", detail:"Colpi di arma da fuoco - Allerta generale", alarm:true};
  if (security.swat_phase === "LOCAL SUPPORT" && security.alert === "CONTACT") return {title:"ALLERTA",detail:dialogue || "Contatto · Supporto delle guardie",alarm:true};
  if (security.swat_phase === "SPOTTED") return {title: "SPOTTED", detail: "Intruso avvistato", alarm: true};
  if (security.swat_phase === "RADIO") return {title: "SPOTTED", detail: "Chiamata alla radio in corso", alarm: true};
  if (security.swat_phase === "DISPATCH" || security.alert === "CONTACT") return {title: "ALLERTA", detail: "Intervento della SWAT", alarm: true};
  if (security.alert === "SEARCH" && security.contact) return {title: "EVASIONE", detail: "Investigazione sull'ultimo punto noto", alarm: false};
  if (security.alert && security.alert !== "CALM") {
    const detail = security.swat_phase === "RETURNING" ? "Guardia alta · SWAT in ritirata" : "Guardia alta · Sorveglianza rafforzata";
    return {title: "CAUTELA", detail: dialogue ? `${detail} · ${dialogue}` : detail, alarm: false};
  }
  return {title: "NORMALE", detail: dialogue || "Nessun contatto confermato", alarm: false};
}

function refreshUI() {
  if (!state) return;
  const alert = alertPresentation(state);
  updateRadarStatus(performance.now());
  if (ui.detail.textContent !== alert.detail) ui.detail.textContent = alert.detail;
  document.body.dataset.alarm = String(alert.alarm);
  const radio = document.getElementById("radioMessage");
  if (radio) {
    const messages = state.security?.radio_log || Object.entries(state.security?.dialogue || {}).filter(([,text]) => text).map(([sender,text])=>({sender,text}));
    const entries = messages.slice(-3);
    const markup = entries.map(entry => {
      const time = entry.time == null ? "" : `${String(Math.floor(entry.time/60)).padStart(2,"0")}:${String(Math.floor(entry.time%60)).padStart(2,"0")} `;
      const text = `${time}${entry.sender} · ${entry.text}`.replace(/[&<>"']/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
      return `<p class="radio-entry">${text}</p>`;
    }).join("");
    if (radio.dataset.logKey !== markup) { radio.innerHTML = markup; radio.dataset.logKey = markup; }
    radio.hidden = !entries.length;
  }
  document.getElementById("levelLabel").textContent = `SETTORE ${String(state.level || 1).padStart(2, "0")} / 03`;
  const learningStatus = document.getElementById("learningStatus");
  if (learningStatus) {
    const lesson = state.previous_learning;
    learningStatus.hidden = !lesson;
    learningStatus.textContent = lesson ? `IA · ${lesson.status === "TRAINED" ? "PERCORSO APPRESO" : lesson.status === "FAILED" ? "APPRENDIMENTO NON RIUSCITO" : "ANALISI DEL PERCORSO"} · SETTORE ${String(lesson.level || 1).padStart(2,"0")}` : "";
  }
  const briefing=state.mission;
  const missionTitle=document.getElementById("missionTitle"), missionObjective=document.getElementById("missionObjective"), requirements=document.getElementById("missionRequirements");
  if (missionTitle) missionTitle.textContent=briefing ? `MISSIONE ${briefing.level} / 3 · ${briefing.title}` : "MISSIONE / INFILTRAZIONE";
  if (missionObjective) missionObjective.textContent=briefing?.objective || "Raggiungi l’esfiltrazione";
  if (requirements) requirements.textContent=(briefing?.requirements || []).map(r=>`${r.ready ? "✓" : "○"} ${r.name}`).join(" · ");
  const intelTotal = state.map?.intel?.length || 0;
  const collected = state.player?.stolen_count || 0;
  document.getElementById("intelLabel").textContent = state.mode === "guard" ? `OBIETTIVI ${String(collected).padStart(2,"0")} / ${String(intelTotal).padStart(2,"0")}` : "OBIETTIVO / ESTRAZIONE";
  document.getElementById("intelProgress").style.width = `${intelTotal ? collected / intelTotal * 100 : 0}%`;
  const gear = state.player || {};
  refreshBackpack(gear);
  refreshPlayerHUD(gear);
  const weapon = gear.weapon ? (itemNames[gear.weapon] || gear.weapon).toUpperCase() : "DISARMATO";
  document.getElementById("equipmentLabel").textContent = `${weapon}${gear.weapon ? ` ${gear.ammo || 0}` : ""}${gear.knife ? " · COLTELLO" : ""} · B ${gear.flash || 0} · N ${gear.smoke || 0} · BENDE ${gear.bandages || 0} · KIT ${gear.medkits || 0}`;
  document.getElementById("radarCoords").textContent = radarDisturbed(state) ? "SEGNALE ASSENTE" : `X ${(state.player?.x || 0).toFixed(1).padStart(4,"0")} / Y ${(state.player?.y || 0).toFixed(1).padStart(4,"0")}`;
  let outcome = "";
  if (state.status === "ESCAPED") outcome = state.level < 3 ? "Livello completato - INVIO: prossimo livello" : "Missione completata - INVIO: riprova";
  else if (state.status === "CAPTURED") outcome = "Catturato - INVIO: riprova";
  else if (state.status === "NEUTRALIZED") outcome = "Neutralizzato - INVIO: riprova";
  const learning = state.learning || {};
  if (outcome && learning.status === "TRAINING") outcome += " · IA in addestramento";
  if (outcome && learning.status === "TRAINED") outcome += " · IA aggiornata";
  if (outcome && learning.status === "FAILED") outcome += " · Addestramento non riuscito";
  const message = outcome || window.shadowSceneError || "";
  ui.banner.textContent = message;
  ui.banner.classList.toggle("hidden", !message);
}

function updateRadarStatus(now) {
  if (!state) return;
  const session = `${state.revision}:${state.mode}:${state.level}:${state.scenario}`;
  if (session !== radarClockSession) {
    radarClockSession = session; radarPhase = ""; radarJammedSince = null; radarFrozenAt = null;
  }
  if (state.status === "RUNNING") radarFrozenAt = null;
  else if (radarFrozenAt === null) radarFrozenAt = now;
  now = radarFrozenAt ?? now;
  const title = alertPresentation(state).title;
  const phase = title === "SPOTTED" ? "ALLERTA" : title;
  const jammed = phase === "ALLERTA" || phase === "EVASIONE";
  if (phase !== radarPhase) { radarPhase = phase; radarPhaseSince = now; }
  if (jammed && radarJammedSince === null) radarJammedSince = now;
  if (!jammed) radarJammedSince = null;
  const interrupted = jammed && now-radarJammedSince < 1200;
  const label = interrupted ? "DISTURBATO" : phase;
  ui.hud.hidden = phase === "NORMALE";
  ui.hud.dataset.phase = phase;
  if (ui.title.textContent !== label) ui.title.textContent = label;
  const timer = document.getElementById("alertDuration");
  if (timer) {
    const countdown = state.security?.countdown;
    timer.hidden = (countdown ? phase === "NORMALE" : !jammed) || interrupted;
    const seconds = Math.max(0, Math.floor((now-radarPhaseSince)/1000));
    const remaining = Math.max(0,countdown?.remaining || 0);
    const duration = countdown ? phase === "CAUTELA" ? `${String(Math.floor(Math.ceil(remaining)/60)).padStart(2,"0")}:${String(Math.ceil(remaining)%60).padStart(2,"0")}` : remaining.toFixed(2).padStart(5,"0") : `${String(Math.floor(seconds/60)).padStart(2,"0")}:${String(seconds%60).padStart(2,"0")}`;
    if (timer.textContent !== duration) timer.textContent = duration;
  }
  const panel = document.getElementById("radarPanel");
  if (panel) panel.dataset.phase = phase;
}

function transform() {
  const rect = radarCanvas.getBoundingClientRect();
  const w = state?.map?.width || 32;
  const h = state?.map?.height || 20;
  const pad = radarCanvas === canvas ? 32 : 10;
  const s = Math.min((rect.width-pad*2)/w, (rect.height-pad*2)/h);
  const ox = (rect.width - w*s)/2;
  const oy = (rect.height - h*s)/2;
  return {s, ox, oy, w, h, rect};
}
function X(x,t) { return t.ox + x*t.s; }
function Y(y,t) { return t.oy + y*t.s; }

function draw() {
  const rect = radarCanvas.getBoundingClientRect();
  ctx.clearRect(0,0,rect.width,rect.height);
  if (!state?.map) return;
  const t = transform();

  const alarm = alertPresentation(state).alarm;
  ctx.fillStyle = alarm ? "#230b13" : "#12251a";
  ctx.fillRect(t.ox, t.oy, t.w*t.s, t.h*t.s);

  ctx.save();
  ctx.beginPath(); ctx.rect(t.ox,t.oy,t.w*t.s,t.h*t.s); ctx.clip();
  const disturbed = radarDisturbed(state);
  if (disturbed) drawRadarJamming(t);
  else drawSimpleRadar(t);
  ctx.restore();
  const panel = document.getElementById("radarPanel");
  if (panel) panel.dataset.signal = disturbed ? "jammed" : "clear";
  if (movementPointer && radarCanvas === canvas) {
    const x = movementPointer.x - rect.left, y = movementPointer.y - rect.top;
    ctx.strokeStyle = "rgba(98,215,255,.35)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(x,y,30,0,Math.PI*2); ctx.stroke();
    ctx.fillStyle = "rgba(98,215,255,.65)";
    ctx.beginPath(); ctx.arc(x+dragDirection.x*20,y+dragDirection.y*20,6,0,Math.PI*2); ctx.fill();
  }
}

function radarDisturbed(current) {
  return ["ALLERTA", "SPOTTED", "EVASIONE"].includes(alertPresentation(current).title);
}

function drawSimpleRadar(t) {
  ctx.fillStyle = "#0c1e26";
  ctx.fillRect(t.ox, t.oy, t.w*t.s, t.h*t.s);
  ctx.globalAlpha = .85;
  ctx.fillStyle = "#647f85";
  for (let y=0; y<(state.map.grid?.length || 0); y++) {
    for (let x=0; x<state.map.grid[y].length; x++) if (state.map.grid[y][x]) {
      ctx.fillRect(X(x-.5,t),Y(y-.5,t),t.s,t.s);
    }
  }
  for (const door of state.map.doors || []) if (door.locked) ctx.fillRect(X(door.x-.5,t),Y(door.y-.5,t),t.s,t.s);
  ctx.globalAlpha = 1;
  drawRadarContacts(t);
  const player = predictedPlayer || state.player;
  if (player) drawRadarMarker(player, t, "#d5f4dd", facingAngle, 3.5);
}

function drawRadarMarker(point, t, color, angle=0, size=2.5) {
  ctx.save(); ctx.translate(X(point.x,t),Y(point.y,t)); ctx.rotate(angle);
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.moveTo(size*1.5,0); ctx.lineTo(-size,-size); ctx.lineTo(-size,size); ctx.closePath(); ctx.fill();
  ctx.restore();
}

function drawRadarContacts(t) {
  const contacts = [...(state.guards || (state.guard ? [state.guard] : [])), ...(state.swat || [])];
  for (const source of contacts) {
    if (!source.alive || source.unconscious) continue;
    const agent = displayedAgent(source), palette = agentPalette(agent);
    if (ui.vision.checked) drawVision(agent,t,palette);
    drawRadarMarker(agent,t,palette.color,agent.angle || 0);
  }
  for (const source of state.cameras || []) if (source.online) {
    const camera = displayedAgent(source);
    if (ui.vision.checked) drawVision(camera,t,agentPalette(camera));
    ctx.fillStyle = "#87b3c0"; ctx.fillRect(X(camera.x,t)-1.5,Y(camera.y,t)-1.5,3,3);
  }
}

function drawRadarJamming(t) {
  const time = performance.now() / 1000;
  ctx.save(); ctx.beginPath(); ctx.rect(t.ox,t.oy,t.w*t.s,t.h*t.s); ctx.clip();
  const width=t.w*t.s, height=t.h*t.s;
  ctx.fillStyle="#071119"; ctx.fillRect(t.ox,t.oy,width,height);
  for (let i=0;i<32;i++) {
    const phase=Math.sin(i*127.1+Math.floor(time*14)*17.7)*43758.5453;
    const jitter=phase-Math.floor(phase);
    const y=t.oy+((i/32+time*.06)%1)*height;
    ctx.fillStyle=`rgba(153,187,194,${.04+jitter*.16})`;
    ctx.fillRect(t.ox+jitter*width*.18,y,width*(.5+jitter*.5),1+jitter*2);
  }
  const sweep=(time*37)%Math.max(1,height);
  ctx.fillStyle="rgba(177,203,205,.2)"; ctx.fillRect(t.ox,t.oy+sweep,width,4);
  ctx.fillStyle="rgba(4,13,19,.92)"; ctx.fillRect(t.ox,t.oy+height*.43,width,height*.2);
  ctx.restore();
}

function drawGrid(t) {
  const grid = state.map.grid;
  ctx.fillStyle = "#12272e";
  ctx.fillRect(X(-.5,t),Y(-.5,t),t.w*t.s,t.h*t.s);
  for (let y=0; y<grid.length; y++) {
    for (let x=0; x<grid[y].length; x++) {
      if (grid[y][x]) {
        ctx.fillStyle = alertPresentation(state).alarm ? "#704452" : "#b3c9cb";
        ctx.fillRect(X(x-.5,t), Y(y-.5,t), t.s, t.s);
        ctx.strokeStyle = alertPresentation(state).alarm ? "#bc7788" : "#d6e9e9";
        ctx.lineWidth = .65;
        ctx.strokeRect(X(x-.5,t)+.5, Y(y-.5,t)+.5, t.s-1, t.s-1);
      }
    }
  }

  ctx.strokeStyle = "rgba(155,210,225,.09)";
  ctx.lineWidth = 1;
  for (let x=0; x<=t.w; x++) {
    ctx.beginPath(); ctx.moveTo(X(x,t),Y(0,t)); ctx.lineTo(X(x,t),Y(t.h,t)); ctx.stroke();
  }
  for (let y=0; y<=t.h; y++) {
    ctx.beginPath(); ctx.moveTo(X(0,t),Y(y,t)); ctx.lineTo(X(t.w,t),Y(y,t)); ctx.stroke();
  }
}

function drawZones(t) {
  for (const hazard of state.map.hazards || []) {
    ctx.fillStyle="rgba(246,108,75,.25)";
    ctx.fillRect(X(hazard.x,t),Y(hazard.y,t),hazard.width*t.s,hazard.height*t.s);
    ctx.strokeStyle="#ff8868"; ctx.lineWidth=1;
    ctx.strokeRect(X(hazard.x,t),Y(hazard.y,t),hazard.width*t.s,hazard.height*t.s);
  }
  for (const zone of state.map.zones || []) {
    ctx.font = `700 ${Math.max(4,Math.min(8,t.s*1.05))}px ui-monospace, monospace`;
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(175,219,229,.75)";
    ctx.fillText(zone.name, X(zone.x,t), Y(zone.y,t));
  }
}

function drawSemantic(t) {
  if (!ui.semantic.checked) return;
  for (const p of state.map.semantic || []) {
    ctx.globalAlpha = p.cleared ? .18 : .55;
    ctx.strokeStyle = semanticColor(p.type);
    ctx.lineWidth = 1;
    const x=X(p.x,t), y=Y(p.y,t);
    ctx.strokeRect(x-3,y-3,6,6);
  }
  ctx.globalAlpha = 1;
}
function semanticColor(type) {
  if (type === "CORNER") return "#f9c66d";
  if (type === "DOOR" || type === "ROOM_ENTRY") return "#64d9ff";
  if (type === "COVER_POINT") return "#c891ff";
  return "#79a99a";
}

function drawIntel(t) {
  for (const p of state.map.intel || []) {
    if (p.stolen) continue;
    const x=X(p.x,t), y=Y(p.y,t);
    ctx.save(); ctx.translate(x,y); ctx.rotate(Math.PI/4);
    ctx.fillStyle="#ddd18a";
    ctx.fillRect(-4,-4,8,8);
    ctx.strokeStyle="rgba(235,220,152,.35)";
    ctx.strokeRect(-7,-7,14,14);
    ctx.restore();
  }
}

function drawSmoke(t) {
  for (const c of state.smoke || []) {
    ctx.beginPath();
    ctx.arc(X(c.x,t),Y(c.y,t),c.radius*t.s,0,Math.PI*2);
    ctx.fillStyle="rgba(190,204,213,.58)";
    ctx.fill();
    ctx.strokeStyle="rgba(190,220,210,.28)";
    ctx.stroke();
  }
  for (const flash of state.flashes || []) {
    ctx.fillStyle = `rgba(255,250,215,${Math.min(1, flash.ttl / .45) * .8})`;
    ctx.beginPath();
    ctx.arc(X(flash.x,t),Y(flash.y,t),3*t.s,0,Math.PI*2);
    ctx.fill();
  }
  if (state.player?.flashed) {
    ctx.fillStyle = "rgba(255,250,225,.45)";
    ctx.fillRect(t.ox,t.oy,t.w*t.s,t.h*t.s);
  }
}

function drawPickups(t) {
  const names = {KNIFE:"COLTELLO",PISTOL:"PISTOLA",RIFLE:"FUCILE",SHOTGUN:"FUCILE A CANNA LISCIA",SMG:"MITRA",ONE_HAND_SMG:"MITRA A UNA MANO",GRENADE:"GRANATA",SNACK:"SNACK",AMMO:"MUNIZIONI",MEDKIT:"KIT MEDICO",BANDAGE:"BENDA",CASING:"BOSSOLO",MAGAZINE:"CARICATORE",FLASH:"FLASH",SMOKE:"FUMOGENO",HAZMAT:"PROTEZIONE BIOLOGICA",SAMPLE_VIAL:"PROVETTA STERILE",SECURITY_CARD:"SCHEDA DI SICUREZZA"};
  const groups = new Map();
  for (const item of state.pickups || []) {
    const key = ["MAP", "PLAYER"].includes(item.source) ? item.id : item.source;
    if (!groups.has(key)) groups.set(key,[]);
    groups.get(key).push(item);
  }
  for (const items of groups.values()) {
    const item = items[0];
    const x=X(item.x,t), y=Y(item.y,t);
    ctx.fillStyle = (item.kind === "MEDKIT" || item.kind === "BANDAGE") ? "#62f2ad" : item.source === "MAP" ? "#62d7ff" : "#ffcf67";
    ctx.save();
    const color=ctx.fillStyle;
    ctx.fillStyle=item.fixed ? "#12382d" : "#0e222a";
    ctx.fillRect(x-8,y-8,16,16);
    ctx.strokeStyle=color; ctx.lineWidth=1; ctx.strokeRect(x-8,y-8,16,16);
    if (typeof Path2D !== "undefined") {
      ctx.translate(x-6,y-6); ctx.scale(.5,.5);
      ctx.lineWidth=2; ctx.stroke(new Path2D(itemPaths[item.kind] || itemPaths.CASING));
    } else { ctx.fillStyle=color; ctx.fillRect(x-3,y-3,6,6); }
    ctx.restore();
    ctx.fillStyle=color;
    const near = state.player && Math.hypot(state.player.x-item.x,state.player.y-item.y) <= 1.2;
    const name = item.fixed && item.kind === "MEDKIT" ? "POSTAZIONE MEDICA" : names[item.kind] || item.kind;
    label(x+6,y-5,`${near ? (item.fixed ? "TIENI E · " : "E · ") : ""}${name}${items.length > 1 ? ` +${items.length-1}` : ""}`,ctx.fillStyle);
  }
}

function drawExplosives(t) {
  for (const g of state.grenades || []) {
    ctx.fillStyle = "#ff735e";
    ctx.beginPath(); ctx.arc(X(g.x,t),Y(g.y,t),4,0,Math.PI*2); ctx.fill();
    ctx.strokeStyle = "rgba(255,115,94,.5)";
    ctx.beginPath(); ctx.arc(X(g.target_x,t),Y(g.target_y,t),g.radius*t.s,0,Math.PI*2); ctx.stroke();
    label(X(g.x,t)+7,Y(g.y,t)-8,`GRANATA ${g.fuse.toFixed(1)}s`,"#ff735e");
  }
  for (const b of state.blasts || []) {
    ctx.fillStyle = `rgba(255,150,55,${Math.min(.7,b.ttl)})`;
    ctx.beginPath(); ctx.arc(X(b.x,t),Y(b.y,t),b.radius*t.s*(1-b.ttl),0,Math.PI*2); ctx.fill();
  }
}

function drawBlood(t) {
  for (const mark of state.blood || []) {
    const x=X(mark.x,t), y=Y(mark.y,t);
    const ageFade = Math.max(.22, 1 - (mark.age || 0) / 100);
    ctx.fillStyle=`rgba(151,16,35,${.72*ageFade})`;
    ctx.beginPath();
    ctx.ellipse(x,y,Math.max(2,t.s*.14),Math.max(1.2,t.s*.08),mark.id*.71,0,Math.PI*2);
    ctx.fill();
  }
}

function drawAccessDoors(t) {
  for (const door of state.map.doors || []) {
    ctx.fillStyle = door.locked ? "#d69a4e" : "#459d78";
    ctx.fillRect(X(door.x-.42,t), Y(door.y-.42,t), t.s*.84, t.s*.84);
    if (door.locked && Math.hypot(state.player.x-door.x,state.player.y-door.y)<3) {
      label(X(door.x,t)+8,Y(door.y,t)-10,"E · TESSERA ARCHIVIO","#f2c578");
    }
  }
}

function drawNoise(t) {
  for (const event of state.noise || []) {
    const x=X(event.x,t), y=Y(event.y,t);
    const life=Math.max(0,Math.min(1,(event.ttl||0)/.72));
    const radius=(event.radius||4)*t.s*(1-life*.55);
    ctx.strokeStyle=`rgba(255,207,103,${life*.52})`;
    ctx.lineWidth=1.2;
    ctx.beginPath(); ctx.arc(x,y,radius,0,Math.PI*2); ctx.stroke();
  }
}

function drawDiversions(t) {
  for (const device of state.diversions || []) {
    const x=X(device.x,t), y=Y(device.y,t);
    ctx.save(); ctx.translate(x,y); ctx.rotate(Math.PI/4);
    ctx.fillStyle=device.landed ? "#ffcf67" : "#f7f2c5";
    ctx.fillRect(-3,-3,6,6);
    ctx.restore();
    if (!device.landed) label(x+6,y-7,"THROWN","#ffcf67");
  }
}

function drawCameras(t) {
  if (state.mode !== "guard") return;
  for (const source of state.cameras || []) {
    const camera = displayedAgent(source);
    const palette = agentPalette(camera);
    if (ui.vision.checked && camera.online) {
      drawVision(camera,t,palette);
    }
    const x=X(camera.x,t), y=Y(camera.y,t);
    ctx.save(); ctx.translate(x,y); ctx.rotate(camera.angle||0);
    ctx.fillStyle=palette.color;
    ctx.fillRect(-4,-3,7,6);
    ctx.beginPath(); ctx.moveTo(3,-4); ctx.lineTo(8,0); ctx.lineTo(3,4); ctx.closePath(); ctx.fill();
    ctx.restore();
    if (camera.tracking) label(x+7,y-8,`${camera.id} TRACK`,"#ff4a5b");
  }
}

function drawBodies(t) {
  for (const body of state.bodies || []) {
    const x=X(body.x,t), y=Y(body.y,t);
    const fall=(body.fall||0);
    ctx.save();
    ctx.translate(x + Math.sin(fall*19)*fall*2.2, y);
    ctx.rotate((body.angle||0)+Math.PI/2*(1-fall));
    ctx.strokeStyle=body.lethal ? "#6d2733" : "#71817c";
    ctx.lineWidth=Math.max(2,t.s*.16);
    ctx.lineCap="round";
    ctx.beginPath(); ctx.moveTo(-t.s*.22,0); ctx.lineTo(t.s*.24,0); ctx.stroke();
    ctx.beginPath(); ctx.arc(-t.s*.25,0,Math.max(2,t.s*.09),0,Math.PI*2); ctx.stroke();
    ctx.restore();
    if (body.discovered) label(x+7,y+8,"FOUND","#ff6474");
  }
}

function drawGuardMode(t) {
  drawSwatMode(t);
  const guards=(state.guards || (state.guard ? [state.guard] : [])).map(displayedAgent);
  for (const g of guards) {
    if (ui.vision.checked && g.alive && !g.unconscious) {
      drawVision(g,t,agentPalette(g));
    }
    if (ui.memory.checked && g.last_seen) drawMemory(g.last_seen,t,`${g.id||"G"} MEM`);
  }

  if (ui.memory.checked && state.security?.contact) {
    drawMemory(state.security.contact,t,"RADIO CONTACT");
  }

  for (const g of guards) {
    if (!g.alive || g.unconscious) continue;
    drawAgent(g,t,g.id || "G",agentPalette(g).color,g.action);
    drawVitals(g,t);
  }
}

function drawSwatMode(t) {
  const units = (state.swat || []).map(displayedAgent);
  for (const u of units) {
    if (ui.vision.checked && u.alive) {
      drawVision(u,t,agentPalette(u));
    }
    if (ui.memory.checked && u.last_seen) drawMemory(u.last_seen,t,`${u.role} MEM`);
  }

  if (ui.memory.checked && state.blackboard?.contact) {
    const p=state.blackboard.contact;
    const x=X(p.x,t), y=Y(p.y,t);
    ctx.save();
    ctx.translate(x,y); ctx.rotate(Math.PI/4);
    ctx.strokeStyle="#ffcf67"; ctx.lineWidth=2;
    ctx.strokeRect(-6,-6,12,12);
    ctx.restore();
    label(x+9,y-8,"TEAM CONTACT","#ffcf67");
  }

  for (const u of units) {
    drawAgent(u,t,u.role[0],agentPalette(u).color,u.action, !u.alive);
    if (u.alive) drawVitals(u,t);
  }
}
function agentPalette(agent) {
  if (agent.combat_alert) return {rgb: "255,89,103", color: "rgb(255,89,103)"};
  const confidence = Math.max(0, Math.min(1, agent.confidence ?? agent.visual_confidence ??
    (agent.visual || agent.local_visual ? 1 : 0)));
  if (confidence < .5 && (agent.vision_status === "SUSPICIOUS" || agent.sound || agent.local_sound ||
      (agent.suspicion ?? 0) >= .25 || agent.action === "INVESTIGATE")) {
    return {rgb: "255,212,106", color: "rgb(255,212,106)"};
  }
  const low = confidence <= .5 ? [146,216,248] : [255,212,106];
  const high = confidence <= .5 ? [255,212,106] : [255,89,103];
  const mix = confidence <= .5 ? confidence * 2 : (confidence - .5) * 2;
  const rgb = low.map((value, i) => Math.round(value + (high[i] - value) * mix));
  return {rgb: rgb.join(","), color: `rgb(${rgb.join(",")})`};
}

function updateRenderedAgents(dt) {
  const agents = [...(state?.guards || []), ...(state?.swat || []), ...(state?.cameras || [])];
  const active = new Set();
  const gain = 1 - Math.exp(-20 * dt);
  for (const agent of agents) {
    active.add(agent.id);
    const old = renderedAgents.get(agent.id);
    if (!old || Math.hypot(old.x-agent.x, old.y-agent.y) > 2) {
      renderedAgents.set(agent.id, {...agent});
      continue;
    }
    const angleDelta = Math.atan2(Math.sin(agent.angle-old.angle), Math.cos(agent.angle-old.angle));
    renderedAgents.set(agent.id, {...agent, x: old.x+(agent.x-old.x)*gain, y: old.y+(agent.y-old.y)*gain, angle: old.angle+angleDelta*gain});
  }
  for (const id of renderedAgents.keys()) if (!active.has(id)) renderedAgents.delete(id);
}
function displayedAgent(agent) { return renderedAgents.get(agent.id) || agent; }

function visionOutline(agent, grid) {
  const range = agent.vision_range || 8;
  const half = ((agent.fov || 104) * Math.PI / 180) / 2;
  const rays = Math.ceil(half * 2 / .025);
  const points = [];
  for (let i=0; i<=rays; i++) {
    const angle = (agent.angle || 0)-half+half*2*i/rays;
    const dx = Math.cos(angle), dy = Math.sin(angle);
    let distance = range;
    for (let step=.12; step<=range; step+=.12) {
      const cx = Math.round(agent.x+dx*step), cy = Math.round(agent.y+dy*step);
      if (cy<0 || cy>=grid.length || cx<0 || cx>=grid[cy].length || grid[cy][cx]) {
        distance = Math.max(0, step-.12);
        break;
      }
    }
    points.push({x:agent.x+dx*distance, y:agent.y+dy*distance});
  }
  return points;
}

function drawVision(a,t,palette) {
  const x=X(a.x,t), y=Y(a.y,t);
  const range=(a.vision_range || 8)*t.s;
  ctx.beginPath();
  const outline = visionOutline(a, state.map.grid);
  for (const [i,point] of outline.entries()) {
    if (!i) ctx.moveTo(X(point.x,t),Y(point.y,t));
    else ctx.lineTo(X(point.x,t),Y(point.y,t));
  }
  for (const point of [...outline].reverse()) {
    const dx=point.x-a.x,dy=point.y-a.y,length=Math.hypot(dx,dy);
    const ratio=length>0 ? Math.min(1,(a.vision_inner_range || 0)/length) : 0;
    ctx.lineTo(X(a.x+dx*ratio,t),Y(a.y+dy*ratio,t));
  }
  ctx.closePath();
  const gradient = ctx.createRadialGradient(x,y,0,x,y,range);
  gradient.addColorStop(0,`rgba(${palette.rgb},.28)`);
  gradient.addColorStop(1,`rgba(${palette.rgb},.10)`);
  ctx.fillStyle=gradient;
  ctx.fill();
  ctx.strokeStyle=`rgba(${palette.rgb},.28)`; ctx.lineWidth=.7; ctx.stroke();
}

function drawMemory(p,t,text) {
  const x=X(p.x,t), y=Y(p.y,t);
  ctx.strokeStyle="#ffcf67"; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.arc(x,y,7,0,Math.PI*2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x-9,y); ctx.lineTo(x+9,y); ctx.moveTo(x,y-9); ctx.lineTo(x,y+9); ctx.stroke();
  label(x+10,y+12,text,"#ffcf67");
}

function enemyActionLabel(agent) {
  if (!agent.observed || !agent.alive || agent.unconscious) return "";
  if ((agent.shield || agent.role === "SHIELD") && (agent.suppression_timer || 0) > 0) return "SCUDO SOTTO FUOCO";
  if (agent.draw_remaining > 0) return "CAMBIO ARMA";
  if (agent.firearm?.reload_mode) return agent.firearm.reload_mode === "FAST" ? "RICARICA RAPIDA" : "RICARICA";
  if (agent.secondary_weapon && agent.weapon === agent.secondary_weapon) return "PISTOLA SECONDARIA";
  return "";
}

function drawAgent(a,t,char,color,action,down=false) {
  const x=X(a.x,t), y=Y(a.y,t);
  const tip = Math.max(5, Math.min(7, t.s * 0.24));
  const tail = tip * 0.66;
  ctx.save();
  ctx.translate(x,y);
  ctx.rotate(a.angle || 0);
  ctx.globalAlpha = down ? .3 : 1;
  ctx.fillStyle=color;
  ctx.shadowColor=color; ctx.shadowBlur=5;
  ctx.beginPath();
  ctx.moveTo(tip,0);
  ctx.lineTo(-tail,-tail);
  ctx.lineTo(-tail,tail);
  ctx.closePath();
  ctx.fill();
  if (a.observed && !down && Math.abs(a.lean_amount || 0) > .1) {
    const lean = Math.max(-1,Math.min(1,a.lean_amount));
    ctx.shadowBlur=0; ctx.strokeStyle="#dce8ef"; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(tip,lean*tail*1.3); ctx.stroke();
  }
  if (a.shield && !down) {
    ctx.shadowBlur=0; ctx.fillStyle="#263746"; ctx.strokeStyle=(a.suppression_timer || 0)>0 ? "#ffcf67" : "#b4c6df"; ctx.lineWidth=1.5;
    ctx.fillRect(11,-11,5,22); ctx.strokeRect(11,-11,5,22);
    ctx.fillStyle="#8ed7e2"; ctx.fillRect(12,-4,3,8);
  }
  ctx.restore();
  if (a.is_medic && a.observed && !down) label(x+9,y-9,"+ MEDICO","#62f2ad");
  const equipmentAction = down ? "" : enemyActionLabel(a);
  if (equipmentAction) label(x+9,y-21,equipmentAction,"#dce8ef");
}

function playerColor(current) {
  if (current.player?.compliant) return "#f4f4f4";
  if (current.takedown?.kind === "CHOKE") return "#62f2ad";
  if (current.takedown_candidate) return current.takedown_candidate.can_lethal === false ? "#62f2ad" : "#ffcf67";
  return "#f0f4cc";
}

function drawVitals(agent,t,isPlayer=false) {
  if (!isPlayer && agent.observed !== true) return;
  const x = X(agent.x,t), y = Y(agent.y,t) + 13;
  if (agent.health !== undefined) {
    ctx.fillStyle = "rgba(0,0,0,.7)"; ctx.fillRect(x-10,y,20,2);
    ctx.fillStyle = agent.health < .55 ? "#ff5967" : "#62f2ad";
    ctx.fillRect(x-10,y,20*Math.max(0,Math.min(1,agent.health)),2);
  }
  if (agent.stamina !== undefined) {
    ctx.fillStyle = "rgba(0,0,0,.7)"; ctx.fillRect(x-10,y+4,20,2);
    ctx.fillStyle = agent.exhausted ? "#ffcf67" : "#62d7ff";
    ctx.fillRect(x-10,y+4,20*Math.max(0,Math.min(1,agent.stamina)),2);
  }
  if (agent.armor > 0) {
    ctx.fillStyle = "#394652"; ctx.fillRect(x-10,y+8,20,2);
    ctx.fillStyle = "#b4c6df"; ctx.fillRect(x-10,y+8,20*agent.armor,2);
  }

  if (agent.aiming) {
    const angle = agent.angle || 0;
    const length = 8 + 12 * (agent.aim_progress || 0);
    ctx.strokeStyle = "#ff5967"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x,y-13);
    ctx.lineTo(x+Math.cos(angle)*length,y-13+Math.sin(angle)*length); ctx.stroke();
    label(x+10,y-20,"AIM","#ff5967");
  }
  if (agent.melee_windup > 0) label(x+10,y-20,agent.shield ? "COLPO DI SCUDO · SCHIVA!" : "CORPO A CORPO · F PER DISARMARE","#ff735e");
  else if (agent.disarmed && agent.observed) label(x+10,y-20,"DISARMATO","#ffcf67");
}

function drawPlayer(t) {
  const p=state.player;
  if (!p) return;
  const rp = predictedPlayer || p;
  const x=X(rp.x,t),y=Y(rp.y,t);
  const speed=Math.hypot(p.vx||0,p.vy||0);
  let angle=0;
  angle=p.angle ?? (speed>.01 ? Math.atan2(p.vy,p.vx) : 0);

  ctx.save();
  ctx.translate(x,y); ctx.rotate(angle);
  ctx.fillStyle=playerColor(state);
  ctx.shadowColor=playerColor(state); ctx.shadowBlur=9;
  const flat = p.diving || (p.prone && !p.rising);
  const radius = p.crouched ? 5 : 7;
  ctx.strokeStyle = playerColor(state);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.ellipse(0, 0, flat ? 12 : radius, flat ? 4 : radius, 0, 0, Math.PI*2);
  ctx.fillStyle = "#172c33";
  ctx.fill(); ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.fillStyle = playerColor(state);
  ctx.beginPath(); ctx.arc(0, 0, 2.5, 0, Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.moveTo(flat ? 17 : 12, 0);
  ctx.lineTo(flat ? 10 : 6, -4); ctx.lineTo(flat ? 10 : 6, 4);
  ctx.closePath(); ctx.fill();
  ctx.restore();

  if (p.knife_remaining > 0) {
    const progress = 1 - Math.min(1, p.knife_remaining / .3);
    const swingAngle = (p.knife_angle ?? angle) - .9 + progress * 1.8;
    ctx.save(); ctx.translate(x, y);
    ctx.strokeStyle = p.knife_impact ? "#ffd46a" : "#e6f6ff";
    ctx.lineWidth = 3; ctx.globalAlpha = 1 - progress * .55;
    ctx.beginPath(); ctx.arc(0, 0, 23, swingAngle - .55, swingAngle); ctx.stroke();
    ctx.rotate(swingAngle);
    ctx.fillStyle = "#e6f6ff";
    ctx.beginPath(); ctx.moveTo(30, 0); ctx.lineTo(13, -3); ctx.lineTo(13, 3); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#617d85"; ctx.fillRect(7, -2, 7, 4);
    if (p.knife_impact) {
      ctx.strokeStyle = "#ffd46a"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(29, -5); ctx.lineTo(34, -9);
      ctx.moveTo(31, 0); ctx.lineTo(38, 0); ctx.moveTo(29, 5); ctx.lineTo(34, 9); ctx.stroke();
    }
    ctx.restore();
  }
  if (p.panting) {
    const pulse = (Math.sin(performance.now() / 160) + 1) / 2;
    ctx.save(); ctx.strokeStyle = "#ffd46a"; ctx.globalAlpha = .2 + pulse * .35;
    ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 15 + pulse * 5, 0, Math.PI*2); ctx.stroke(); ctx.restore();
    label(x+13,y-64,"FIATONE - RUMORE","#ffd46a");
  }

  if (p.treatment) label(x+13,y-50,`${p.treatment === "BANDAGE" ? "BENDA" : "KIT MEDICO"} ${p.treatment_remaining.toFixed(1)}s`,"#62f2ad");
  else if (p.regen_remaining > 0) label(x+13,y-50,"RECUPERO SALUTE","#62f2ad");
  if (p.bleeding > 0) label(x+13,y-36,"SANGUINAMENTO","#ff5e65");
  if (p.boost_remaining > 0) label(x+13,y-22,`${p.fear ? "ADRENALINA" : "SCATTO"} ${p.boost_remaining.toFixed(1)}s`,"#62d7ff");
  else if (p.exhausted) label(x+13,y-22,"ESAUSTO","#ffcf67");

  if (p.compliant) label(x+10,y-12,"COMPLY","#f4f4f4");
  else if (p.rising) label(x+10,y-12,"MI RIALZO...","#62d7ff");
  else if (p.knockdown_remaining > 0) label(x+10,y-12,p.dropped_item ? "ATTERRATO! OGGETTO CADUTO" : "ATTERRATO!","#ff735e");
  else if (p.prone) label(x+10,y-12,"A TERRA - MUOVITI / SPAZIO PER RIALZARSI","#62d7ff");
  else if (p.crouched) label(x+10,y-12,"ACCOVACCIATO","#62d7ff");
  if (p.injured) label(x+13,y-78,`INFORTUNIO · ZOPPIA · C ${Math.ceil(p.crouch_remaining)}s`,"#ff735e");
  if (p.fear) label(x+13,y-92,p.surrendering ? "PANICO · MI COSTITUISCO" : "PAURA · COMANDI INVERTITI","#ffcf67");

  if (state.gameplay?.spawn_protected || (state.gameplay?.spawn_grace || 0) > 0) {
    ctx.strokeStyle = "rgba(98,215,255,.75)";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x,y,15,0,Math.PI*2); ctx.stroke();
  }

  if (p.bloody) {
    ctx.strokeStyle="rgba(190,25,48,.75)"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.arc(x,y,12,0,Math.PI*2); ctx.stroke();
  }

  if (state.takedown) {
    const ratio=Math.min(1,(state.takedown.progress||0)/(state.takedown.duration||3));
    ctx.strokeStyle="#62f2ad"; ctx.lineWidth=3;
    ctx.beginPath(); ctx.arc(x,y,15,-Math.PI/2,-Math.PI/2+ratio*Math.PI*2); ctx.stroke();
    label(x+17,y,`CHOKE ${Math.ceil((1-ratio)*3)}s`,"#62f2ad");
  } else if (state.takedown_candidate) {
    label(x+13,y+15,state.takedown_candidate.can_lethal === false ? "TIENI F · STORDISCI" : "F · COLTELLO / TIENI F · STORDISCI",playerColor(state));
  }


}

function drawExit(t) {
  const e=state.map.exit;
  if (!e) return;
  const x=X(e.x,t),y=Y(e.y,t);
  ctx.strokeStyle="#62f2ad"; ctx.lineWidth=2;
  ctx.strokeRect(x-8,y-8,16,16);
  label(x+10,y+14,"EXIT","#62f2ad");
}

function drawBorder(t) {
  ctx.strokeStyle = alertPresentation(state).alarm ? "#a52942" : "#2d4c43"; ctx.lineWidth=1;
  ctx.strokeRect(t.ox,t.oy,t.w*t.s,t.h*t.s);
  ctx.strokeStyle=alertPresentation(state).alarm ? "#cb7078" : "#93b782";
  ctx.lineWidth=2;
  for (const [x,y,sx,sy] of [[t.ox,t.oy,1,1],[X(t.w,t),t.oy,-1,1],[t.ox,Y(t.h,t),1,-1],[X(t.w,t),Y(t.h,t),-1,-1]]) {
    ctx.beginPath(); ctx.moveTo(x+sx*16,y); ctx.lineTo(x,y); ctx.lineTo(x,y+sy*16); ctx.stroke();
  }
  ctx.font="8px Consolas, monospace"; ctx.fillStyle="#6f926c"; ctx.textAlign="center";
  for (let x=4; x<t.w; x+=4) {
    const px=X(x,t); ctx.fillText(String(x).padStart(2,"0"),px,t.oy-8);
    ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(px,t.oy); ctx.lineTo(px,t.oy+4); ctx.stroke();
  }
}

function drawRadarOverlay(t) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(t.ox,t.oy,t.w*t.s,t.h*t.s);
  ctx.clip();
  const alarm = alertPresentation(state).alarm;
  ctx.fillStyle = alarm ? "rgba(255,35,62,.09)" : "rgba(98,242,173,.025)";
  for (let y=t.oy; y<t.oy+t.h*t.s; y+=4) {
    ctx.fillRect(t.ox,y,t.w*t.s,1);
  }
  const sweep=((performance.now()/26)%(t.w*t.s));
  const grad=ctx.createLinearGradient(t.ox+sweep-45,0,t.ox+sweep+5,0);
  grad.addColorStop(0,"rgba(98,242,173,0)");
  grad.addColorStop(1, alarm ? "rgba(255,35,62,.18)" : "rgba(98,242,173,.055)");
  ctx.fillStyle=grad;
  ctx.fillRect(t.ox+sweep-45,t.oy,50,t.h*t.s);
  ctx.restore();


}

function label() {}

connect();


const itemNames = {AMMO:"Munizioni",KNIFE:"Coltello",PISTOL:"Pistola",RIFLE:"Fucile",SHOTGUN:"Canna liscia",SMG:"Mitra",ONE_HAND_SMG:"Mitra a una mano",GRENADE:"Granata",SNACK:"Snack",BANDAGE:"Benda",MEDKIT:"Kit medico",CASING:"Bossolo",MAGAZINE:"Caricatore vuoto",FLASH:"Flashbang",SMOKE:"Fumogeno",HAZMAT:"Protezione",SAMPLE_VIAL:"Provetta sterile",SECURITY_CARD:"Scheda sicurezza",
  PISTOL_SUPPRESSED:"Pistola silenziata",PISTOL_SUBSONIC:"Pistola subsonica",RIFLE_SUPPRESSED:"Fucile silenziato",RIFLE_SUBSONIC:"Fucile subsonico",SMG_SUPPRESSED:"Mitra silenziato",SMG_SUBSONIC:"Mitra subsonico"};
const itemPaths = {
  SNACK:"M5 3h14v18H5z M5 7h14 M8 10h8 M8 14h8 M5 18h14",
  GRENADE:"M9 3h6v4 M15 4h4v5 M8 8h8l3 5-2 8H7l-2-8z",
  SHOTGUN:"M2 9h20v3H9l-3 8H2l3-8 M13 13h7v3h-7",
  SMG:"M3 7h17v6H9l-2 7H3l2-7 M12 13v7h4v-7 M20 10h3",
  ONE_HAND_SMG:"M3 6h18v7H10l-2 8H4l2-8 M14 13v6h4v-6",
  HAZMAT:"M8 3h8l2 5-3 3 4 10h-5l-2-6-2 6H5l4-10-3-3z M9 5h6v3H9z",
  SAMPLE_VIAL:"M9 3h6v3H9z M10 6v11a2 2 0 004 0V6 M10 12h4",
  SECURITY_CARD:"M3 6h18v13H3z M6 9h4v4H6z M13 10h5 M6 16h12",
  AMMO:"M4 5h6v16H4z M5 5l2-3 2 3 M14 5h6v16h-6z M15 5l2-3 2 3",
  KNIFE:"M5 20l5-5 3 3-5 5z M10 15L21 3c2 7-2 10-8 15",
  PISTOL:"M3 7h19v6H12l-2 8H5l2-8H3z M15 10h7",
  RIFLE:"M2 9h17v4H9l-2 7H3l2-7H2z M19 11h4 M12 13v5h4v-5",
  BANDAGE:"M5 3h14v18H5z M5 8h14 M5 16h14 M10 10v4 M14 10v4",
  MEDKIT:"M3 7h18v14H3z M8 7V3h8v4 M12 10v8 M8 14h8",
  CASING:"M8 3h8v18H8z M8 7h8 M7 21h10",
  MAGAZINE:"M7 3h10l3 17H9z M9 8h8 M10 12h8 M11 16h8",
  FLASH:"M8 7h9v13H8z M10 7V3h5v4 M17 3h4 M4 8L2 6 M20 10l3-1",
  SMOKE:"M8 9h9v12H8z M10 9V6h5v3 M12 6c-5-1 4-3 0-5",
};
function itemIcon(kind) {
  return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${itemPaths[kind] || itemPaths[kind.replace(/_(SUPPRESSED|SUBSONIC)$/, "")] || itemPaths.CASING}"/></svg>`;
}
function selectBackpackItem(kind) {
  if (!itemNames[kind]) return;
  selectedItem=kind;
  send(inputPacket());
  canvas.focus({preventScroll:true});
}
const backpack = document.getElementById("backpackItems");
if (backpack) backpack.onclick = event => {
  const button = event.target.closest("button[data-item]");
  if (button && !button.disabled) selectBackpackItem(button.dataset.item);
};
const dropItemButton = document.getElementById("dropItemBtn");
if (dropItemButton) dropItemButton.onclick = () => {
  dropItemPulse = true;
  send(inputPacket());
  canvas.focus({preventScroll:true});
};
function refreshBackpack(gear) {
  const container=document.getElementById("backpackItems");
  if (!container) return;
  const entries=[];
  if (gear.knife) entries.push(["KNIFE",gear.item_quantities?.KNIFE ?? 1]);
  for (const [kind,key] of [["HAZMAT","hazmat"],["SAMPLE_VIAL","sample_vial"],["SECURITY_CARD","security_card"]]) if (gear[key]) entries.push([kind,gear.item_quantities?.[kind] ?? 1]);
  for (const weapon of gear.weapons || (gear.weapon ? [gear.weapon] : [])) entries.push([weapon,gear.weapon_ammo?.[weapon] ?? gear.ammo ?? 0]);
  for (const [kind,key] of [["AMMO","ammo"],["BANDAGE","bandages"],["MEDKIT","medkits"],["CASING","casings"],["MAGAZINE","magazines"],["FLASH","flash"],["SMOKE","smoke"],["GRENADE","grenades"],["SNACK","snacks"]]) {
    const count = kind === "AMMO" ? gear.loose_ammo ?? gear.ammo_stock ?? gear.ammo : gear[key];
    if (count>0) entries.push([kind,count]);
  }
  const current=gear.selected || "KNIFE";
  const capacity=document.getElementById("backpackCapacity");
  if (capacity) capacity.textContent=`${gear.inventory_slots ?? entries.length}/${gear.inventory_capacity ?? 10} POSTI · ${(gear.backpack_weight ?? 0).toFixed(1)} kg · MOBILITÀ ${Math.round((gear.weapon_speed_multiplier ?? 1)*(gear.backpack_speed_multiplier ?? 1)*100)}%`;
  if (dropItemButton) dropItemButton.disabled=!entries.some(([kind])=>kind===current);
  const markup=entries.map(([kind,count])=>`<button type="button" class="pack-slot ${kind===current ? "selected" : ""}" data-item="${kind}" aria-pressed="${kind===current}" title="Seleziona ${itemNames[kind]}">${itemIcon(kind)}<span>${itemNames[kind]}</span><b>${count}</b></button>`).join("") || '<span class="pack-empty">Zaino vuoto</span>';
  const renderKey=JSON.stringify({entries,current});
  if (container.dataset.renderKey!==renderKey) {
    container.innerHTML=markup;
    container.dataset.renderKey=renderKey;
  }
  const selected=document.getElementById("selectedItemLabel");
  if (selected) selected.textContent=current === "NONE" ? "Nessun oggetto equipaggiato" : `${itemNames[current] || "Nessun oggetto"} · ${current === "SNACK" ? "E · +20% vigore subito · non cura la salute" : current === "HAZMAT" ? "Indossata" : current === "SAMPLE_VIAL" ? "E sul campione per prelevare" : current === "SECURITY_CARD" ? "Accesso R&D" : current === "AMMO" ? "Scorta munizioni" : current === "KNIFE" ? "F attacca · tieni F per soffocare" : ["CASING","MAGAZINE"].includes(current) ? "Q per lanciare" : "E breve per usare"}`;
}
function refreshPlayerHUD(gear) {
  const weaponStatus = document.getElementById("weaponStatus");
  if (weaponStatus) {
    const gun = gear.firearm;
    weaponStatus.textContent = gun ? `${state.player.draw_remaining > 0 ? `ESTRAZIONE · ${state.player.draw_remaining.toFixed(1)}s\n` : gun.reload_mode ? `RICARICA ${gun.reload_mode === "FAST" ? "RAPIDA" : "TATTICA"} · ${gun.reload_remaining.toFixed(1)}s\n` : ""}CAR ${gun.magazine}/${gun.capacity} + CANNA ${gun.chamber} · SCORTA ${gun.reserve}${gun.subsonic ? " · SUB" : gun.suppressed ? " · SIL" : ""}\n${(state.player.weapon_weight || 0).toFixed(1)} kg · MOBILITÀ ${Math.round((state.player.weapon_speed_multiplier ?? 1)*100)}%` : "";
    weaponStatus.hidden = !gun;
  }
  for (const [key,id] of [["health","playerHealth"],["stamina","playerStamina"]]) {
    const bar=document.getElementById(id+"Fill"), value=document.getElementById(id+"Value");
    const percent=Math.round(Math.max(0,Math.min(1,gear[key] ?? 1))*100);
    if (bar) bar.style.width=`${percent}%`;
    if (value) value.textContent=`${percent}%`;
  }
  const status=document.getElementById("playerCondition");
  if (status) status.textContent=gear.treatment ? `${gear.station_id ? "TIENI E · " : ""}${gear.treatment==="BANDAGE" ? "Benda" : "Kit medico"} · ${gear.treatment_remaining.toFixed(1)}s` : gear.smoke_exposure>0 ? (gear.exhausted ? "FUMO · ESAUSTO" : "FUMO · VIGORE IN CALO") : gear.bleeding>0 ? "FERITO · SANGUINAMENTO" : gear.regen_remaining>0 ? "RECUPERO GRADUALE" : gear.diving ? "TUFFO" : "OPERATIVO";
  if (status && (gear.injured || gear.fear)) status.textContent = [gear.injured ? `INFORTUNIO · SERVE MEDIKIT · C ${Math.ceil(gear.crouch_remaining)}s` : "",gear.fear ? gear.snack_eaten ? `SNACK ASSUNTO · CALMA ${Number(gear.calm_seconds || 0).toFixed(1)}/3s · SERVONO SALUTE E VIGORE 100%` : "PAURA · SALUTE E VIGORE 100% + SNACK" : ""].filter(Boolean).join(" · ");
  const effects = document.getElementById("vitalEffects");
  if (effects) {
    const lines = [];
    if (gear.snack_effect_remaining > 0) lines.push(`SNACK · VIGORE +${Math.round((gear.snack_stamina_gain || 0)*100)}% SUBITO · SALUTE INVARIATA`);
    if (gear.stamina < .999 && gear.still_seconds > .5) lines.push(gear.recovery_delay > 0 ? `RECUPERO VIGORE TRA ${gear.recovery_delay.toFixed(1)}s` : gear.exhausted_recovery ? "RECUPERO DOPO ESAURIMENTO · RITMO NORMALE" : `RECUPERO VIGORE ×${(gear.stamina_recovery_multiplier || 1).toFixed(1)} · RESTA FERMO PER ACCELERARE`);
    if (gear.superficial_wound) lines.push(gear.healing_blocked ? "FERITA SUPERFICIALE · GUARIGIONE SOSPESA DAI MALUS" : gear.wound_healing_wait > 0 ? `FERITA SUPERFICIALE · RIMARGINAZIONE TRA ${Math.ceil(gear.wound_healing_wait)}s` : "FERITA IN RIMARGINAZIONE · SALUTE +0,15%/s");
    if (gear.regen_remaining > 0) lines.push(`CURA GRADUALE · SALUTE +${Math.round(gear.regen_remaining*100)}% IN ${Math.ceil(gear.regen_seconds)}s`);
    const text = lines.join("\n");
    if (effects.textContent !== text) effects.textContent = text;
    effects.hidden = !text;
  }
}
