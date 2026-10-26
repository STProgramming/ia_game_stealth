const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function frontend() {
  const ids = ["game", "banner", "alertHud", "alertTitle", "alertDetail", "fullscreenBtn", "visionBtn", "levelLabel", "intelLabel", "intelProgress", "equipmentLabel", "radarCoords", "backpackItems", "selectedItemLabel", "playerHealthFill", "playerHealthValue", "playerStaminaFill", "playerStaminaValue", "playerCondition", "missionTitle", "missionObjective", "missionRequirements"];
  const handlers = {};
  const nodes = Object.fromEntries(ids.map(id => [id, {
    hidden: false, textContent: "", dataset: {}, style: {},
    setAttribute() {}, querySelector() {return {textContent: ""};},
    classList: {add() {}, remove() {}, toggle() {}},
    addEventListener(name, handler) { handlers["canvas:" + name] = handler; }, focus() {}, setPointerCapture() {},
    getBoundingClientRect: () => ({width: 1280, height: 720}),
  }]));
  const ctx = new Proxy({}, {get(target, prop) {
    if (prop in target) return target[prop];
    if (prop === "createLinearGradient" || prop === "createRadialGradient") return () => ({addColorStop() {}});
    return () => {};
  }});
  nodes.game.getContext = () => ctx;
  const sandbox = {
    document: {getElementById: id => nodes[id], body: {dataset: {}}, documentElement: {}},
    window: {addEventListener(name, handler) { handlers[name] = handler; }, devicePixelRatio: 1},
    location: {protocol: "http:", host: "localhost"},
    performance: {now: () => 0},
    setInterval() {}, setTimeout() {}, requestAnimationFrame() {},
    WebSocket: class {static OPEN = 1},
    ResizeObserver: class {observe() {}}, URLSearchParams,
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../frontend/app.js"), "utf8"), sandbox);
  return {sandbox, nodes, handlers};
}

function update(sandbox, data) {
  sandbox.incoming = data;
  vm.runInContext("state = incoming; refreshUI(); draw();", sandbox);
}

const base = {mode: "guard", status: "RUNNING", gameplay: {spawn_grace: 0}};

test("alert sequence is stable and never renders transient center messages", () => {
  const {sandbox, nodes} = frontend();
  for (const [phase, alert, title, detail] of [
    ["SPOTTED", "CONTACT", "SPOTTED", "Intruso avvistato"],
    ["RADIO", "CONTACT", "SPOTTED", "Chiamata alla radio in corso"],
    ["DISPATCH", "CONTACT", "ALLERTA", "Intervento della SWAT"],
    ["ACTIVE", "CONTACT", "ALLERTA", "Intervento della SWAT"],
    ["ACTIVE", "SEARCH", "EVASIONE", "Investigazione sull'ultimo punto noto"],
    ["ACTIVE", "SUSPICIOUS", "CAUTELA", "Guardia alta · Sorveglianza rafforzata"],
  ]) {
    update(sandbox, {...base, message: "HOLD G", security: {swat_phase: phase, alert, contact: {x: 10, y: 10}}});
    assert.equal(nodes.alertTitle.textContent, title);
    assert.equal(nodes.alertDetail.textContent, detail);
    assert.equal(nodes.banner.textContent, "");
  }
});

test("spawn protection hides all alarms", () => {
  const {sandbox, nodes} = frontend();
  update(sandbox, {...base, gameplay: {spawn_grace: 2}, security: {alert: "CONTACT", swat_phase: "RADIO"}});
  assert.equal(nodes.alertTitle.textContent, "NORMALE");
  assert.equal(sandbox.document.body.dataset.alarm, "false");
});

test("spawn alcove protection hides alarms after the countdown expires", () => {
  const {sandbox, nodes} = frontend();
  update(sandbox, {...base, gameplay: {spawn_grace: 0, spawn_protected: true}, security: {alert: "CONTACT", swat_phase: "RADIO"}});
  assert.equal(nodes.alertTitle.textContent, "NORMALE");
  assert.equal(sandbox.document.body.dataset.alarm, "false");
  update(sandbox, {...base, gameplay: {spawn_grace: 0, spawn_protected: false}, security: {alert: "CONTACT", swat_phase: "RADIO"}});
  assert.equal(nodes.alertHud.hidden, false);
});

test("frame drawing runs with the minimal fullscreen DOM", () => {
  const {sandbox} = frontend();
  update(sandbox, {...base, map: {width: 48, height: 30, grid: [[0, 1], [0, 0]], exit: {x: 1, y: 1}}, player: {x: 1, y: 1}, guards: [{x: 0, y: 0, alive: true, action: "PATROL"}], swat: []});
});


test("preview keyboard events work with and without physical key codes", () => {
  const {sandbox, handlers} = frontend();
  for (const event of [{key: "w", code: "KeyW"}, {key: "w", code: ""}, {key: "ArrowUp", code: ""}]) {
    handlers.keydown({...event, preventDefault() {}, repeat: false});
    assert.equal(vm.runInContext("inputPacket().up", sandbox), true);
    handlers.keyup(event);
    assert.equal(vm.runInContext("inputPacket().up", sandbox), false);
  }
});

test("pointer drag moves the player and release stops movement", () => {
  const {sandbox, handlers} = frontend();
  handlers["canvas:pointerdown"]({button:0, pointerId:1, clientX:100, clientY:100});
  handlers["canvas:pointermove"]({pointerId:1, clientX:130, clientY:100});
  assert.equal(vm.runInContext("inputPacket().right", sandbox), true);
  assert.equal(vm.runInContext("keyDirection().x", sandbox), 1);
  handlers["canvas:pointerup"]();
  assert.equal(vm.runInContext("inputPacket().right", sandbox), false);
});

test("C toggles crouch on each press, ignores repeat and survives release", () => {
  const {sandbox, handlers} = frontend();
  for (const event of [{key: "c", code: "KeyC"}, {key: "C", code: ""}]) {
    handlers.keydown({...event, preventDefault() {}, repeat: false});
    assert.equal(vm.runInContext("inputPacket().crouch", sandbox), true);
    handlers.keydown({...event, preventDefault() {}, repeat: true});
    handlers.keyup(event);
    assert.equal(vm.runInContext("inputPacket().crouch", sandbox), true);
    handlers.keydown({...event, preventDefault() {}, repeat: false});
    assert.equal(vm.runInContext("inputPacket().crouch", sandbox), false);
    handlers.keyup(event);
  }
  handlers.keydown({key: "Control", code: "ControlLeft", preventDefault() {}, repeat: false});
  assert.equal(vm.runInContext("inputPacket().crouch", sandbox), false);
});

test("player color shows takedown availability and silent choke progress", () => {
  const {sandbox} = frontend();
  assert.equal(vm.runInContext('playerColor({player:{},takedown_candidate:{guard_id:"G1"}})', sandbox), "#ffcf67");
  assert.equal(vm.runInContext('playerColor({player:{},takedown:{kind:"CHOKE"}})', sandbox), "#62f2ad");
  assert.equal(vm.runInContext('playerColor({player:{}})', sandbox), "#f0f4cc");
  assert.equal(vm.runInContext('playerColor({player:{},takedown_candidate:{can_lethal:false,can_stun:true}})', sandbox), "#62f2ad");
});

test("firing uses H or right mouse without starting movement drag", () => {
  const {sandbox, handlers} = frontend();
  handlers.keydown({key:"h",code:"KeyH",repeat:false,preventDefault(){}});
  assert.equal(vm.runInContext("inputPacket().fire",sandbox),true);
  handlers.keyup({key:"h",code:"KeyH"});
  assert.equal(vm.runInContext("inputPacket().fire",sandbox),false);
  handlers["canvas:pointerdown"]({button:2,pointerId:7,clientX:100,clientY:100});
  assert.equal(vm.runInContext("inputPacket().fire",sandbox),true);
  assert.equal(vm.runInContext("movementPointer",sandbox),null);
  handlers["canvas:pointerup"]();
  assert.equal(vm.runInContext("inputPacket().fire",sandbox),false);
});

test("empty inventory and map pickups render and local support is not labeled SWAT", () => {
  const {sandbox,nodes} = frontend();
  update(sandbox,{...base,security:{alert:"CONTACT",swat_phase:"LOCAL SUPPORT"},
    player:{x:1,y:1,weapon:null,ammo:0,diversions:0,scrap:0},
    map:{width:5,height:5,grid:Array.from({length:5},()=>Array(5).fill(0))},
    pickups:[{id:"supply",kind:"CASING",x:1,y:1,amount:2,source:"MAP"}]});
  assert.match(nodes.equipmentLabel.textContent,/DISARMATO/);
  assert.doesNotMatch(nodes.equipmentLabel.textContent,/ROTTAMI|D 0/);
  assert.match(nodes.alertDetail.textContent,/Supporto delle guardie/);
  assert.doesNotMatch(nodes.alertDetail.textContent,/SWAT/);
});

test("separate player drops render at their own map positions", () => {
  const {sandbox} = frontend();
  const labels = [];
  sandbox.captureLabel = (x, y, text) => labels.push({x, y, text});
  vm.runInContext(`
    label = captureLabel;
    state = {pickups: [
      {id:"player-drop-1",kind:"KNIFE",source:"PLAYER",x:2,y:3},
      {id:"player-drop-2",kind:"PISTOL",source:"PLAYER",x:8,y:9}
    ]};
    drawPickups({ox:0,oy:0,s:10});
  `, sandbox);
  assert.deepEqual(labels, [
    {x:26,y:25,text:"COLTELLO"},
    {x:86,y:85,text:"PISTOLA"}
  ]);
});

test("client prediction uses exhaustion fatigue and boost from server", () => {
  const {sandbox} = frontend();
  vm.runInContext(`
    state = {mode:"guard",status:"RUNNING",player:{health:1,exhausted:true,speed_multiplier:.7},
             gameplay:{walk_speed:1,sprint_speed:2},map:{grid:Array.from({length:20},()=>Array(20).fill(0))}};
    predictedPlayer = {x:5,y:5}; keys.add("KeyD"); keys.add("ShiftLeft");
    updatePredictedPlayer(1);
  `,sandbox);
  assert.equal(vm.runInContext("predictedPlayer.x", sandbox),5.7);
  vm.runInContext(`predictedPlayer.x=5; state.player.exhausted=false; state.player.boost_remaining=2.5; updatePredictedPlayer(1);`,sandbox);
  assert.ok(Math.abs(vm.runInContext("predictedPlayer.x", sandbox) - 7.31) < 1e-6);
});

test("injury disables sprint prediction and an exhausted crouch budget forces walking", () => {
  const {sandbox} = frontend();
  vm.runInContext(`
    state={mode:"guard",status:"RUNNING",player:{health:1,injured:true,can_sprint:false,can_crouch:false,speed_multiplier:.45},
      gameplay:{walk_speed:2,sprint_speed:4,crouch_speed:1},map:{grid:Array.from({length:20},()=>Array(20).fill(0))}};
    predictedPlayer={x:5,y:5}; keys.add("KeyD"); keys.add("ShiftLeft"); crouchToggled=true;
    updatePredictedPlayer(1);
  `,sandbox);
  assert.equal(vm.runInContext("predictedPlayer.x",sandbox),5.9);
});

test("fear uses authoritative movement and HUD explains injury and calming requirements", () => {
  const {sandbox,nodes} = frontend();
  vm.runInContext(`
    state={status:"RUNNING",player:{fear:true}}; predictedPlayer={x:5,y:5}; keys.add("KeyD");
    updatePredictedPlayer(1);
    refreshPlayerHUD({health:.68,stamina:.4,injured:true,crouch_remaining:42,fear:true});
  `,sandbox);
  assert.equal(vm.runInContext("predictedPlayer.x",sandbox),5);
  assert.match(nodes.playerCondition.textContent,/INFORTUNIO.*42s.*PAURA.*100%.*SNACK/);
});

test("new weapons snacks and explosive warnings render in the interface", () => {
  const {sandbox,nodes} = frontend();
  update(sandbox,{...base,player:{x:4,y:4,weapon:"SHOTGUN",weapons:["SHOTGUN","SMG","ONE_HAND_SMG"],selected:"SNACK",snacks:2,grenades:1},
    map:{width:10,height:10,grid:Array.from({length:10},()=>Array(10).fill(0))},
    grenades:[{x:5,y:4,target_x:6,target_y:4,radius:3,fuse:1.2}],blasts:[{x:8,y:8,radius:3,ttl:.4}]});
  assert.match(nodes.equipmentLabel.textContent,/CANNA LISCIA/);
  for (const kind of ["SHOTGUN","SMG","ONE_HAND_SMG","SNACK","GRENADE"]) assert.match(nodes.backpackItems.innerHTML,new RegExp('data-item="'+kind+'"'));
  assert.match(nodes.selectedItemLabel.textContent,/Snack/);
});

test("local diversion dialogue keeps normal alert and repeated dialogue appears in caution", () => {
  const {sandbox, nodes} = frontend();
  update(sandbox, {...base, security: {alert: "CALM", dialogue: {G1: "Vado a controllare."}}});
  assert.equal(nodes.alertTitle.textContent, "NORMALE");
  assert.equal(nodes.alertDetail.textContent, "Vado a controllare.");
  update(sandbox, {...base, security: {alert: "SUSPICIOUS", dialogue: {G1: "C'è qualcosa di strano.", G2: "Ti copro."}}});
  assert.equal(nodes.alertTitle.textContent, "CAUTELA");
  assert.match(nodes.alertDetail.textContent, /qualcosa di strano/);
  assert.match(nodes.alertDetail.textContent, /Ti copro/);
});

test("outcome banner reports training completion and failures", () => {
  const {sandbox, nodes} = frontend();
  for (const [status, text] of [["TRAINING", "IA in addestramento"], ["TRAINED", "IA aggiornata"], ["FAILED", "Addestramento non riuscito"]]) {
    update(sandbox, {...base, status: "CAPTURED", learning: {status}});
    assert.ok(nodes.banner.textContent.includes(text));
  }
});


test("mouse backpack selection replaces J and K and HUD shows player vitals", () => {
  const {sandbox, nodes, handlers} = frontend();
  handlers.keydown({key:"j", code:"KeyJ", repeat:false, preventDefault() {}});
  handlers.keydown({key:"k", code:"KeyK", repeat:false, preventDefault() {}});
  assert.equal(vm.runInContext("inputPacket().use_bandage || inputPacket().use_medkit", sandbox), false);
  nodes.backpackItems.onclick({target:{closest:()=>({disabled:false,dataset:{item:"BANDAGE"}})}});
  assert.equal(vm.runInContext("inputPacket().selected_item", sandbox), "BANDAGE");
  update(sandbox, {...base, map:{width:10,height:10,grid:Array.from({length:10},()=>Array(10).fill(0))},
    player:{x:2,y:2,health:.5,stamina:.3,knife:true,bandages:2,selected:"BANDAGE",treatment:"BANDAGE",treatment_remaining:.5},
    pickups:[{id:"med",kind:"MEDKIT",fixed:true,x:4,y:4,amount:1,source:"MAP"}]});
  assert.match(nodes.backpackItems.innerHTML, /<svg/);
  assert.match(nodes.backpackItems.innerHTML, /data-item="BANDAGE"/);
  assert.match(nodes.selectedItemLabel.textContent, /Benda/);
  assert.equal(nodes.playerHealthValue.textContent,"50%");
  assert.equal(nodes.playerStaminaFill.style.width,"30%");
});

test("F distinguishes held choke from a quick knife tap and space dives", () => {
  const {sandbox, handlers} = frontend();
  handlers.keydown({key:"f",code:"KeyF",repeat:false,preventDefault(){}});
  assert.equal(vm.runInContext("inputPacket().attack_held",sandbox),true);
  assert.equal(vm.runInContext("inputPacket().lethal",sandbox),false);
  handlers.keyup({key:"f",code:"KeyF"});
  assert.equal(vm.runInContext("inputPacket().lethal",sandbox),true);
  handlers.keydown({key:" ",code:"Space",repeat:false,preventDefault(){}});
  assert.equal(vm.runInContext("inputPacket().dive",sandbox),true);
  assert.equal(vm.runInContext("inputPacket().compliant",sandbox),false);
});

test("enemy health stamina and armor bars require direct observation", () => {
  const {sandbox}=frontend();
  vm.runInContext("var fills=0; ctx.fillRect=()=>fills++; drawVitals({x:2,y:2,health:1,stamina:1,armor:1,observed:false},{s:10,ox:0,oy:0});",sandbox);
  assert.equal(vm.runInContext("fills",sandbox),0);
  vm.runInContext("drawVitals({x:2,y:2,health:1,stamina:1,armor:1,observed:true},{s:10,ox:0,oy:0});",sandbox);
  assert.equal(vm.runInContext("fills",sandbox),6);
});


test("unchanged snapshots preserve backpack buttons for reliable mouse clicks", () => {
  const {sandbox,nodes}=frontend();
  let updates=0;
  let markup="";
  Object.defineProperty(nodes.backpackItems,"innerHTML",{get:()=>markup,set:value=>{updates++;markup=value;}});
  update(sandbox,{...base,player:{x:2,y:2,knife:true,selected:"KNIFE"}});
  update(sandbox,{...base,player:{x:2,y:2,knife:true,selected:"KNIFE"}});
  assert.equal(updates,1);
});

test("pointer movement cannot rotate vision and movement facing persists at rest", () => {
  const {sandbox, handlers} = frontend();
  update(sandbox,{...base,player:{x:4,y:4},map:{width:10,height:10,grid:Array.from({length:10},()=>Array(10).fill(0))}});
  const initial = vm.runInContext("inputPacket()",sandbox);
  handlers["canvas:pointermove"]({clientX:30,clientY:400});
  assert.equal(vm.runInContext("inputPacket().aim_y",sandbox),initial.aim_y);
  handlers.keydown({key:"w",code:"KeyW",preventDefault(){}});
  assert.ok(vm.runInContext("inputPacket().aim_y",sandbox) < 4);
  handlers.keyup({key:"w",code:"KeyW"});
  assert.ok(vm.runInContext("inputPacket().aim_y",sandbox) < 4);
});

test("player draws without health and stamina bars below the avatar", () => {
  const {sandbox} = frontend();
  update(sandbox,{...base,player:{x:4,y:4,health:1,stamina:1}});
  vm.runInContext("drawVitals = () => {throw Error('Unexpected avatar bars')}; drawPlayer({x:4,y:4})",sandbox);
});

test("smoke exposure and exhaustion are visible in the player HUD", () => {
  const {sandbox,nodes} = frontend();
  update(sandbox,{...base,player:{x:4,y:4,health:.9,stamina:0,exhausted:true,smoke_exposure:1}});
  assert.match(nodes.playerCondition.textContent,/FUMO.*ESAUSTO/);
  assert.equal(nodes.playerStaminaValue.textContent,'0%');
});

test("campaign briefing shows mission and lab equipment requirements", () => {
  const {sandbox,nodes}=frontend();
  update(sandbox,{...base,level:3,mission:{level:3,title:'LABORATORI',objective:'Preleva il campione',requirements:[{name:'Protezione biologica',ready:true},{name:'Provetta sterile',ready:false}]},player:{x:4,y:4,hazmat:true,sample_vial:true}});
  assert.match(nodes.missionTitle.textContent,/3.*LABORATORI/);
  assert.equal(nodes.missionObjective.textContent,'Preleva il campione');
  assert.match(nodes.missionRequirements.textContent,/✓ Protezione.*○ Provetta/);
  assert.match(nodes.backpackItems.innerHTML,/Protezione/);
  assert.match(nodes.backpackItems.innerHTML,/Provetta sterile/);
});


test("prediction stops while prone and while getting up", () => {
  const {sandbox} = frontend();
  for (const posture of [{prone:true}, {rising:true}, {diving:true}]) {
    sandbox.posture = posture;
    vm.runInContext("state = {status:'RUNNING', player:{x:2,y:2,...posture}}; predictedPlayer={x:2,y:2}; keys.add('KeyD'); updatePredictedPlayer(.2);", sandbox);
    assert.equal(vm.runInContext("predictedPlayer.x", sandbox), 2);
  }
});


test("knife swing and heavy breathing effects render with player postures", () => {
  const {sandbox} = frontend();
  for (const posture of [{}, {crouched:true}, {prone:true}, {rising:true}]) {
    update(sandbox, {...base, player:{x:4,y:4,health:1,stamina:0,panting:true,knife_remaining:.15,knife_angle:1,knife_impact:true,...posture}});
  }
});


test("gunfire displays general alert immediately during the radio call", () => {
  const {sandbox,nodes} = frontend();
  update(sandbox, {...base,security:{source:"GUNFIRE",alert:"CONTACT",swat_phase:"SPOTTED"}});
  assert.equal(nodes.alertTitle.textContent,"ALLERTA");
  assert.equal(sandbox.document.body.dataset.alarm,"true");
  assert.equal(vm.runInContext('agentPalette({confidence:0,combat_alert:true}).color',sandbox),"rgb(255,89,103)");
  update(sandbox, {...base,security:{alert:"SUSPICIOUS",swat_phase:"QUIET",high_guard:true,threat_level:1}});
  assert.equal(nodes.alertTitle.textContent,"CAUTELA");
  assert.equal(sandbox.document.body.dataset.alarm,"false");
});
