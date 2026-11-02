const {test, before} = require("node:test");
const assert = require("node:assert/strict");

let THREE, graphics, spatial;
before(async () => {
  [THREE, graphics, spatial] = await Promise.all([
    import("../frontend/vendor/three/three.module.js"),
    import("../frontend/scene3d.js"), import("../frontend/spatial.js"),
  ]);
});

function map(width = 12, height = 10) {
  return {width, height, grid: Array.from({length: height}, () => Array(width).fill(0))};
}
function sceneContext() {
  const context = Object.create(graphics.ShadowScene.prototype);
  Object.assign(context, {scene: new THREE.Scene(), matrix: new THREE.Matrix4(),
    actors: new Map(), cones: new Map(), items: new Map(), effects: new Map()});
  return context;
}

test("3D walls occupy exactly the server's solid cells and floor boundaries", () => {
  const layout = map(); layout.grid[2][3] = 1; layout.grid[7][9] = 1;
  const facility = graphics.buildFacility(layout, {labels: false});
  const {walls, cells} = facility.userData;
  assert.equal(walls.count, 2);
  assert.deepEqual(cells, [{x: 3, y: 2}, {x: 9, y: 7}]);
  const position = new THREE.Vector3(), matrix = new THREE.Matrix4();
  cells.forEach((cell, i) => {
    walls.getMatrixAt(i, matrix); position.setFromMatrixPosition(matrix);
    assert.deepEqual(position.toArray(), [cell.x, 1, cell.y]);
  });
  const floor = facility.children[0];
  assert.equal(floor.position.x - floor.geometry.parameters.width / 2, -.5);
  assert.equal(floor.position.z + floor.geometry.parameters.depth / 2, layout.height - .5);
});

test("nearby camera-side walls lower and return to full height after moving away", () => {
  const layout = map(); layout.grid[6][5] = 1; layout.grid[1][5] = 1;
  const context = sceneContext(); context.facility = graphics.buildFacility(layout, {labels: false});
  context.updateWalls({x: 5, y: 5});
  const matrix = new THREE.Matrix4(), scale = new THREE.Vector3();
  context.facility.userData.walls.getMatrixAt(1, matrix); scale.setFromMatrixScale(matrix);
  assert.ok(Math.abs(scale.y - .42) < 1e-6);
  context.facility.userData.walls.getMatrixAt(0, matrix); scale.setFromMatrixScale(matrix);
  assert.equal(scale.y, 2);
  context.updateWalls({x: 10, y: 9});
  context.facility.userData.walls.getMatrixAt(1, matrix); scale.setFromMatrixScale(matrix);
  assert.equal(scale.y, 2);
});

test("overview camera fits every map corner in portrait and landscape", () => {
  for (const [width, height] of [[32, 20], [64, 40]]) for (const aspect of [.5, 1, 16 / 9, 2.4]) {
    const camera = new THREE.PerspectiveCamera(45, aspect, .1, 250);
    const target = new THREE.Vector3((width - 1) / 2, 0, (height - 1) / 2);
    const elevation = spatial.overviewHeight(width, height, aspect);
    camera.position.set(target.x, elevation, target.z + elevation * .72);
    camera.lookAt(target); camera.updateMatrixWorld();
    for (const x of [-.5, width - .5]) for (const z of [-.5, height - .5]) {
      const screen = new THREE.Vector3(x, 0, z).project(camera);
      assert.ok(Math.abs(screen.x) <= 1 && Math.abs(screen.y) <= 1 && screen.z < 1,
        `Map ${width}x${height}, aspect ${aspect}: ${screen.toArray()}`);
    }
  }
});

test("3D vision is clipped by walls, locked doors and map edges", () => {
  const layout = map(); layout.grid.forEach(row => { row[8] = 1; });
  const guard = {x: 3, y: 5, angle: 0, fov: 90, vision_range: 12};
  const points = spatial.clippedVision(guard, layout.grid);
  assert.ok(points.every(p => p.x < 7.5 && p.y >= -.5 && p.y < 9.5));
  const doors = [{x: 6, y: 5, locked: true}];
  const blocked = spatial.clippedVision({...guard, fov: 0}, layout.grid, doors);
  assert.ok(blocked.every(p => p.x < 5.5));
  doors[0].locked = false;
  assert.ok(spatial.clippedVision({...guard, fov: 0}, layout.grid, doors).every(p => p.x > 7));
  assert.ok(spatial.clippedVision({...guard, vision_range: 0}, layout.grid).every(p => p.x === 3 && p.y === 5));
});

test("operative orientation agrees with movement and firing in every direction", () => {
  const actor = graphics.createOperative();
  for (const angle of [0, Math.PI / 2, Math.PI, -Math.PI / 2]) {
    graphics.updateOperative(actor, {x: 4, y: 6, angle}, 1);
    const forward = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), actor.rotation.y);
    assert.ok(Math.abs(forward.x - Math.cos(angle)) < 1e-6);
    assert.ok(Math.abs(forward.z - Math.sin(angle)) < 1e-6);
    assert.deepEqual(actor.position.toArray(), [4, 0, 6]);
  }
});

test("posture, animation, equipment and fallen units reflect server snapshots", () => {
  const actor = graphics.createOperative();
  graphics.updateOperative(actor, {x: 4, y: 6, crouched: true, knife: true}, 1);
  assert.ok(actor.userData.rig.scale.y < .65);
  assert.equal(actor.userData.knife.visible, true);
  graphics.updateOperative(actor, {x: 4.1, y: 6, prone: true, weapon: "RIFLE", shield: true}, 1);
  assert.ok(Math.abs(actor.userData.rig.rotation.x - Math.PI / 2) < .001);
  assert.equal(actor.userData.weapon.visible, true);
  assert.equal(actor.userData.shield.visible, true);
  graphics.updateOperative(actor, {x: 4.2, y: 6, down: true}, 1);
  assert.equal(actor.userData.marker.visible, false);
  graphics.updateOperative(actor, {x: 4.3, y: 6}, 1);
  assert.ok(actor.userData.rig.rotation.x < .001);
  assert.ok(actor.userData.rig.scale.y > .99);
  assert.notEqual(actor.userData.legs[0].rotation.x, 0);
});

test("weapon extraction raises the model into firing position and switches its silhouette", () => {
  const actor=graphics.createOperative();
  graphics.updateOperative(actor,{x:2,y:3,weapon:"RIFLE",draw_remaining:1,draw_duration:1},1);
  assert.ok(actor.userData.weapon.rotation.x>1);
  assert.ok(actor.userData.weapon.position.y<.7);
  const rifle=actor.userData.model;
  graphics.updateOperative(actor,{x:2,y:3,weapon:"PISTOL_SUPPRESSED",draw_remaining:0},1);
  assert.equal(rifle.parent,null);
  assert.ok(actor.userData.model.userData.length<rifle.userData.length);
  assert.equal(actor.userData.suppressor.visible,true);
  assert.equal(actor.userData.weapon.rotation.x,0);
  assert.equal(actor.userData.weapon.position.y,.92);
});

test("SWAT peek inclines the upper body at the physical position and keeps feet planted", () => {
  const actor = graphics.createOperative("swat");
  const snapshot = {x: 5, y: 7, angle: .3, weapon: "SMG", lean_amount: 0};
  graphics.updateOperative(actor, snapshot, 1);
  actor.updateMatrixWorld(true);
  const feet = actor.userData.legs.map(leg => leg.getWorldPosition(new THREE.Vector3()).toArray());
  for (const lean of [-1, 1]) {
    graphics.updateOperative(actor, {...snapshot, lean_amount: lean, formation: lean < 0 ? "PEEK LEFT" : "PEEK RIGHT"}, 1);
    actor.updateMatrixWorld(true);
    assert.deepEqual(actor.position.toArray(), [5, 0, 7]);
    assert.ok(Math.abs(actor.userData.leanPivot.rotation.z + lean * .22) < .001);
    assert.deepEqual(actor.userData.legs.map(leg => leg.getWorldPosition(new THREE.Vector3()).toArray()), feet);
    assert.equal(actor.userData.weapon.parent, actor.userData.upperBody);
    assert.ok(actor.userData.arms.every(arm => arm.parent === actor.userData.upperBody));
  }
  graphics.updateOperative(actor, {...snapshot, lean_amount: 1, down: true}, 1);
  assert.ok(Math.abs(actor.userData.leanPivot.rotation.z) < .001);
});

test("automatic fire makes the shield brace and lowers its pistol until suppression expires", () => {
  const actor = graphics.createOperative("swat");
  const snapshot = {x: 5, y: 7, weapon: "PISTOL", shield: true, observed: true, aiming: true};
  graphics.updateOperative(actor, snapshot, 1);
  graphics.updateOperative(actor, {...snapshot, suppression_timer: .8, lean_amount: 1}, 1);
  assert.ok(actor.userData.leanPivot.rotation.x > .11);
  assert.ok(Math.abs(actor.userData.leanPivot.rotation.z) < .001);
  assert.ok(actor.userData.shield.rotation.x > .13);
  assert.equal(actor.userData.weapon.rotation.x, .5);
  assert.equal(actor.userData.weapon.position.y, .8);
  assert.equal(actor.userData.arms[1].rotation.x, -.35);
  assert.equal(actor.userData.shield.visible, true);
  graphics.updateOperative(actor, {...snapshot, suppression_timer: 0}, 1);
  assert.ok(actor.userData.leanPivot.rotation.x < .001);
  assert.ok(actor.userData.shield.rotation.x < .001);
  assert.equal(actor.userData.weapon.rotation.x, 0);
  assert.equal(actor.userData.weapon.position.y, .92);
});

test("SWAT sidearm draw and reload retract the lean while retaining their own weapon animation", () => {
  const actor = graphics.createOperative("swat");
  const snapshot = {x: 5, y: 7, weapon: "SMG", lean_amount: 1};
  graphics.updateOperative(actor, snapshot, 1);
  const primary = actor.userData.model;
  graphics.updateOperative(actor, {...snapshot, weapon: "PISTOL", secondary_weapon: "PISTOL", draw_remaining: .42, draw_duration: .42}, 1);
  assert.equal(primary.parent, null);
  assert.ok(actor.userData.model.userData.length < primary.userData.length);
  assert.ok(actor.userData.weapon.rotation.x > 1);
  assert.ok(Math.abs(actor.userData.leanPivot.rotation.z) < .001);
  graphics.updateOperative(actor, {...snapshot, weapon: "PISTOL", firearm: {reload_mode: "FAST", reload_remaining: .5}}, 1);
  assert.ok(Math.abs(actor.userData.leanPivot.rotation.z) < .001);
  assert.equal(actor.userData.weapon.rotation.x, -.35);
  graphics.updateOperative(actor, {...snapshot, weapon: "PISTOL", draw_remaining: 0}, 1);
  assert.ok(actor.userData.leanPivot.rotation.z < -.21);
  assert.equal(actor.userData.weapon.rotation.x, 0);
});

test("wall fixtures disappear with cutaway walls and never occupy walkable cells", () => {
  const layout=map(); layout.grid[4][4]=1;
  const context=sceneContext(); context.facility=graphics.buildFacility(layout,{labels:false});
  const {fixtures,panels}=context.facility.userData;
  assert.equal(fixtures.length,4);
  assert.ok(fixtures.every(f=>Math.abs(f.x-4)<=.5&&Math.abs(f.y-4)<=.5));
  context.updateWalls({x:4,y:3});
  const transform=new THREE.Matrix4(),scale=new THREE.Vector3();
  panels.getMatrixAt(0,transform); scale.setFromMatrixScale(transform);
  assert.equal(scale.x,0);
  context.updateWalls({x:10,y:9});
  panels.getMatrixAt(0,transform); scale.setFromMatrixScale(transform);
  assert.equal(scale.x,1);
});

test("prediction moves only the player; guards and bodies use their own positions", () => {
  const context = sceneContext();
  const state = {player: {x: 1, y: 1}, guards: [{id: "G0", x: 4, y: 5, alive: true}],
    bodies: [{id: "B0", x: 7, y: 8, angle: 1}], swat: []};
  context.updateActors(state, {x: 2, y: 3}, .5, a => ({...a, x: a.x + .1}), () => ({color: "#fff"}), .016);
  assert.equal(context.actors.get("player").position.x, 2);
  assert.equal(context.actors.get("guard:G0").position.x, 4.1);
  assert.equal(context.actors.get("body:B0").position.z, 8);
  assert.equal(context.actors.get("body:B0").userData.marker.visible, false);
  assert.equal(state.player.x, 1);
});

test("enemy vitals appear only under direct observation and never on the player", () => {
  const guard = graphics.createOperative("guard"), player = graphics.createOperative("player");
  graphics.updateOperative(guard, {x: 2, y: 3, observed: false, health: .4}, 1);
  assert.equal(guard.userData.vitals.visible, false);
  graphics.updateOperative(guard, {x: 2, y: 3, observed: true, health: .4}, 1);
  assert.equal(guard.userData.vitals.visible, true);
  assert.equal(guard.userData.bars[0].fill.scale.x, .4);
  assert.equal(guard.userData.bars[2].fill.visible, false);
  graphics.updateOperative(player, {x: 2, y: 3, observed: true, health: 1}, 1);
  assert.equal(player.userData.vitals.visible, false);
});

test("vision toggle removes GPU geometry and stale agents are disposed", () => {
  const context = sceneContext();
  const layout = map();
  const state = {map: layout, guards: [{id: "G0", x: 3, y: 3, alive: true, angle: 0}]};
  context.updateCones(state, a => a, () => ({color: "#92d8f8"}), true);
  assert.equal(context.cones.size, 1);
  const cone = context.cones.get("G0");
  let geometryDisposals = 0, materialDisposals = 0;
  cone.geometry.addEventListener("dispose", () => geometryDisposals++);
  cone.material.addEventListener("dispose", () => materialDisposals++);
  context.updateCones(state, a => a, () => ({color: "#92d8f8"}), false);
  assert.equal(context.cones.size, 0);
  assert.equal(geometryDisposals, 1); assert.equal(materialDisposals, 1);
  assert.equal(context.scene.children.length, 0);
});

test("camera vision geometry leaves a blind area directly beneath its mount", () => {
  const context=sceneContext(), layout=map(25,25);
  const state={map:layout,cameras:[{id:"C0",x:10,y:10,angle:0,online:true,vision_inner_range:1.25,vision_range:8}]};
  context.updateCones(state,a=>a,()=>({color:"#92d8f8"}),true);
  const vertices=context.cones.get("C0").geometry.attributes.position;
  for (let i=0;i<vertices.count;i++) assert.ok(Math.hypot(vertices.getX(i)-10,vertices.getZ(i)-10) >= 1.2499);
});

test("smoke, shots, blood, grenades and distractions update and clear", () => {
  const context = sceneContext();
  const state = {
    smoke: [{x: 3, y: 4, radius: 2, ttl: 3}],
    shots: [{from: {x: 2, y: 4}, to: {x: 6, y: 5}, ttl: .1}],
    blood: [{id: 0, x: 6, y: 5}],
    grenades: [{id: 1, x: 3, y: 4, target_x: 7, target_y: 8, radius: 3, fuse: 1}],
    diversions: [{id: 2, x: 2, y: 2, landed: true}],
    blasts: [{x: 4, y: 4, radius: 3, ttl: .4}], flashes: [{x: 8, y: 8, ttl: .3}],
    noise: [{id: 3, x: 2, y: 4, radius: 4, ttl: .5}],
  };
  context.updateEffects(state, 1);
  assert.equal(context.effects.size, 9);
  assert.equal(context.effects.get("warning:1").position.x, 7);
  const shot = context.effects.get("shot:0").userData.line.geometry.attributes.position;
  assert.equal(shot.getX(0), 2); assert.equal(shot.getZ(1), 5);
  context.updateEffects({}, 2);
  assert.equal(context.effects.size, 0); assert.equal(context.scene.children.length, 0);
});

test("interaction prompts prioritise actions and disappear after the mission", () => {
  const state = {status: "RUNNING", player: {x: 4, y: 4},
    map: {intel: [{x: 4, y: 4, name: "Scheda di sicurezza"}]},
    pickups: [{x: 4, y: 4, kind: "BANDAGE"}]};
  assert.match(spatial.interactionHint(state), /Scheda/);
  state.map.intel[0].stolen = true;
  assert.match(spatial.interactionHint(state, {BANDAGE: "Benda"}), /Benda/);
  state.takedown_candidate = {can_lethal: false};
  assert.match(spatial.interactionHint(state), /TIENI F/);
  state.status = "ESCAPED";
  assert.equal(spatial.interactionHint(state), "");
});

test("blood rendering respects damage radius, footprint shape and fading", () => {
  const context=sceneContext();
  context.updateEffects({blood:[{id:1,x:3,y:4,radius:.1,kind:"WOUND",age:0,lifetime:45},
    {id:2,x:4,y:4,radius:.2,kind:"WOUND",age:0,lifetime:45},
    {id:3,x:5,y:4,radius:.12,kind:"FOOTPRINT",age:6,lifetime:12}]},1);
  assert.equal(context.effects.get("blood:2").scale.x / context.effects.get("blood:1").scale.x,2);
  const footprint=context.effects.get("blood:3").children[0];
  assert.equal(footprint.scale.x,.55);
  assert.ok(Math.abs(footprint.material.opacity-.35)<1e-6);
});
