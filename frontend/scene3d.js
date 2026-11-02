import * as THREE from "./vendor/three/three.module.js";
import {wallCells, clippedVision, overviewHeight, interactionHint} from "./spatial.js";
import {surfaceTexture, roundedBox, contactTexture} from "./surfaces3d.js";

const PALETTE = {player: 0xdce9d7, guard: 0x7fabb2, swat: 0x627783};
const ITEM_COLORS = {MEDKIT: 0x6ee7a9, BANDAGE: 0x6ee7a9, HAZMAT: 0xffd075,
  SAMPLE_VIAL: 0x9ce98a, SECURITY_CARD: 0xffd075};
const material = (color, extra = {}) => new THREE.MeshStandardMaterial({color, roughness: .78, metalness: .18, ...extra});

function box(parent, size, position, mat) {
  const mesh = new THREE.Mesh(roundedBox(size), mat);
  mesh.position.set(...position);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function ring(parent, color, inner = .4, outer = .46) {
  const mesh = new THREE.Mesh(new THREE.RingGeometry(inner, outer, 48),
    new THREE.MeshBasicMaterial({color, transparent: true, opacity: .85, depthWrite: false, side: THREE.DoubleSide}));
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = .035;
  parent.add(mesh);
  return mesh;
}

function disposeTree(root) {
  const geometries = new Set(), materials = new Set(), textures = new Set();
  root.traverse(object => {
    if (object.isInstancedMesh) object.dispose();
    if (object.geometry) geometries.add(object.geometry);
    for (const mat of Array.isArray(object.material) ? object.material : object.material ? [object.material] : []) {
      materials.add(mat);
      for (const value of Object.values(mat)) if (value?.isTexture) textures.add(value);
    }
  });
  geometries.forEach(value => value.dispose());
  materials.forEach(value => value.dispose());
  textures.forEach(value => value.dispose());
  root.removeFromParent();
}

function textTexture(text, color = "#c1d9d6", background = false) {
  const surface = document.createElement("canvas");
  surface.width = 512; surface.height = 96;
  const ctx = surface.getContext("2d");
  if (background) { ctx.fillStyle = "rgba(5,16,20,.86)"; ctx.fillRect(0, 8, 512, 80); }
  ctx.fillStyle = color;
  ctx.font = `600 ${text.length > 24 ? 23 : 28}px Consolas, monospace`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, 256, 48, 490);
  const texture = new THREE.CanvasTexture(surface);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function label(parent, text, color, y = 2) {
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: textTexture(text, color, true),
    transparent: true, depthWrite: false, sizeAttenuation: true}));
  sprite.scale.set(3.5, .66, 1);
  sprite.position.y = y;
  parent.add(sprite);
  return sprite;
}

export function createWeaponModel(kind) {
  const group = new THREE.Group(), base = kind.replace(/_(SUPPRESSED|SUBSONIC)$/, "");
  const steel = material(0x222e32, {roughness: .4, metalness: .72}), grip = material(0x172323);
  const pistol = base === "PISTOL", shotgun = base === "SHOTGUN", compact = ["SMG", "ONE_HAND_SMG"].includes(base);
  const length = pistol ? .28 : compact ? .45 : .67;
  box(group, [pistol ? .07 : .105, .12, length], [0, 0, .10], steel);
  box(group, [.075, .20, .10], [0, -.14, -.01], grip).rotation.x = -.18;
  if (!pistol) {
    box(group, [.08, .12, .24], [0, -.015, -.34], grip);
    box(group, [.12, .25, .055], [0, -.04, -.47], grip);
    box(group, [.10, .11, .20], [0, .012, .30], material(0x526057));
    box(group, [.07, .22, .13], [0, -.14, .16], steel).rotation.x = shotgun ? 0 : -.15;
    box(group, [.04, .035, .30], [0, .09, .1], steel);
    for (let i = 0; i < 4; i++) box(group, [.115, .015, .025], [0, .085, .22 + i * .045], steel);
  }
  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(shotgun ? .038 : .025, .025, pistol ? .08 : .3, 12), steel);
  barrel.rotation.x = Math.PI / 2; barrel.position.set(0, .015, length / 2 + .22); group.add(barrel);
  const suppressor = new THREE.Mesh(new THREE.CylinderGeometry(.05, .05, .27, 16), steel);
  suppressor.rotation.x = Math.PI / 2; suppressor.position.set(0, .015, length / 2 + (pistol ? .35 : .5));
  suppressor.visible = /_(SUPPRESSED|SUBSONIC)$/.test(kind); group.add(suppressor);
  box(group, [.03, .04, .035], [0, .09, length / 2 + .07], steel);
  group.userData = {suppressor, base, length};
  return group;
}

// Exported separately so geometry, poses and collision alignment can be tested
// without constructing a GPU renderer or connecting to a live game session.
export function createOperative(role = "player") {
  const root = new THREE.Group(), rig = new THREE.Group();
  root.add(rig);
  const cloth = surfaceTexture("fabric");
  const uniform = material(role === "swat" ? 0x344550 : role === "guard" ? 0x6d827e : 0x879b83,
    {map: cloth, bumpMap: cloth, bumpScale: .018, roughness: .98, metalness: 0});
  const vest = material(role === "player" ? 0x273635 : 0x23343c);
  const boots = material(0x121d23), skin = material(0xb59a80), accent = material(PALETTE[role]);
  box(rig, [.48, .55, .3], [0, 1.02, 0], uniform);
  box(rig, [.51, .42, .12], [0, 1.04, .19], vest);
  box(rig, [.36, .38, .18], [0, 1.05, -.24], vest);
  box(rig, [.45, .13, .33], [0, .75, 0], boots);
  for (const x of [-.16, 0, .16]) {
    box(rig, [.11, .16, .09], [x, .97, .28], vest);
    box(rig, [.09, .018, .014], [x, 1.03, .335], accent);
  }
  for (const side of [-1, 1]) {
    box(rig, [.065, .38, .045], [side * .185, 1.15, .198], boots);
    box(rig, [.09, .22, .13], [side * .28, .79, -.04], boots);
  }
  const head = new THREE.Mesh(new THREE.SphereGeometry(.17, 18, 12), skin);
  head.position.set(0, 1.48, .035); head.castShadow = true; rig.add(head);
  const helmet = new THREE.Mesh(new THREE.SphereGeometry(.195, 20, 12, 0, Math.PI * 2, 0, Math.PI * .62), uniform);
  helmet.position.set(0, 1.52, 0); helmet.castShadow = true; rig.add(helmet);
  box(rig, [.31, .09, .06], [0, 1.51, .165], boots);
  box(rig, [.24, .055, .025], [0, 1.51, .205], material(0x6bafa2, {metalness: .65, roughness: .18}));
  for (const side of [-1, 1]) box(rig, [.075, .13, .12], [side * .18, 1.47, .015], boots);
  box(rig, [.09, .08, .06], [.07, 1.65, .15], boots);
  const legs = [], arms = [];
  for (const side of [-1, 1]) {
    const leg = new THREE.Group(); leg.position.set(side * .145, .7, 0); rig.add(leg);
    box(leg, [.2, .49, .23], [0, -.22, 0], uniform);
    box(leg, [.155, .16, .055], [0, -.29, .135], vest);
    box(leg, [.22, .19, .36], [0, -.59, .06], boots);
    legs.push(leg);
    const arm = new THREE.Group(); arm.position.set(side * .32, 1.22, 0); rig.add(arm);
    box(arm, [.17, .43, .19], [0, -.19, .05], uniform);
    box(arm, [.2, .13, .23], [0, -.04, .025], vest);
    box(arm, [.14, .14, .16], [0, -.45, .08], boots);
    arms.push(arm);
  }
  const weapon = new THREE.Group(); weapon.position.set(.2, .92, .4); rig.add(weapon);
  const model = createWeaponModel("PISTOL"); weapon.add(model);
  const suppressor = model.userData.suppressor;
  const knife = box(rig, [.045, .035, .35], [.33, .8, .3], material(0xd0e0db, {metalness: .8}));
  const shield = box(rig, [.62, .9, .1], [-.2, .95, .57], vest);
  shield.visible = false;
  const medicBadge = new THREE.Group();
  const medicColor = material(0x8fffc0);
  box(medicBadge, [.17, .055, .025], [0, 1.19, .272], medicColor);
  box(medicBadge, [.055, .17, .025], [0, 1.19, .272], medicColor);
  medicBadge.visible = false; rig.add(medicBadge);
  const marker = ring(root, PALETTE[role], .42, .47);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1.25, 1.25), new THREE.MeshBasicMaterial({
    map: contactTexture(), color: 0x030a0d, transparent: true, opacity: .55, depthWrite: false}));
  shadow.rotation.x = -Math.PI / 2; shadow.position.y = .008; root.add(shadow);
  if (role === "player") { marker.material.depthTest = false; marker.renderOrder = 8; }
  const vitals = new THREE.Group(); vitals.position.y = 2.1; vitals.visible = false; root.add(vitals);
  const bars = [];
  if (role !== "player") for (const [i, color] of [0x7addaa, 0x8acfe9, 0xc7b8ed].entries()) {
    const track = new THREE.Mesh(new THREE.PlaneGeometry(.7, .045), new THREE.MeshBasicMaterial({color: 0x15232b}));
    const fill = new THREE.Mesh(new THREE.PlaneGeometry(.7, .045), new THREE.MeshBasicMaterial({color}));
    track.position.y = fill.position.y = -i * .085; fill.position.z = .002;
    vitals.add(track, fill); bars.push({track, fill});
  }
  // A waist pivot inclines the torso, head and weapon together. The feet and
  // authoritative root stay at the server's physical peek position.
  const leanPivot = new THREE.Group(), upperBody = new THREE.Group();
  leanPivot.position.y = .75; upperBody.position.y = -.75;
  for (const child of [...rig.children]) if (!legs.includes(child)) upperBody.add(child);
  leanPivot.add(upperBody); rig.add(leanPivot);
  root.userData = {rig, leanPivot, upperBody, legs, arms, weapon, model, weaponKind: "PISTOL", suppressor, knife, shield, medicBadge, marker, vitals, bars, role, phase: 0, lastX: null, lastY: null};
  return root;
}

export function updateOperative(root, data, dt) {
  const user = root.userData;
  const distance = user.lastX === null ? 0 : Math.hypot(data.x - user.lastX, data.y - user.lastY);
  const moving = distance > .0005;
  user.phase += Math.min(.2, distance) * (data.crouched ? 11 : 8);
  user.lastX = data.x; user.lastY = data.y;
  root.position.set(data.x, 0, data.y);
  root.rotation.y = Math.PI / 2 - (data.angle || 0);
  const flat = data.prone || data.diving || data.down;
  const crouch = data.crouched && !flat;
  const gain = 1 - Math.exp(-18 * dt);
  user.rig.rotation.x += ((flat ? Math.PI / 2 : 0) - user.rig.rotation.x) * gain;
  user.rig.position.y += ((flat ? .25 : moving ? Math.abs(Math.sin(user.phase)) * .035 : 0) - user.rig.position.y) * gain;
  user.rig.position.z = flat ? -.65 : 0;
  user.rig.scale.y += ((crouch ? .63 : 1) - user.rig.scale.y) * gain;
  user.legs.forEach((leg, i) => { leg.rotation.x = flat ? 0 : moving ? Math.sin(user.phase + i * Math.PI) * .55 : 0; });
  const armed = Boolean(data.weapon && !data.disarmed);
  if (armed && data.weapon !== user.weaponKind) {
    disposeTree(user.model);
    user.model = createWeaponModel(data.weapon); user.weapon.add(user.model);
    user.weaponKind = data.weapon; user.suppressor = user.model.userData.suppressor;
  }
  user.weapon.visible = armed && !data.down;
  user.suppressor.visible = /_(SUPPRESSED|SUBSONIC)$/.test(data.weapon || "");
  user.knife.visible = !armed && user.role === "player" && Boolean(data.knife);
  user.shield.visible = Boolean(data.shield || data.role === "SHIELD") && !data.down;
  user.medicBadge.visible = Boolean(data.is_medic);
  const drawing = data.draw_remaining > 0;
  const reloading = Boolean(data.firearm?.reload_mode);
  const braced = user.shield.visible && !flat && (data.suppression_timer || 0) > 0;
  const lean = !flat && !drawing && !reloading && !braced
    ? THREE.MathUtils.clamp(Number(data.lean_amount) || 0, -1, 1) : 0;
  user.leanPivot.rotation.z += (-lean * .22 - user.leanPivot.rotation.z) * gain;
  user.leanPivot.rotation.x += ((braced ? .12 : 0) - user.leanPivot.rotation.x) * gain;
  user.shield.rotation.x += ((braced ? .14 : 0) - user.shield.rotation.x) * gain;
  user.arms.forEach((arm, i) => { arm.rotation.x = armed ? -1 : moving ? Math.sin(user.phase + (1 - i) * Math.PI) * .35 : -.12; });
  if (drawing) {
    const remaining = Math.min(1, data.draw_remaining / Math.max(.01, data.draw_duration));
    user.weapon.rotation.x = remaining * 1.3;
    user.weapon.position.y = .92 - remaining * .35;
    user.arms.forEach(arm => { arm.rotation.x = -1 + remaining * 1.3; });
  } else if (reloading) {
    user.arms[0].rotation.x = -.5 + Math.sin(data.firearm.reload_remaining * 11) * .4;
    user.arms[1].rotation.x = -.4;
    user.weapon.rotation.x = -.35;
  } else user.weapon.rotation.x = 0;
  if (!drawing) user.weapon.position.y = .92;
  if (user.shield.visible && !flat && !drawing) {
    // The support arm keeps the shield presented while the pistol hand reloads.
    user.arms[0].rotation.x = braced ? -1.2 : -.95;
    if (reloading) user.arms[1].rotation.x = -.45 + Math.sin(data.firearm.reload_remaining * 11) * .15;
  }
  if (braced && !drawing && !reloading) {
    user.arms[1].rotation.x = -.35;
    user.weapon.rotation.x = .5;
    user.weapon.position.y = .8;
  }
  if (data.knife_remaining > 0) user.arms[1].rotation.x = -1.6 + (data.knife_remaining / .3) * 2;
  user.marker.visible = !data.down;
  user.vitals.visible = user.role !== "player" && Boolean(data.observed) && !data.down;
  for (const [i, key] of ["health", "stamina", "armor"].entries()) {
    const bar = user.bars[i];
    if (!bar) continue;
    const value = THREE.MathUtils.clamp(data[key] ?? 1, 0, 1);
    bar.fill.scale.x = value; bar.fill.position.x = -(1 - value) * .35;
    bar.track.visible = bar.fill.visible = key !== "armor" || data.armor != null;
  }
}

export function buildFacility(map, {labels = true} = {}) {
  const root = new THREE.Group(); root.name = "facility";
  const w = map.width, h = map.height;
  let texture = null;
  if (labels) { texture = surfaceTexture("floor"); texture?.repeat.set(w / 4, h / 4); }
  const floor = new THREE.Mesh(new THREE.BoxGeometry(w, .3, h), material(0x8e9da0,
    {map: texture, bumpMap: texture, bumpScale: .026, roughness: .75, metalness: .12}));
  floor.position.set((w - 1) / 2, -.2, (h - 1) / 2); floor.receiveShadow = true; root.add(floor);
  const cells = wallCells(map);
  const wallMap = labels ? surfaceTexture("wall") : null;
  const walls = new THREE.InstancedMesh(new THREE.BoxGeometry(.99, 1, .99), material(0xc1c7c1,
    {map: wallMap, bumpMap: wallMap, bumpScale: .018, roughness: .9, metalness: .04}), cells.length);
  const caps = new THREE.InstancedMesh(roundedBox([1.01, .07, 1.01], .015), material(0x4e6268, {metalness: .45, roughness: .5}), cells.length);
  const base = new THREE.InstancedMesh(new THREE.BoxGeometry(1.015, .18, 1.015), material(0x22343b), cells.length);
  walls.castShadow = walls.receiveShadow = true; caps.receiveShadow = true;
  const matrix = new THREE.Matrix4();
  cells.forEach((cell, i) => {
    matrix.makeScale(1, 2, 1); matrix.setPosition(cell.x, 1, cell.y); walls.setMatrixAt(i, matrix);
    matrix.makeTranslation(cell.x, 2.035, cell.y); caps.setMatrixAt(i, matrix);
    matrix.makeTranslation(cell.x, .09, cell.y); base.setMatrixAt(i, matrix);
    walls.setColorAt(i, new THREE.Color().setHSL(.49, .035, .66 + ((cell.x * 13 + cell.y * 7) % 5) * .025));
  });
  // Heights change for cutaway rendering; fixed full-height bounds prevent
  // frustum culling from retaining an outdated, shortened bounding sphere.
  walls.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  caps.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  walls.computeBoundingSphere(); caps.computeBoundingSphere();
  root.add(walls, caps, base);
  const shadows = new THREE.InstancedMesh(new THREE.PlaneGeometry(2.4, 2.4), new THREE.MeshBasicMaterial({
    map: labels ? contactTexture() : null, color: 0x031019, transparent: true, opacity: .35, depthWrite: false}), cells.length);
  cells.forEach((cell, i) => { matrix.makeRotationX(-Math.PI / 2); matrix.setPosition(cell.x, -.035, cell.y); shadows.setMatrixAt(i, matrix); });
  root.add(shadows);
  // Mounted details stay on existing solid cells. Walkable space and sight lines
  // continue to use exactly the server geometry, including the cutaway view.
  const fixtures = [];
  cells.forEach((cell, index) => {
    if ((cell.x * 7 + cell.y * 11) % 4) return;
    for (const [dx, dy] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
      if (map.grid[cell.y + dy]?.[cell.x + dx] !== 0) continue;
      fixtures.push({index, x: cell.x + dx * .495, y: cell.y + dy * .495, angle: Math.atan2(dx, dy)});
    }
  });
  const panels = new THREE.InstancedMesh(roundedBox([.64, .28, .035], .012), material(0x253b43, {metalness: .6}), fixtures.length);
  const strips = new THREE.InstancedMesh(new THREE.BoxGeometry(.49, .035, .045), material(0xb3dfd8,
    {emissive: 0x7ebeb2, emissiveIntensity: 1.5, metalness: .1}), fixtures.length);
  const vents = new THREE.InstancedMesh(new THREE.BoxGeometry(.50, .015, .045), material(0x819596, {metalness: .7}), fixtures.length * 4);
  fixtures.forEach((fixture, i) => {
    matrix.makeRotationY(fixture.angle); matrix.setPosition(fixture.x, 1.42, fixture.y); panels.setMatrixAt(i, matrix);
    matrix.setPosition(fixture.x, 1.67, fixture.y); strips.setMatrixAt(i, matrix);
    for (let n = 0; n < 4; n++) { matrix.setPosition(fixture.x, 1.34 + n * .048, fixture.y); vents.setMatrixAt(i * 4 + n, matrix); }
  });
  for (const mesh of [panels, strips, vents]) { mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); mesh.computeBoundingSphere(); }
  root.add(panels, strips, vents);
  const edgeMat = material(0x284550, {emissive: 0x1c4351, emissiveIntensity: .3});
  box(root, [w, .12, .1], [(w - 1) / 2, .02, -.5], edgeMat);
  box(root, [w, .12, .1], [(w - 1) / 2, .02, h - .5], edgeMat);
  box(root, [.1, .12, h], [-.5, .02, (h - 1) / 2], edgeMat);
  box(root, [.1, .12, h], [w - .5, .02, (h - 1) / 2], edgeMat);
  if (labels) for (const zone of map.zones || []) {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(5, .94),
      new THREE.MeshBasicMaterial({map: textTexture(zone.name, "#8ca9ad"), transparent: true, opacity: .55, depthWrite: false}));
    mesh.rotation.x = -Math.PI / 2; mesh.position.set(zone.x + 1.6, -.035, zone.y); root.add(mesh);
  }
  for (const hazard of map.hazards || []) {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(hazard.width, hazard.height),
      new THREE.MeshBasicMaterial({color: 0x94af38, transparent: true, opacity: .3, depthWrite: false}));
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(hazard.x + hazard.width / 2, .005, hazard.y + hazard.height / 2); root.add(mesh);
  }
  root.userData = {walls, caps, cells, fixtures, panels, strips, vents};
  return root;
}

export class ShadowScene {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false, powerPreference: "high-performance"});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.10;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x071119);
    this.scene.fog = new THREE.Fog(0x071119, 38, 115);
    this.camera = new THREE.PerspectiveCamera(45, 1, .1, 250);
    this.target = new THREE.Vector3(); this.desired = new THREE.Vector3();
    this.zoom = 12; this.height = 12; this.overview = false; this.initialized = false;
    this.scene.add(new THREE.HemisphereLight(0xc1d9e7, 0x273339, 1.8));
    this.sun = new THREE.DirectionalLight(0xffe3bf, 2.7);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    Object.assign(this.sun.shadow.camera, {left: -22, right: 22, top: 22, bottom: -22, near: .5, far: 80});
    this.sun.shadow.bias = -.0005; this.sun.shadow.normalBias = .04;
    this.scene.add(this.sun, this.sun.target);
    const rim = new THREE.DirectionalLight(0x89bfcf, 1.05); rim.position.set(20, 12, 8); this.scene.add(rim);
    this.fill = new THREE.PointLight(0x6fbacb, 16, 17, 2); this.scene.add(this.fill);
    this.actors = new Map(); this.cones = new Map(); this.items = new Map(); this.effects = new Map();
    this.frameTime = null; this.coneTime = -Infinity; this.cutawayKey = "";
    this.matrix = new THREE.Matrix4();
    this.resize = this.resize.bind(this);
    this.observer = new ResizeObserver(this.resize); this.observer.observe(canvas);
    this.resize();
    canvas.addEventListener("wheel", event => {
      event.preventDefault(); this.zoom = THREE.MathUtils.clamp(this.zoom + Math.sign(event.deltaY) * 1.3, 8, 27);
    }, {passive: false});
    canvas.addEventListener("webglcontextlost", event => {
      event.preventDefault(); window.shadowSceneError = "Contesto 3D interrotto. Ricarica la pagina per ripristinarlo.";
      document.body.dataset.renderer = "radar";
    });
    canvas.addEventListener("webglcontextrestored", () => {
      window.shadowSceneError = ""; document.body.dataset.renderer = "3d";
    });
    this.cameraButton = document.getElementById("cameraBtn");
    this.cameraButton?.addEventListener("click", () => { this.toggleOverview(); canvas.focus({preventScroll: true}); });
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(1, rect.width), height = Math.max(1, rect.height);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
  }

  toggleOverview() {
    this.overview = !this.overview;
    if (this.cameraButton) {
      this.cameraButton.textContent = this.overview ? "T · SEGUI OPERATORE" : "T · PANORAMICA";
      this.cameraButton.setAttribute("aria-pressed", String(this.overview));
    }
  }

  resetMap(state) {
    if (this.facility) disposeTree(this.facility);
    for (const collection of [this.actors, this.cones, this.items, this.effects]) {
      collection.forEach(disposeTree); collection.clear();
    }
    this.facility = buildFacility(state.map); this.scene.add(this.facility);
    this.mapGrid = state.map.grid;
    this.mapKey = `${state.mode}:${state.level}:${state.revision}`;
    this.initialized = false; this.cutawayKey = ""; this.coneTime = -Infinity;
  }

  sync(collection, entries, create, update) {
    const active = new Set();
    for (const entry of entries) {
      const key = entry.key;
      active.add(key);
      let object = collection.get(key);
      if (!object) { object = create(entry); collection.set(key, object); this.scene.add(object); }
      update(object, entry);
    }
    for (const [key, object] of collection) if (!active.has(key)) { disposeTree(object); collection.delete(key); }
  }

  updateWalls(player) {
    const key = `${Math.round(player.x * 3)},${Math.round(player.y * 3)},${this.overview}`;
    if (key === this.cutawayKey) return;
    this.cutawayKey = key;
    const {walls, caps, cells, fixtures, panels, strips, vents} = this.facility.userData;
    const heights = [];
    cells.forEach((cell, i) => {
      const distance = Math.hypot(cell.x - player.x, cell.y - player.y);
      // Lower nearby walls on the camera side of the operator so doorways
      // and the character stay readable in narrow rooms.
      const height = !this.overview && distance < 4 && cell.y > player.y - .7 ? .42 : this.overview ? 1.1 : 2;
      heights.push(height);
      this.matrix.makeScale(1, height, 1); this.matrix.setPosition(cell.x, height / 2, cell.y); walls.setMatrixAt(i, this.matrix);
      this.matrix.makeTranslation(cell.x, height + .035, cell.y); caps.setMatrixAt(i, this.matrix);
    });
    walls.instanceMatrix.needsUpdate = true; caps.instanceMatrix.needsUpdate = true;
    fixtures.forEach((fixture, i) => {
      const visible = heights[fixture.index] > 1.7;
      this.matrix.makeRotationY(fixture.angle); this.matrix.scale(new THREE.Vector3().setScalar(visible ? 1 : 0));
      this.matrix.setPosition(fixture.x, 1.42, fixture.y); panels.setMatrixAt(i, this.matrix);
      this.matrix.setPosition(fixture.x, 1.67, fixture.y); strips.setMatrixAt(i, this.matrix);
      for (let n = 0; n < 4; n++) { this.matrix.setPosition(fixture.x, 1.34 + n * .048, fixture.y); vents.setMatrixAt(i * 4 + n, this.matrix); }
    });
    for (const mesh of [panels, strips, vents]) mesh.instanceMatrix.needsUpdate = true;
  }

  updateActors(state, player, facing, displayed, colorFor, dt) {
    const entries = [
      {...state.player, ...player, angle: facing, key: "player", roleType: "player"},
      ...(state.guards || (state.guard ? [state.guard] : [])).filter(a => a.alive && !a.unconscious).map(a => ({...displayed(a), key: `guard:${a.id}`, roleType: "guard"})),
      ...(state.swat || []).filter(a => a.alive).map(a => ({...displayed(a), key: `swat:${a.id}`, roleType: "swat"})),
      ...(state.bodies || []).map(a => ({...a, key: `body:${a.id}`, roleType: "guard", down: true})),
      ...(state.swat || []).filter(a => !a.alive).map(a => ({...a, key: `fallen:${a.id}`, roleType: "swat", down: true})),
    ];
    this.sync(this.actors, entries, data => createOperative(data.roleType), (object, data) => {
      updateOperative(object, data, this.initialized ? dt : 1);
      if (this.camera) object.userData.vitals.quaternion.copy(object.quaternion).invert().multiply(this.camera.quaternion);
      object.userData.marker.material.color.set(data.key === "player" ? state.takedown ? 0x62f2ad : state.takedown_candidate ? 0xffd075 : PALETTE.player : colorFor(data).color);
    });
  }

  updateCones(state, displayed, colorFor, visible) {
    const entries = visible ? [
      ...(state.guards || (state.guard ? [state.guard] : [])).filter(a => a.alive && !a.unconscious),
      ...(state.swat || []).filter(a => a.alive),
      ...(state.cameras || []).filter(a => a.online),
    ].map(a => ({...displayed(a), key: a.id})) : [];
    this.sync(this.cones, entries, () => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(56 * 18), 3));
      return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({color: 0x92d8f8, transparent: true,
        opacity: .14, depthWrite: false, side: THREE.DoubleSide}));
    }, (object, data) => {
      const points = clippedVision(data, state.map.grid, state.map.doors);
      const position = object.geometry.attributes.position;
      const inner = point => {
        const dx=point.x-data.x, dy=point.y-data.y, length=Math.hypot(dx,dy);
        const ratio=length > 0 ? Math.min(1,(data.vision_inner_range || 0)/length) : 0;
        return {x:data.x+dx*ratio,y:data.y+dy*ratio};
      };
      for (let i = 0; i < points.length - 1; i++) {
        const a=inner(points[i]), b=inner(points[i+1]);
        for (const [j,p] of [a,points[i],points[i+1],a,points[i+1],b].entries()) position.setXYZ(i*6+j,p.x,.012,p.y);
      }
      position.needsUpdate = true; object.geometry.computeBoundingSphere();
      object.material.color.set(colorFor(data).color);
      object.material.opacity = data.combat_alert || (data.confidence || data.visual_confidence || 0) > .5 ? .23 : .13;
    });
  }

  createItem(data) {
    const group = new THREE.Group();
    if (data.type === "door") {
      group.userData.panel = box(group, [.88, 1.7, .88], [0, .85, 0], material(0x936d36, {metalness: .55}));
      box(group, [.93, .09, .93], [0, 1.75, 0], material(0xffce7c, {emissive: 0xffb84f, emissiveIntensity: .6}));
    } else if (data.type === "camera") {
      box(group, [.1, 1.7, .1], [0, .85, 0], material(0x475d65));
      const head = new THREE.Group(); head.position.y = 1.7; group.add(head); group.userData.head = head;
      box(head, [.24, .2, .42], [0, 0, .1], material(0xb9c6bf));
      group.userData.lens = box(head, [.16, .12, .03], [0, 0, .325], material(0xff7670, {emissive: 0xff403b}));
    } else if (data.type === "exit") {
      ring(group, 0x72e4c0, .7, .8);
      box(group, [.12, 2.5, .14], [-.85, 1.25, 0], material(0x345b59));
      box(group, [.12, 2.5, .14], [.85, 1.25, 0], material(0x345b59));
      box(group, [1.82, .13, .14], [0, 2.5, 0], material(0x80f1c9, {emissive: 0x3dbb94, emissiveIntensity: 1}));
      label(group, "ESFILTRAZIONE", "#a6f5d6", 3);
    } else {
      const intel = data.type === "intel", med = data.kind === "MEDKIT" || data.kind === "BANDAGE";
      const color = intel ? 0xffd075 : ITEM_COLORS[data.kind] || 0x8ed6e7;
      if (/^(PISTOL|RIFLE|SMG|SHOTGUN|ONE_HAND_SMG)(_|$)/.test(data.kind || "")) {
        const model = createWeaponModel(data.kind); model.rotation.z = Math.PI / 2;
        model.rotation.y = .4; model.position.y = .14; group.add(model);
      } else if (["SMOKE", "FLASH", "GRENADE", "SAMPLE_VIAL"].includes(data.kind)) {
        const canister = new THREE.Mesh(new THREE.CylinderGeometry(.11, .11, .32, 18), material(0x637266, {metalness: .4}));
        canister.position.y = .19; group.add(canister);
        box(group, [.19, .065, .19], [0, .37, 0], material(0x29383d));
        box(group, [.04, .10, .035], [.1, .36, 0], material(0xbac2b7, {metalness: .8}));
        const band = new THREE.Mesh(new THREE.CylinderGeometry(.113, .113, .06, 18), material(color)); band.position.y = .23; group.add(band);
      } else {
        box(group, intel ? [.65, .7, .55] : data.fixed ? [.7, .7, .5] : [.45, .22, .32],
          [0, intel || data.fixed ? .35 : .16, 0], material(intel ? 0x485452 : med ? 0x6a8074 : 0x596454));
        const surface = box(group, [.3, .035, .22], [0, intel || data.fixed ? .72 : .29, 0],
          material(color, {emissive: color, emissiveIntensity: .25}));
        if (intel) {
          surface.rotation.x = .2; label(group, data.name || "INTEL", "#ffda8c", 1.6);
          for (let i = 0; i < 3; i++) box(group, [.4, .025, .02], [0, .27 + i * .12, .285], material(0xa3b2a6));
        }
        if (med) {
          box(group, [.22, .025, .08], [0, data.fixed ? .76 : .33, 0], material(0xe6f2dc));
          box(group, [.08, .025, .22], [0, data.fixed ? .76 : .33, 0], material(0xe6f2dc));
        }
        if (data.kind === "AMMO") for (const side of [-1, 1]) box(group, [.045, .24, .34], [side * .14, .17, 0], material(0x263b35));
      }
      group.userData.marker = ring(group, color, intel ? .65 : .32, intel ? .69 : .35);
    }
    return group;
  }

  updateItems(state, time) {
    const entries = [
      ...(state.map.intel || []).filter(item => !item.stolen).map(item => ({...item, type: "intel", key: `intel:${item.id}`})),
      ...(state.pickups || []).map(item => ({...item, type: "pickup", key: `pickup:${item.id}`})),
      ...(state.map.doors || []).map(item => ({...item, type: "door", key: `door:${item.x}:${item.y}`})),
      ...(state.cameras || []).map(item => ({...item, type: "camera", key: `camera:${item.id}`})),
      ...(state.map.exit ? [{...state.map.exit, key: "exit", type: "exit"}] : []),
    ];
    this.sync(this.items, entries, data => this.createItem(data), (object, data) => {
      object.position.set(data.x, 0, data.y);
      if (object.userData.marker) object.userData.marker.material.opacity = .45 + Math.sin(time * 3) * .2;
      if (data.type === "door") object.scale.y = data.locked ? 1 : .06;
      if (data.type === "camera") {
        object.userData.head.rotation.y = Math.PI / 2 - (data.angle || 0);
        object.userData.lens.material.emissiveIntensity = data.online ? 1 : 0;
      }
    });
  }

  createEffect(data) {
    const group = new THREE.Group();
    if (data.type === "shot") {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
      const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({color: 0xffd28a, transparent: true}));
      group.add(line); group.userData.line = line;
    } else if (data.type === "smoke") {
      const mat = new THREE.MeshStandardMaterial({color: 0x9aaeb3, roughness: 1, transparent: true, opacity: .17, depthWrite: false});
      for (let i = 0; i < 7; i++) {
        const puff = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 1), mat);
        puff.position.set(Math.cos(i * 2.4) * .4, .2 + (i % 3) * .35, Math.sin(i * 2.4) * .4);
        group.add(puff);
      }
    } else if (["blast", "flash"].includes(data.type)) {
      const sphere = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 2),
        new THREE.MeshBasicMaterial({color: data.type === "flash" ? 0xfff4cc : 0xffa35b, transparent: true, opacity: .5, depthWrite: false, wireframe: data.type === "blast"}));
      group.add(sphere);
    } else if (data.type === "casing") {
      const casing = new THREE.Mesh(new THREE.CylinderGeometry(.035, .035, .15, 6), material(0xc19b52, {metalness: .7}));
      casing.rotation.z = Math.PI / 2; casing.position.y = .04; group.add(casing);
    } else if (data.type === "blood") {
      const geometry = new THREE.CircleGeometry(.24, 32);
      const vertices = geometry.attributes.position;
      for (let i = 1; i < vertices.count; i++) {
        const angle = Math.atan2(vertices.getY(i), vertices.getX(i));
        const radius = .82 + .1 * Math.sin(angle * 5) + .07 * Math.cos(angle * 11);
        vertices.setXY(i, vertices.getX(i) * radius, vertices.getY(i) * radius);
      }
      const stain = new THREE.Mesh(geometry,
        new THREE.MeshStandardMaterial({color: 0x631d21, roughness: .28, metalness: .05, transparent: true, opacity: .7, depthWrite: false}));
      stain.rotation.x = -Math.PI / 2; stain.scale.x = 1.4; stain.position.y = .006; group.add(stain);
    } else if (data.type === "noise" || data.type === "warning") {
      group.userData.marker = ring(group, data.type === "warning" ? 0xff705a : 0xe7bf7f, .97, 1);
    } else {
      const mesh = new THREE.Mesh(new THREE.IcosahedronGeometry(data.type === "grenade" ? .13 : .1, 1),
        material(data.type === "grenade" ? 0xff725c : 0xffd075, {emissive: 0x7d4324}));
      mesh.position.y = .22; mesh.castShadow = true; group.add(mesh);
    }
    return group;
  }

  updateEffects(state, time) {
    const entries = [];
    for (const [field, type] of [["shots", "shot"], ["smoke", "smoke"], ["blood", "blood"], ["casings", "casing"],
      ["blasts", "blast"], ["flashes", "flash"], ["noise", "noise"], ["grenades", "grenade"], ["diversions", "diversion"]]) {
      (state[field] || []).forEach((item, i) => entries.push({...item, type, key: `${type}:${item.id ?? i}`}));
    }
    (state.grenades || []).forEach((item, i) => entries.push({...item, x: item.target_x, y: item.target_y, type: "warning", key: `warning:${item.id ?? i}`}));
    this.sync(this.effects, entries, data => this.createEffect(data), (object, data) => {
      if (data.type === "shot") {
        const position = object.userData.line.geometry.attributes.position;
        position.setXYZ(0, data.from.x, .95, data.from.y); position.setXYZ(1, data.to.x, .9, data.to.y);
        position.needsUpdate = true; object.userData.line.geometry.computeBoundingSphere();
        object.userData.line.material.opacity = Math.min(1, data.ttl / .12);
        return;
      }
      object.position.set(data.x, 0, data.y);
      if (data.type === "casing") object.position.y = data.landed ? 0 : .55;
      if (data.type === "blood") {
        const radius = data.radius ?? .24;
        object.scale.setScalar(radius/.24);
        object.children[0].scale.x = data.kind === "FOOTPRINT" ? .55 : 1.4;
        object.rotation.y = data.kind === "FOOTPRINT" ? Math.PI/2-(data.angle || 0) : (Number(data.id) || 0)*.71;
        object.children[0].material.opacity = .7*Math.max(0,1-(data.age || 0)/(data.lifetime || 45));
      }
      if (data.type === "smoke") {
        object.scale.setScalar(data.radius * .72);
        object.rotation.y = time * .12;
        object.children[0].material.opacity = Math.min(.19, data.ttl * .12);
      } else if (data.type === "noise" || data.type === "warning") {
        const life = Math.max(0, Math.min(1, (data.ttl || 0) / .72));
        object.scale.setScalar((data.radius || 3) * (data.type === "warning" ? 1 : 1 - life * .55));
        object.userData.marker.material.opacity = data.type === "warning" ? .45 + Math.sin(time * 15) * .2 : life * .45;
      } else if (data.type === "blast" || data.type === "flash") {
        object.scale.setScalar(data.type === "blast" ? (data.radius || 3) * Math.max(.15, 1 - data.ttl) : 2);
        object.children[0].material.opacity = Math.min(.65, data.ttl);
      }
    });
  }

  render(state, {player, facing, displayed, colorFor, vision, now, itemNames}) {
    const time = now / 1000;
    const dt = Math.min(.05, Math.max(.001, this.frameTime === null ? .016 : time - this.frameTime));
    this.frameTime = time;
    if (!state?.map?.grid || !state.player) { this.renderer.render(this.scene, this.camera); return; }
    if (this.mapGrid !== state.map.grid || this.mapKey !== `${state.mode}:${state.level}:${state.revision}`) this.resetMap(state);
    player = player || state.player;
    const desiredTarget = this.overview ? this.desired.set((state.map.width - 1) / 2, 0, (state.map.height - 1) / 2) : this.desired.set(player.x, 0, player.y);
    const height = this.overview ? overviewHeight(state.map.width, state.map.height, this.camera.aspect) : this.zoom;
    if (!this.initialized) { this.target.copy(desiredTarget); this.height = height; }
    else { this.target.lerp(desiredTarget, 1 - Math.exp(-7 * dt)); this.height += (height - this.height) * (1 - Math.exp(-8 * dt)); }
    this.camera.position.set(this.target.x, this.height, this.target.z + this.height * .72);
    this.camera.lookAt(this.target);
    this.scene.fog.near = this.overview ? 150 : 38; this.scene.fog.far = this.overview ? 230 : 115;
    this.sun.position.set(player.x - 9, 25, player.y - 7); this.sun.target.position.set(player.x, 0, player.y);
    this.fill.position.set(player.x, 4, player.y + 2);
    this.updateWalls(player);
    this.updateActors(state, player, facing, displayed, colorFor, dt);
    if (time - this.coneTime > .08 || vision !== this.lastVision) {
      this.updateCones(state, displayed, colorFor, vision); this.coneTime = time; this.lastVision = vision;
    }
    this.updateItems(state, time); this.updateEffects(state, time);
    this.initialized = true;
    this.renderer.render(this.scene, this.camera);
    const hint = document.getElementById("interactionHint");
    if (hint) { const text = interactionHint(state, itemNames); hint.textContent = text; hint.hidden = !text; }
    const posture = document.getElementById("stanceLabel");
    if (posture) posture.textContent = state.player.prone ? "A TERRA" : state.player.crouched ? "ACCOVACCIATO" : state.player.speed > 2 ? "IN CORSA" : "OPERATIVO";
    const flash = document.getElementById("flashOverlay");
    if (flash) flash.style.opacity = state.player.flashed ? ".88" : "0";
  }
}

// The main input/network script remains independent of the graphics module.
// A clear radar fallback also keeps the game playable if WebGL2 is unavailable.
if (typeof window !== "undefined" && document.getElementById("radar")) {
  try {
    window.shadowScene = new ShadowScene(document.getElementById("game"));
    document.body.dataset.renderer = "3d";
  } catch (error) {
    console.error("Shadow Protocol 3D:", error);
    window.shadowSceneError = "3D non disponibile: serve WebGL 2. Radar tattico attivo.";
    document.body.dataset.renderer = "radar";
  }
}
