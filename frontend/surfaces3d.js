import * as THREE from "./vendor/three/three.module.js";

// Deterministic surfaces: resets keep the same wear and require no downloads.
export function surfaceTexture(kind) {
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas"); canvas.width = canvas.height = 512;
  const ctx = canvas.getContext("2d");
  let seed = 1729;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  ctx.fillStyle = kind === "floor" ? "#66737a" : kind === "fabric" ? "#b6b9b3" : "#a5aaa4";
  ctx.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 18000; i++) {
    ctx.fillStyle = `rgba(${random() > .5 ? "255,255,240" : "12,24,28"},${random() * .10})`;
    ctx.fillRect(random() * 512, random() * 512, 1 + random() * 3, 1 + random() * 3);
  }
  if (kind === "floor") {
    for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) {
      ctx.fillStyle = `rgba(4,17,22,${.02 + random() * .08})`;
      ctx.fillRect(x * 128, y * 128, 128, 128);
      ctx.strokeStyle = "#37474c"; ctx.lineWidth = 3;
      ctx.strokeRect(x * 128 + 1.5, y * 128 + 1.5, 125, 125);
      ctx.strokeStyle = "#87918f"; ctx.lineWidth = 1;
      ctx.strokeRect(x * 128 + 4, y * 128 + 4, 121, 121);
      for (let i = 0; i < 12; i++) {
        const px = x * 128 + random() * 110 + 9, py = y * 128 + random() * 110 + 9;
        ctx.strokeStyle = `rgba(18,29,34,${random() * .12})`;
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + random() * 18, py - random() * 3); ctx.stroke();
      }
    }
  } else if (kind === "wall") {
    ctx.fillStyle = "#52686b"; ctx.fillRect(0, 370, 512, 142);
    ctx.fillStyle = "#c6b686"; ctx.fillRect(0, 365, 512, 5);
    ctx.strokeStyle = "#687b7b"; ctx.lineWidth = 3; ctx.strokeRect(3, 3, 506, 506);
    for (const x of [12, 500]) for (const y of [12, 352, 495]) {
      ctx.fillStyle = "#5c6b6b"; ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    }
  } else {
    for (let i = 0; i < 512; i += 4) {
      ctx.fillStyle = "#78837a30"; ctx.fillRect(i, 0, 1, 512); ctx.fillRect(0, i, 512, 1);
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = 8;
  return texture;
}

export function roundedBox(size, radius = .035) {
  radius = Math.min(radius, ...size.map(n => n / 4));
  const geometry = new THREE.BoxGeometry(...size, 4, 4, 4);
  const positions = geometry.attributes.position;
  const point = new THREE.Vector3(), core = new THREE.Vector3(), normal = new THREE.Vector3();
  for (let i = 0; i < positions.count; i++) {
    point.fromBufferAttribute(positions, i); core.copy(point);
    for (const [axis, n] of ["x", "y", "z"].map((axis, index) => [axis, size[index]])) {
      core[axis] = THREE.MathUtils.clamp(core[axis], -n / 2 + radius, n / 2 - radius);
    }
    normal.copy(point).sub(core).normalize(); point.copy(core).addScaledVector(normal, radius);
    positions.setXYZ(i, point.x, point.y, point.z);
  }
  geometry.computeVertexNormals();
  return geometry;
}

export function contactTexture() {
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas"); canvas.width = canvas.height = 64;
  const ctx = canvas.getContext("2d"), gradient = ctx.createRadialGradient(32, 32, 4, 32, 32, 32);
  gradient.addColorStop(0, "rgba(0,6,11,.65)"); gradient.addColorStop(.55, "rgba(0,6,11,.28)"); gradient.addColorStop(1, "rgba(0,6,11,0)");
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(canvas);
}
