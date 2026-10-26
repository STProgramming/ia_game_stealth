const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

function radar() {
  const elements = new Map();
  const context = new Proxy({}, {get: (_, key) => key === "createRadialGradient" || key === "createLinearGradient" ? () => ({addColorStop() {}}) : () => {}});
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      classList: {add() {}, remove() {}, toggle() {}}, dataset: {}, style: {},
      addEventListener() {}, setAttribute() {}, querySelector() {return element(`${id}-span`);},
      focus() {}, getContext() {return context;},
      getBoundingClientRect() {return {width:1200, height:600, left:0, top:0};},
    });
    return elements.get(id);
  }
  const sandbox = vm.createContext({
    document: {getElementById: element, body: {dataset: {}}},
    window: {addEventListener() {}, devicePixelRatio:1},
    performance: {now: () => 0}, location: {protocol:"http:", host:"localhost", search:""},
    WebSocket: class {static OPEN=1;}, ResizeObserver: class {observe() {}},
    setInterval() {}, setTimeout() {}, requestAnimationFrame() {}, URLSearchParams,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname,"../frontend/app.js"),"utf8"),sandbox);
  return code => vm.runInContext(code,sandbox);
}

test("vision is blocked by walls and map edges", () => {
  const run = radar();
  assert.equal(run(`
    (() => {
      const grid = Array.from({length:20}, () => Array(20).fill(0));
      for (const row of grid) row[8] = 1;
      const points = visionOutline({x:3, y:10, angle:0, fov:104, vision_range:11.5}, grid);
      return points.every(p => p.x < 7.5 && p.y >= -.5 && p.y < 19.5);
    })()
  `), true);
});

test("spotting smoothly colors each cone independently of global alert", () => {
  const run = radar();
  run('state = {security: {alert: "CONTACT"}}');
  for (const [confidence,color] of [[0,"rgb(146,216,248)"],[.25,"rgb(201,214,177)"],[.5,"rgb(255,212,106)"],[.75,"rgb(255,151,105)"],[1,"rgb(255,89,103)"]]) {
    assert.equal(run(`agentPalette({confidence:${confidence}, vision_status:"ALERT"}).color`),color);
    assert.equal(run(`agentPalette({visual_confidence:${confidence}}).color`),color);
  }
});

test("caution remains visible after SWAT withdrawal and stale contact", () => {
  const run = radar();
  assert.equal(run(`alertPresentation({mode:"guard", security:{alert:"SUSPICIOUS", swat_phase:"WITHDRAWN", contact_age:300}}).title`),"CAUTELA");
  assert.equal(run(`alertPresentation({mode:"guard", security:{alert:"SEARCH", contact:{x:1,y:2}}}).title`),"EVASIONE");
  assert.equal(run(`alertPresentation({mode:"guard", security:{alert:"CALM"}}).title`),"NORMALE");
});

test("all render layers accept normal, evasion and alert snapshots", () => {
  const run = radar();
  for (const phase of ["CALM","SEARCH","CONTACT","SUSPICIOUS"]) {
    run(`
      state = {
        mode:"guard", level:1, status:"RUNNING", security:{alert:"${phase}", contact:{x:8,y:8}},
        map:{width:20,height:20,grid:Array.from({length:20},()=>Array(20).fill(0)),intel:[{x:2,y:3}],exit:{x:18,y:18},zones:[]},
        player:{x:1,y:1,health:1}, guards:[{id:"G0",x:5,y:5,angle:0,alive:true,vision_status:"NORMAL"}],
        swat:[{id:"S0",role:"SHIELD",x:10,y:10,angle:1,alive:true,vision_status:"SUSPICIOUS"}],
        cameras:[{id:"C0",x:4,y:4,angle:1,online:true}], gameplay:{},
      };
      updateRenderedAgents(.033); refreshUI(); draw();
    `);
  }
});


test("heard noises make only suspicious guards' cones yellow", () => {
  const run = radar();
  for (const cue of [{vision_status:"SUSPICIOUS"}, {sound:true}, {local_sound:true}, {suspicion:.48}, {action:"INVESTIGATE"}]) {
    assert.equal(run(`agentPalette({confidence:0,...${JSON.stringify(cue)}}).color`), "rgb(255,212,106)");
    assert.equal(run(`agentPalette({confidence:1,...${JSON.stringify(cue)}}).color`), "rgb(255,89,103)");
  }
  assert.equal(run('agentPalette({confidence:0,vision_status:"NORMAL"}).color'), "rgb(146,216,248)");
});
