const {test, before} = require("node:test");
const assert = require("node:assert/strict");
let Soundscape;
before(async () => { ({Soundscape} = await import("../frontend/sound.js")); });

test("sounds play once per event and only within the server's hearing radius", () => {
  const audio = new Soundscape(); audio.context = {state:"running"};
  const played = []; audio.play = (...event) => played.push(event);
  const snapshot = {revision:1,mode:"guard",player:{x:0,y:0},noise:[
    {id:1,source:"SUPPRESSED SHOT",x:1,y:0,radius:4.5},
    {id:2,source:"SUBSONIC SHOT",x:5,y:0,radius:1.6},
  ]};
  audio.update(snapshot); audio.update(snapshot);
  assert.equal(played.length,1); assert.equal(played[0][0],"SUPPRESSED SHOT");
  assert.ok(played[0][1] > 0 && played[0][1] < 1);
});

test("muting consumes events without replaying old gunshots when re-enabled", () => {
  const audio = new Soundscape(); audio.context = {state:"running"};
  let played=0; audio.play=()=>played++;
  const snapshot={revision:1,mode:"guard",player:{x:0,y:0},noise:[{id:1,source:"GUNSHOT",x:0,y:0,radius:80}]};
  audio.enabled=false; audio.update(snapshot);
  audio.enabled=true; audio.update(snapshot); assert.equal(played,0);
  snapshot.revision=2; audio.update(snapshot); assert.equal(played,1);
});
