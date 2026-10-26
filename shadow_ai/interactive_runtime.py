from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import math
import numpy as np
from stable_baselines3 import PPO

from .playable_navigation import HumanMotion, LocalSearch, PlayableNavigation, MotionRecovery, local_search_points
from .playable_vitals import PlayableVitals, SWAT_STAMINA_CAPACITY
from .playable_actions import PlayerActions, spend_energy, projectile_damage
from .playable_blood import BloodTrail, observe_blood
from .playable_equipment import Equipment, map_pickups, FIREARMS
from .playable_combat import lane_blocked, firing_position, trace_shot, EnemyMelee, MELEE_RANGE, PLAYER_SPEED_SCALE, MELEE_PROFILES, WEAPONS, swat_loadout, player_shot
from .playable_threats import RuntimeCombatSystems
from .playable_smoke import smoke_density, smoke_obscures, smoke_escape
from .playable_facility import build_research_grid, facility_zones
from .playable_campaign import MISSIONS, mission_layout, campaign_supplies, mission_room_bounds, MAP_WIDTH, MAP_HEIGHT, ARCHIVE_DOORS
from .swat import SmokeCloud, SwatConfig

from . import (
    Config,
    FacilityAlert,
    PlayerStance,
    ShadowProtocolEnv,
    SwatAction,
    SwatRole,
    SwatTeamCore,
    TacticalAction,
)
from .soliton_security import (
    BloodMark,
    BodyEvidence,
    NoiseEvent,
    SecurityCamera,
    SecurityGuard,
    ThrownDiversion,
    build_soliton_grid,
    can_see,
    is_behind_guard,
    line_clear,
    GUARD_FOV,
    GUARD_VISION_RANGE,
    turn_towards,
)


def _f(value: Any) -> float:
    return float(value)


def _vec(v: Optional[np.ndarray]) -> Optional[dict[str, float]]:
    if v is None:
        return None
    return {"x": float(v[0]), "y": float(v[1])}


def _move_player_grid(
    pos: np.ndarray,
    direction: np.ndarray,
    speed: float,
    free_cell,
) -> np.ndarray:
    if float(np.linalg.norm(direction)) <= 1e-7:
        return pos.copy()

    direction = direction / float(np.linalg.norm(direction))
    desired = direction * speed
    candidate = pos + desired

    cx = int(round(float(candidate[0])))
    cy = int(round(float(candidate[1])))
    if free_cell(cx, cy):
        return candidate.astype(np.float32)

    # Wall slide: try the axes independently.
    x_candidate = pos + np.array([desired[0], 0.0], dtype=np.float32)
    cx = int(round(float(x_candidate[0])))
    cy = int(round(float(x_candidate[1])))
    if free_cell(cx, cy):
        return x_candidate.astype(np.float32)

    y_candidate = pos + np.array([0.0, desired[1]], dtype=np.float32)
    cx = int(round(float(y_candidate[0])))
    cy = int(round(float(y_candidate[1])))
    if free_cell(cx, cy):
        return y_candidate.astype(np.float32)

    return pos.copy()


@dataclass
class InputState:
    up: bool = False
    down: bool = False
    left: bool = False
    right: bool = False
    sprint: bool = False
    crouch: bool = False
    compliant: bool = False
    interact: bool = False
    craft: bool = False
    throw: bool = False
    lethal: bool = False
    choke: bool = False
    fire: bool = False
    use_flash: bool = False
    use_smoke: bool = False
    use_bandage: bool = False
    use_medkit: bool = False
    use_snack: bool = False
    use_grenade: bool = False
    attack_held: bool = False
    interact_held: bool = False
    dive: bool = False
    selected_item: str = ""
    aim_x: float = 0.0
    aim_y: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputState":
        return cls(
            up=bool(data.get("up")),
            down=bool(data.get("down")),
            left=bool(data.get("left")),
            right=bool(data.get("right")),
            sprint=bool(data.get("sprint")),
            crouch=bool(data.get("crouch")),
            compliant=bool(data.get("compliant")),
            interact=bool(data.get("interact")),
            craft=bool(data.get("craft")),
            throw=bool(data.get("throw")),
            lethal=bool(data.get("lethal")),
            choke=bool(data.get("choke")),
            fire=bool(data.get("fire")),
            use_flash=bool(data.get("use_flash")),
            use_smoke=bool(data.get("use_smoke")),
            use_bandage=bool(data.get("use_bandage")),
            use_medkit=bool(data.get("use_medkit")),
            use_snack=bool(data.get("use_snack")),
            use_grenade=bool(data.get("use_grenade")),
            attack_held=bool(data.get("attack_held")),
            interact_held=bool(data.get("interact_held")),
            dive=bool(data.get("dive")),
            selected_item=str(data.get("selected_item", "")),
            aim_x=float(data.get("aim_x", 0.0) or 0.0),
            aim_y=float(data.get("aim_y", 0.0) or 0.0),
        )

    def merge_transients(self, previous: "InputState") -> None:
        self.interact = self.interact or previous.interact
        self.craft = self.craft or previous.craft
        self.throw = self.throw or previous.throw
        self.lethal = self.lethal or previous.lethal
        self.fire = self.fire or previous.fire
        self.use_flash = self.use_flash or previous.use_flash
        self.use_smoke = self.use_smoke or previous.use_smoke
        self.use_bandage = self.use_bandage or previous.use_bandage
        self.use_medkit = self.use_medkit or previous.use_medkit
        self.use_snack = self.use_snack or previous.use_snack
        self.use_grenade = self.use_grenade or previous.use_grenade
        self.dive = self.dive or previous.dive

    def clear_transients(self) -> None:
        self.interact = False
        self.craft = False
        self.throw = False
        self.lethal = False
        self.fire = False
        self.use_flash = False
        self.use_smoke = False
        self.use_bandage = False
        self.use_medkit = False
        self.use_snack = False
        self.use_grenade = False
        self.dive = False

    def direction(self) -> np.ndarray:
        return np.array(
            [
                (-1.0 if self.left else 0.0) + (1.0 if self.right else 0.0),
                (-1.0 if self.up else 0.0) + (1.0 if self.down else 0.0),
            ],
            dtype=np.float32,
        )


class InteractiveGuardRuntime(RuntimeCombatSystems):
    MODE = "guard"

    def __init__(self, model_path: Path, seed: int = 71) -> None:
        self.model_path = Path(model_path)
        self.model = PPO.load(self.model_path, device="cpu")
        self.env = ShadowProtocolEnv(
            seed=seed,
            config=Config(width=MAP_WIDTH, height=MAP_HEIGHT),
        )
        self.env.grid = build_soliton_grid(48, 30)
        self.env._patrol_points = [
            np.array(p, dtype=np.float32)
            for p in [
                (12, 12), (22, 12), (27, 9), (32, 12),
                (36, 16), (31, 20), (22, 18), (13, 17),
            ]
        ]
        self.env._intel = [
            np.array([4.0, 4.0], dtype=np.float32),
            np.array([24.0, 6.0], dtype=np.float32),
            np.array([44.0, 22.0], dtype=np.float32),
        ]
        self.env._exit = np.array([MAP_WIDTH - 2., MAP_HEIGHT - 2.], dtype=np.float32)
        self.seed = seed
        self.rng = np.random.default_rng(seed + 991)
        self.last_action = TacticalAction.HOLD
        self.last_phase = "routine"

        # RL movement values were tuned for a 0.20 s training step.
        # The playable runtime runs at 30 Hz, so movement must be scaled
        # instead of applying a full RL step every render/network frame.
        self.game_dt = 1.0 / 30.0
        training_dt = float(self.env.cfg.dt)
        step_scale = self.game_dt / training_dt

        # Small gameplay-only movement relaxation: patrol is slightly slower
        # than in curriculum so the human has room to react.
        guard_gameplay_scale = 0.75
        self.env.cfg.guard_speed *= step_scale * guard_gameplay_scale
        self.env.cfg.buddy_speed *= step_scale * guard_gameplay_scale
        self.env.cfg.reinforcement_speed *= step_scale * guard_gameplay_scale

        # Tactical suit speeds apply only to the human playable runtime.
        self.env.cfg.player_walk_speed = 2.6 * PLAYER_SPEED_SCALE * self.game_dt
        self.env.cfg.player_sprint_speed = 3.7 * PLAYER_SPEED_SCALE * self.game_dt
        self.env.cfg.player_stealth_speed = 1.3 * PLAYER_SPEED_SCALE * self.game_dt
        self.env.cfg.dt = self.game_dt
        self.env.cfg.vision_outer = GUARD_VISION_RANGE
        self.env.cfg.main_fov_degrees = GUARD_FOV
        self.env.cfg.peripheral_fov_degrees = GUARD_FOV

        # ~5 tactical decisions/s, while physics/input run at 30 Hz.
        self.policy_every = max(1, round(0.20 / self.game_dt))
        self.policy_tick = 0
        self.spawn_grace = 0.0
        self.status = "RUNNING"
        self.message = ""
        self._interact_latch = False
        self.aux_guards: list[SecurityGuard] = []
        self.cameras: list[SecurityCamera] = []
        self.noise_events: list[NoiseEvent] = []
        self.diversions: list[ThrownDiversion] = []
        self.bodies: list[BodyEvidence] = []
        self.blood: list[BloodMark] = []
        self.team_contact: Optional[np.ndarray] = None
        self.team_contact_age = 999.0
        self.team_contact_source = ""
        self.facility_alert = "CALM"
        self.radio_updates = 0
        self.lead_alive = True
        self._support_calls = {}
        self._support_cooldowns = {}
        self._room_entries = {}
        self.smoke_clouds = []
        self.flash_events = []
        self.player_flash_timer = 0.0
        self.lead_unconscious = False
        self.takedown: Optional[dict[str, Any]] = None
        self.diversion_inventory = 0
        self.scrap = 0
        self.craft_cooldown = 0.0
        self.bloody_timer = 0.0
        self.last_blood_pos: Optional[np.ndarray] = None
        self._next_event_id = 1
        self.level = 1
        self.campaign_items = set()
        self.env._move_towards = self._move_primary
        self.reset("routine")

    @property
    def scenarios(self) -> list[str]:
        return [
            "routine",
            "peripheral",
            "attention",
            "repeat_visual",
            "sound",
            "repeat_sound",
            "visual",
            "lost_contact",
            "search",
            "arrest",
            "theft",
        ]

    def advance_level(self) -> bool:
        if self.status != "ESCAPED" or self.level >= 3:
            return False
        loadout = self.equipment.next_level_loadout()
        vitals = replace(self.player_vitals)
        self.level += 1
        self.reset("routine")
        loadout.security_card |= self.equipment.security_card
        self.equipment = loadout
        self.player_vitals = vitals
        return True

    def reset(self, scenario: str = "routine") -> None:
        if scenario not in self.scenarios:
            scenario = "routine"
        self.seed += 1
        mission = MISSIONS[self.level]
        self.env.grid, self._mission_zones = mission_layout(self.level)
        self.archive_unlocked = False
        self.env._intel = [np.array([4.,4.],dtype=np.float32) for _ in range(3)]
        for index,point in zip(mission['indices'],mission['points']):
            self.env._intel[index] = np.array(point,dtype=np.float32)
        if self.level == 1:
            self.campaign_items.clear()
        self.env.reset(seed=self.seed, options={"scenario": scenario})

        # The interactive demo always exposes all three objectives to the
        # human player. Guards still learn a theft only after inspection.
        self.env.episode_has_intel[:] = 1
        self.env.episode_intel_order = [0, 1, 2]
        self.env.player_has_intel[:] = 0
        self.pathogen_exposure = False
        self.env.player_goal_idx = 0
        self.env.buddy_available = False
        self.env.buddy_pos = None
        self.env.reinforcements_called = False
        self.env.reinforcements_active = False
        self.env.reinforcement_positions = []
        self.env.reinforcement_angles = []
        # Playable routine uses a deterministic protected infiltration spawn,
        # not the curriculum's random routine spawn.
        if scenario == "routine":
            self.env.guard_pos = np.array([22.0, 12.0], dtype=np.float32)
            self.env.guard_angle = math.pi
            self.env.player_pos = np.array([2.0, 27.0], dtype=np.float32)
            self.env.player_velocity[:] = 0.0
            self.env.player_current_speed = 0.0
            self.env.visual_confidence = 0.0
            self.env.personal_suspicion = 0.0
            self.env.intruder_known = False
            self.env.radio_reported = False
            self.env.last_seen_pos = None
            self.env.last_seen_age = self.env.cfg.memory_seconds + 1.0
            self.env.sense = self.env._sense()
        self.spawn_grace = 2.5

        self.navigator = PlayableNavigation(
            self.env.grid, self.env._cell, self.env._free,
            self.env._astar, self.env._nearest_free_cell)
        self._navigation = self.navigator.paths
        self._searches: dict[str, LocalSearch] = {}
        self._motions: dict[str, HumanMotion] = {}
        # Every playable scenario starts in the same covered infiltration alcove.
        self.env.player_pos = np.array([2.0, 27.0], dtype=np.float32)
        self.spawn_safe = True
        self.env.guard_pos = self._navigable_goal(self.env.guard_pos)
        if self._inside_spawn(self.env.guard_pos):
            self.env.guard_pos = np.array([7.0, 24.0], dtype=np.float32)
        self.env.player_velocity[:] = 0.0
        self.env.player_current_speed = 0.0
        self.env.visual_confidence = 0.0
        self.env.personal_suspicion = 0.0
        self.env.sense = self.env.sense.__class__()
        self.status = "RUNNING"
        self.message = ""
        self.last_action = TacticalAction.HOLD
        self._primary_policy_action = TacticalAction.PATROL
        self.policy_tick = 0
        self.last_phase = self.env._tactical_phase(self.env._decision_context())
        self._interact_latch = False
        self._setup_security()
        self.equipment = Equipment(knife=True)
        self.equipment.security_card = 'SECURITY_CARD' in self.campaign_items
        self.player_actions = PlayerActions()
        self._motion_recovery = MotionRecovery()
        self.pickups = campaign_supplies(self.navigator,self.level)
        self._reset_combat_systems()
        self._contact_helpers = {}
        self._equipment_dropped = set()
        self.aux_guards = self.aux_guards[:(2, 4, 7)[self.level - 1]]
        self.cameras = self.cameras[:(2, 4, 6)[self.level - 1]]
        self.env.episode_has_intel[:] = 0
        self.env.episode_has_intel[mission['indices']] = 1
        self.env.episode_intel_order = mission['indices'][:]
        self.swat_units = []
        self.radio_timer = None
        self.swat_phase = "QUIET"
        self.radio_source = ""
        self._primary_audio_goal = None
        self._primary_audio_timer = 0.0
        route = tuple(tuple(float(v) for v in self._navigable_goal(point)) for point in self.env._patrol_points)
        self._lead_patrol = SecurityGuard("G0", self.env.guard_pos.copy(), self.env.guard_angle, route)
        self._lead_patrol.patrol_index = min(range(len(route)), key=lambda i: np.linalg.norm(self.env.guard_pos - route[i]))
        self._primary_patrol_action = None

    def _setup_security(self) -> None:
        routes = [
            ((3, 4), (7, 4), (7, 11), (3, 11)),
            ((3, 18), (7, 18), (7, 24), (3, 24)),
            ((12, 3), (17, 3), (17, 12), (12, 12)),
            ((30, 3), (36, 3), (36, 12), (30, 12)),
            ((21, 17), (26, 17), (26, 27), (20, 27)),
            ((31, 17), (36, 17), (36, 27), (30, 27)),
            ((40, 3), (45, 3), (45, 12), (40, 12)),
        ]
        # Patrol the new wings as well as the original infiltration sectors.
        routes[0] += ((23,31), (30,35), (40,35))
        routes[1] += ((48,24), (54,24), (58,30), (48,35))
        routes[3] += ((49,6), (54,6), (58,10), (49,14))
        self.aux_guards = []
        for idx, route in enumerate(routes, start=1):
            route = tuple(tuple(self._navigable_goal(np.array(point, dtype=np.float32))) for point in route)
            start = np.asarray(route[0], dtype=np.float32)
            nxt = np.asarray(route[1], dtype=np.float32)
            angle = math.atan2(float(nxt[1] - start[1]), float(nxt[0] - start[0]))
            self.aux_guards.append(
                SecurityGuard(
                    guard_id=f"G{idx}",
                    pos=start.copy(),
                    angle=angle,
                    patrol=route,
                )
            )

        # Lens centers sit just outside the wall face, looking into free space.
        camera_specs = [
            ("C1", (9.58, 2), 0.0),
            ("C2", (20.42, 5), math.pi),
            ("C3", (29.42, 3), math.pi),
            ("C4", (28.42, 19), math.pi),
            ("C5", (37.58, 19), 0.0),
            ("C6", (45.42, 24), math.pi),
        ]
        self.cameras = [
            SecurityCamera(
                camera_id=camera_id,
                pos=np.asarray(pos, dtype=np.float32),
                center_angle=angle,
                sweep=math.radians(50.0),
                phase=idx * 0.77,
                angle=angle,
            )
            for idx, (camera_id, pos, angle) in enumerate(camera_specs)
        ]
        self.noise_events = []
        self.diversions = []
        self.bodies = []
        self.blood = []
        self.team_contact = None
        self.team_contact_age = 999.0
        self.team_contact_source = ""
        self.facility_alert = "CALM"
        self.security_compromised = False
        self._diversion_counts = {}
        self._diversion_assignments = {}
        self._launch_responses = {}
        self._observed_launches = {}
        self._local_diversion_guards = set()
        self.guard_dialogue = {}
        self._dialogue_timer = 0.0
        self.radio_updates = 0
        self.player_vitals = PlayableVitals()
        self._wound_trail = BloodTrail()
        self.threat_level = 0
        self._strong_spots = {}
        self._spot_active = set()
        self._guard_dispatch = None
        self._regular_reinforcements = 0
        self._intel_checks = {}
        self._reported_missing_intel = set()
        self._player_identified = False
        self._search_reports = {}
        self._intel_sweeps = {}
        self._combat_casualties = set()
        self._reinforced_casualties = set()
        self._swat_serial = 0
        self._melee_attacks = {}
        self.shots = []
        self._support_calls = {}
        self._support_cooldowns = {}
        self._room_entries = {}
        self.smoke_clouds = []
        self.flash_events = []
        self.player_flash_timer = 0.0
        self.lead_alive = True
        self.lead_unconscious = False
        self.takedown = None
        self.diversion_inventory = 0
        self.scrap = 0
        self.craft_cooldown = 0.0
        self.bloody_timer = 0.0
        self.last_blood_pos = None
        self._next_event_id = 1

    def _new_id(self) -> int:
        value = self._next_event_id
        self._next_event_id += 1
        return value

    def _primary_active(self) -> bool:
        return self.lead_alive and not self.lead_unconscious

    def _inside_spawn(self, pos: np.ndarray) -> bool:
        return float(pos[0]) < 4.5 and float(pos[1]) > 25.5

    @property
    def spawn_protected(self) -> bool:
        return self.spawn_grace > 1e-6 or (self.spawn_safe and self._inside_spawn(self.env.player_pos))

    def _publish_contact(self, pos: np.ndarray, source: str) -> None:
        if self.spawn_protected:
            return
        point = np.asarray(pos, dtype=np.float32).copy()
        if self.team_contact is None or self.team_contact_age > 12.0:
            self.player_vitals.boost_remaining = 2.5
        changed = (
            self.team_contact is None
            or float(np.linalg.norm(point - self.team_contact)) > 0.45
            or self.team_contact_age > 0.7
        )
        self._contact_response(point, source)
        for _, unit in self.swat_units:
            if unit.active and changed and unit.state != "CHASE":
                self._searches.pop(unit.guard_id, None)
                unit.target = point.copy()
                unit.state = "RESPOND"
                unit.last_seen = point.copy()
                unit.last_seen_age = 0.0
        self.team_contact = point
        self._player_identified = True
        self.security_compromised = True
        self.team_contact_age = 0.0
        self.team_contact_source = source
        self.facility_alert = "CONTACT"
        if changed and self.swat_phase == "ACTIVE":
            self.radio_updates += 1
        # The lead receives a reported position, never hidden live truth.
        if self._primary_active() and source != "G0" and self.swat_phase in {"DISPATCH", "ACTIVE"}:
            self.env.last_seen_pos = point.copy()
            self.env.last_seen_age = 0.0
            self.env.intruder_known = True
            self.env.time_since_confirmed_intruder = 0.0
            self.env.alert = max(int(self.env.alert), int(FacilityAlert.SEARCHING))

    def _active_guards(self):
        guards = [g for g in self.aux_guards + [u for _, u in self.swat_units] if g.active]
        if self._primary_active():
            self._lead_patrol.pos = self.env.guard_pos.copy()
            self._lead_patrol.angle = self.env.guard_angle
            guards.append(self._lead_patrol)
        return guards

    def _start_swat_response(self, source):
        # SWAT takes over any regular patrol that has not arrived yet.
        self._guard_dispatch = None
        if not self.swat_units and self.radio_timer is None:
            self.radio_timer = 1.0
            self.swat_phase = "SPOTTED"
            self.radio_source = source

    def _contact_response(self, point, source):
        if source.startswith("S"):
            return
        guards = self._active_guards()
        spotter = next((g for g in guards if g.guard_id == source), None)
        if spotter is None:
            self._start_swat_response(source)
            return
        helper = next((g for g in guards if g.guard_id == self._contact_helpers.get(source)), None)
        if helper is not None:
            if source in self._support_calls:
                self._support_calls[source]["point"] = point.copy()
            elif helper.state in {"PATROL", "OBSERVE", "SUPPORT", "PROTECT INTEL", "SECURE EXFIL"}:
                helper.state, helper.target = "SUPPORT", point.copy()
                helper.last_seen, helper.last_seen_age = point.copy(), 0.0
            return
        candidates = []
        for guard in guards:
            if guard.guard_id == source or guard.guard_id.startswith("S"):
                continue
            if guard.state not in {"PATROL", "OBSERVE", "PROTECT INTEL", "SECURE EXFIL", "CHASE"}:
                continue
            if np.linalg.norm(guard.pos - spotter.pos) <= 8.0:
                if self.navigator.path_length(spotter.pos, guard.pos) <= 12:
                    candidates.append(guard)
        if candidates:
            helper = min(candidates, key=lambda g: float(np.linalg.norm(g.pos - spotter.pos)))
            self._contact_helpers[source] = helper.guard_id
            self._support_calls[source] = {"remaining": 1.2, "point": point.copy(),
                                           "helper": helper.guard_id, "confirmed": True}
            self.guard_dialogue[source] = f"Contatto! {helper.guard_id}, dammi una mano."
            self._dialogue_timer = 5.0
            if self.radio_timer is None and not self.swat_units:
                self.swat_phase = "LOCAL SUPPORT"
        else:
            self._start_swat_response(source)

    def _step_radio(self) -> None:
        if self.radio_timer is None or self.spawn_protected:
            return
        self.radio_timer = max(0.0, self.radio_timer - self.game_dt)
        if self.radio_timer > 1e-6:
            return
        if self.swat_phase == "SPOTTED":
            self.swat_phase = "RADIO"
            self.radio_timer = 2.0
            return
        if self.swat_phase == "RADIO":
            self.swat_phase = "DISPATCH"
            self.radio_timer = 1.0
            self.radio_updates += 1
            return
        self.radio_timer = None
        self.swat_phase = "ACTIVE"
        for idx, role in enumerate(("SHIELD", "ASSAULT", "LEADER")):
            # All insertion cells are navigable; no unit spawns inside x=46/y=14.
            unit = SecurityGuard(
                guard_id=f"S{self._swat_serial + idx}",
                pos=np.array([self.env.cfg.width - 1., 14.0 + idx], dtype=np.float32),
                angle=math.pi, patrol=((self.env.cfg.width - 1., 14), (self.env.cfg.width - 1., 16)),
            )
            unit.armor = 1.0
            unit.vitals.stamina_capacity = SWAT_STAMINA_CAPACITY
            unit.has_shield = role == "SHIELD"
            unit.weapon = swat_loadout(role, self.seed)
            unit.ammo = 8 if unit.weapon == 'SHOTGUN' else 24
            unit.grenade_count = 0 if unit.has_shield else 1
            unit.exit_pos = unit.pos.copy()
            unit.state = "RESPOND"
            unit.target = self.team_contact.copy()
            unit.last_seen = self.team_contact.copy()
            unit.last_seen_age = self.team_contact_age
            self._motions.pop(unit.guard_id, None)
            self.navigator.paths.pop(unit.guard_id, None)
            self.swat_units.append((role, unit))
        self._swat_serial += 3

    def _navigable_goal(self, goal: np.ndarray) -> np.ndarray:
        return self.navigator.goal(goal)

    def _navigate(self, key: str, pos: np.ndarray, goal: Optional[np.ndarray], speed: float) -> np.ndarray:
        return self.navigator.move(key, pos, goal, speed)

    def _move_primary(self, pos, goal, speed):
        running = goal is not None and np.linalg.norm(goal - pos) > 0.2 and (
            self.env.sense.visual or self.threat_level > 0 or speed > float(self.env.cfg.guard_speed) * 1.05)
        self._lead_patrol.vitals.step(running, self.game_dt)
        self._lead_vitals_stepped = True
        speed *= self._lead_patrol.vitals.speed_multiplier
        if running and not self._lead_patrol.vitals.can_sprint:
            speed = min(speed, float(self.env.cfg.guard_speed) * 0.7)
        motion = self._motions.setdefault("G0", HumanMotion())
        moved, angle = motion.move(self.navigator, "G0", pos, self.env.guard_angle,
                                   goal, speed / self.game_dt, self.game_dt)
        moved = self._motion_recovery.step(self.navigator, "G0", pos, moved, goal,
            speed / self.game_dt, self.game_dt, goal is not None)
        self.env.guard_angle = angle
        return moved

    def _patrol_step(self, guard: SecurityGuard) -> tuple[Optional[np.ndarray], Optional[float]]:
        if guard.state == "OBSERVE":
            guard.pause_remaining = max(0.0, guard.pause_remaining - self.game_dt)
            elapsed = guard.pause_duration - guard.pause_remaining
            look = guard.pause_angle + 0.65 * math.sin(elapsed / guard.pause_duration * math.tau)
            if guard.pause_remaining > 1e-6:
                return None, look
            guard.advance_patrol()
            guard.state = "PATROL"
        guard.target = guard.patrol_target()
        if np.linalg.norm(guard.target - guard.pos) <= 0.2:
            guard.state = "OBSERVE"
            guard.pause_duration = float(self.rng.uniform(1.8, 4.2))
            guard.pause_remaining = guard.pause_duration
            guard.pause_angle = guard.angle
            return None, guard.angle
        return guard.target, None

    def _step_swat_departures(self) -> None:
        self.swat_units = [(role, unit) for role, unit in self.swat_units if unit.state != "EXITED"]
        if self.swat_units and all(unit.state in {"WITHDRAW", "LEAVING"} for _, unit in self.swat_units):
            self.swat_phase = "RETURNING"
        elif not self.swat_units and self.swat_phase in {"ACTIVE", "RETURNING"}:
            self.swat_phase = "WITHDRAWN"
        elif self.swat_units and self.swat_phase == "RETURNING":
            self.swat_phase = "ACTIVE"

    def _vision_status(self, state, visual=False, suspicion=0.0):
        if visual:
            return "ALERT"
        if suspicion >= 0.25:
            return "SUSPICIOUS"
        return "NORMAL"

    def _update_facility_alert(self) -> None:
        if self.spawn_protected and not self.security_compromised:
            self.facility_alert = "CALM"
        elif self.team_contact is not None:
            if self.team_contact_age <= 0.5:
                self.facility_alert = "CONTACT"
            elif self.team_contact_age <= 12.0 or any(
                unit.active and unit.state in {"RESPOND", "SEARCH", "VERIFY", "STACK", "BREACH WAIT", "CLEAR ROOM"}
                for _, unit in self.swat_units
            ):
                self.facility_alert = "SEARCH"
            else:
                self.facility_alert = "SUSPICIOUS"
        elif self.security_compromised:
            self.facility_alert = "SUSPICIOUS"
        else:
            self.facility_alert = "CALM"
        if self.security_compromised:
            self.env.alert = max(int(self.env.alert), int(FacilityAlert.SUSPICIOUS))
            self.env.personal_suspicion = max(self.env.personal_suspicion, 0.48)

    def _raise_evidence_alert(self, source: str, point=None, observer="G0") -> None:
        self.security_compromised = True
        if source in {"BODY FOUND", "GUNFIRE"}:
            self.threat_level = 2
        if source == "BODY FOUND":
            self.guard_dialogue[observer] = "Compagno a terra! Armi pronte, mandate una pattuglia."
            self._dialogue_timer = 7.0
            if point is not None:
                self._request_support(observer, point)
                if observer == "G0":
                    self._primary_audio_goal = point.copy()
                    self._primary_audio_timer = 12.0
                    self._searches["G0"] = LocalSearch(local_search_points(self.env.grid, point, 0), scan_seconds=1.2)
                if (self._guard_dispatch is None and self._regular_reinforcements < 2
                        and self.radio_timer is None and not self.swat_units):
                    self._guard_dispatch = {"remaining": 2.0, "point": point.copy()}
        if self.facility_alert != "CONTACT":
            self.facility_alert = "SEARCH"
        self.team_contact_source = source
        self.env.alert = max(
            int(self.env.alert),
            int(FacilityAlert.SEARCHING),
        )

    def _nearest_takedown_target(self) -> Optional[tuple[str, Any, float]]:
        candidates: list[tuple[str, Any, float]] = []
        player = self.env.player_pos
        if self._primary_active() and is_behind_guard(
            player,
            self.env.guard_pos,
            self.env.guard_angle,
            2.0,
        ) and line_clear(self.env.grid, player, self.env.guard_pos):
            candidates.append(
                ("G0", None, float(np.linalg.norm(player - self.env.guard_pos)))
            )
        for guard in self.aux_guards + [u for _, u in self.swat_units]:
            if guard.active and is_behind_guard(
                player,
                guard.pos,
                guard.angle,
                2.0,
            ) and line_clear(self.env.grid, player, guard.pos):
                candidates.append(
                    (guard.guard_id, guard, float(np.linalg.norm(player - guard.pos)))
                )
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[2])

    def _target_position(self, guard_id: str) -> Optional[np.ndarray]:
        if guard_id == "G0":
            return self.env.guard_pos
        guard = next((g for g in self.aux_guards + [u for _, u in self.swat_units] if g.guard_id == guard_id), None)
        return None if guard is None else guard.pos

    def _down_guard(self, guard_id: str, lethal: bool) -> None:
        if guard_id == "G0":
            pos = self.env.guard_pos.copy()
            angle = float(self.env.guard_angle)
            self.lead_alive = not lethal
            self.lead_unconscious = not lethal
            self._lead_patrol.vitals.health = 0.0 if lethal else self._lead_patrol.vitals.health
            self.env.player_velocity[:] = 0.0
        else:
            guard = next((g for g in self.aux_guards + [u for _, u in self.swat_units] if g.guard_id == guard_id), None)
            if guard is None or not guard.active:
                return
            pos = guard.pos.copy()
            angle = float(guard.angle)
            guard.alive = not lethal
            guard.unconscious = not lethal
            if lethal:
                guard.vitals.health = 0.0
            guard.state = "DOWN"

        self.bodies.append(
            BodyEvidence(
                body_id=self._new_id(),
                guard_id=guard_id,
                pos=pos,
                angle=angle,
                lethal=lethal,
                fall=1.0 if lethal else 0.18,
            )
        )
        if lethal:
            if self.threat_level >= 2 or self._player_identified:
                self._combat_casualties.add(guard_id)
            self._drop_guard_equipment(guard_id, pos)
            self.blood.append(BloodMark(self._new_id(), pos.copy()))
            if np.linalg.norm(self.env.player_pos - pos) < 1.5:
                self.bloody_timer = 22.0
                self.last_blood_pos = None
            self.noise_events.append(
                NoiseEvent(
                    event_id=self._new_id(),
                    pos=pos.copy(),
                    radius=6.2,
                    ttl=0.8,
                    source="BODY IMPACT",
                )
            )
            self.message = f"{guard_id} LETHAL TAKEDOWN â€” IMPACT HEARD"
        else:
            self._drop_guard_equipment(guard_id, pos)
            self.message = f"{guard_id} SILENT TAKEDOWN"
        self.takedown = None

    def _process_takedown(self, inp: InputState) -> None:
        if self.player_actions.movement_locked:
            return
        if self.takedown is not None:
            guard_id = self.takedown["guard_id"]
            guard = next((g for g in self._active_guards() if g.guard_id == guard_id), None)
            valid = (inp.choke and np.linalg.norm(inp.direction()) < .1 and guard is not None
                     and is_behind_guard(self.env.player_pos, guard.pos, guard.angle, 2.0)
                     and line_clear(self.env.grid, self.env.player_pos, guard.pos)
                     and spend_energy(self.player_vitals, .20 * self.game_dt))
            if not valid:
                self.takedown = None
                self.message = "SOFFOCAMENTO INTERROTTO"
                return
            self.takedown["progress"] += self.game_dt
            if self.takedown["progress"] >= 3.:
                self._down_guard(guard_id, lethal=False)
            return
        if inp.lethal and self.equipment.knife and self.player_actions.melee_cooldown <= 0:
            targets = [g for g in self._active_guards() if np.linalg.norm(g.pos - self.env.player_pos) <= 2.0
                       and line_clear(self.env.grid, self.env.player_pos, g.pos)]
            guard = min(targets, key=lambda g: np.linalg.norm(g.pos - self.env.player_pos)) if targets else None
            swing = self.player_actions.knife_swing(self.env.player_pos, self.env.grid, self.player_vitals,
                                                     None if guard is None else guard.pos)
            if swing is None:
                return
            if swing["impact"]:
                self.noise_events.append(NoiseEvent(self._new_id(), swing["pos"], 6., .7, "KNIFE IMPACT"))
            if guard is not None:
                if is_behind_guard(self.env.player_pos, guard.pos, guard.angle, 2.0):
                    self._down_guard(guard.guard_id, lethal=True)
                else:
                    attack = self._melee_attacks.get(guard.guard_id)
                    if guard.has_shield and attack is not None and attack.windup > 0:
                        self.message = "SCUDO — IL COLPO NON È DISARMABILE"
                        return
                    self._counter_disarm(guard.guard_id, guard, guard.has_shield)
                    health = max(0., guard.vitals.health - .25)
                    guard.angle = math.atan2(float(self.env.player_pos[1] - guard.pos[1]), float(self.env.player_pos[0] - guard.pos[0]))
                    guard.visual_confidence = 1.
                    guard.last_seen = self.env.player_pos.copy()
                    guard.state = "CHASE"
                    self._publish_contact(self.env.player_pos, guard.guard_id)
                    if guard.guard_id == "G0":
                        self.env.guard_angle = guard.angle
                        self.env.visual_confidence = 1.
                    if health <= 0:
                        self._down_guard(guard.guard_id, lethal=True)
                    else:
                        guard.vitals.health = health
                        guard.vitals.hit_stun_remaining = .3
                        guard.aim_timer = 0.
                        guard.fire_goal = None
            return
        if inp.choke:
            target = self._nearest_takedown_target()
            if target is not None and spend_energy(self.player_vitals, .15):
                self.takedown = dict(guard_id=target[0], kind="CHOKE", progress=self.game_dt, duration=3.)
                self.message = "TIENI F — SOFFOCAMENTO"

    def _craft_or_throw(self, inp: InputState) -> None:
        if not inp.throw:
            return
        if not self.equipment.take_diversion():
            self.message = "SELEZIONA UN BOSSOLO O UN CARICATORE NELLO ZAINO"
            return

        target = np.array([inp.aim_x, inp.aim_y], dtype=np.float32)
        delta = target - self.env.player_pos
        distance = float(np.linalg.norm(delta))
        if distance <= 0.25:
            facing = self.env.player_velocity
            if float(np.linalg.norm(facing)) <= 1e-5:
                facing = np.array([1.0, 0.0], dtype=np.float32)
            target = self.env.player_pos + facing / max(
                float(np.linalg.norm(facing)), 1e-5
            ) * 5.0
        elif distance > 8.0:
            target = self.env.player_pos + delta / distance * 8.0
        tx, ty = self.env._cell(target)
        if not self.env._free(tx, ty):
            tx, ty = self.env._nearest_free_cell(tx, ty)
            target = np.array([tx, ty], dtype=np.float32)
        self.diversions.append(
            ThrownDiversion(
                device_id=self._new_id(),
                start=self.env.player_pos.copy(),
                target=target.copy(),
                pos=self.env.player_pos.copy(),
            )
        )
        self.diversions[-1].pulses_left = 1
        self.message = "OGGETTO LANCIATO"
        self._observe_diversion_launch(self.diversions[-1])

    def _collect_pickups(self) -> None:
        if self.player_actions.knockdown_remaining > 0 or self.player_actions.rise_timer > 0:
            return
        collected = []
        for item in self.pickups:
            if item.get("fixed"):
                continue
            if (np.linalg.norm(item["pos"] - self.env.player_pos) > 1.2
                    or not line_clear(self.env.grid, self.env.player_pos, item["pos"])):
                continue
            kind, amount = item["kind"], item["amount"]
            if not self.equipment.collect(kind, amount):
                continue
            if kind == "SECURITY_CARD":
                self.campaign_items.add('SECURITY_CARD')
            collected.append(item)
        if collected:
            ids = {item["id"] for item in collected}
            self.pickups = [item for item in self.pickups if item["id"] not in ids]
            self.message = "RACCOLTO: " + ", ".join(item["kind"] for item in collected)

    def _drop_guard_equipment(self, guard_id, pos):
        if guard_id in self._equipment_dropped:
            return
        self._equipment_dropped.add(guard_id)
        guard = next((g for g in self.aux_guards + [u for _, u in self.swat_units]
                      if g.guard_id == guard_id), self._lead_patrol)
        items = [(guard.weapon, guard.ammo), ("MAGAZINE", 1)]
        guard.ammo = 0
        if guard_id.startswith("S"):
            items += [("FLASH", guard.flash_count), ("SMOKE", guard.smoke_count)]
            items += [("GRENADE", guard.grenade_count)]
            guard.grenade_count = 0
            guard.flash_count = guard.smoke_count = 0
        for kind, amount in items:
            if kind and (amount or kind in FIREARMS):
                self.pickups.append({"id": f"{guard_id}-{kind}", "kind": kind, "amount": amount,
                                     "pos": pos.copy(), "source": guard_id})

    def _equipment_target(self, inp):
        point = np.array([inp.aim_x, inp.aim_y], dtype=np.float32)
        delta = point - self.env.player_pos
        distance = float(np.linalg.norm(delta))
        if distance > 8:
            point = self.env.player_pos + delta / distance * 8
        last = self.env.player_pos.copy()
        for t in np.linspace(0, 1, 65):
            candidate = self.env.player_pos + (point - self.env.player_pos) * t
            if not line_clear(self.env.grid, self.env.player_pos, candidate):
                break
            last = candidate
        return last

    def _player_security_event(self, source, point):
        self.spawn_safe = False
        self.spawn_grace = 0.
        self.security_compromised = True
        if source == "SMOKE":
            self.threat_level = max(self.threat_level, 1)
            for guard in self._active_guards():
                guard.suspicion = max(guard.suspicion, .48)
            self.env.personal_suspicion = max(self.env.personal_suspicion, .48)
            self._update_facility_alert()
            return
        self.threat_level = 2
        self._publish_contact(point, "GUNFIRE")
        self.env.alert = int(FacilityAlert.CONFIRMED)
        self.env.personal_suspicion = 1.
        for guard in self._active_guards():
            guard.suspicion = 1.
            if guard.state != "CHASE":
                guard.state = "RESPOND"
                guard.target = point.copy()
                guard.last_seen = point.copy()
                guard.last_seen_age = 0.
                self._searches.pop(guard.guard_id, None)
        self._primary_audio_goal = point.copy()
        self._primary_audio_timer = 12.

    def _player_equipment_action(self, inp):
        self.equipment.fire_cooldown = max(0.0, self.equipment.fire_cooldown - self.game_dt)
        if inp.use_smoke and self.equipment.smoke > 0 and spend_energy(self.player_vitals, .12):
            self.equipment.smoke -= 1
            self.smoke_clouds.append({"pos": self._equipment_target(inp), "radius": 3.0, "ttl": 8.0})
            self._player_security_event("SMOKE", self.smoke_clouds[-1]["pos"])
        if inp.use_flash and self.equipment.flash > 0 and spend_energy(self.player_vitals, .12):
            self.equipment.flash -= 1
            point = self._equipment_target(inp)
            self.flash_events.append({"pos": point.copy(), "ttl": .45})
            for guard in self._active_guards():
                if np.linalg.norm(guard.pos - point) < 3 and line_clear(self.env.grid, point, guard.pos):
                    guard.blinded_timer = 2.4
            if np.linalg.norm(self.env.player_pos - point) < 3 and line_clear(self.env.grid, point, self.env.player_pos):
                self.player_flash_timer = 2.4
        if (not inp.fire or not self.player_vitals.can_aim or self.takedown or self.equipment.fire_cooldown > 0
                or self.player_actions.knockdown_remaining > 0 or self.player_actions.rise_timer > 0):
            return
        if not self.equipment.weapon or self.equipment.ammo <= 0:
            self.message = "RACCOGLI UN'ARMA E MUNIZIONI [E]"
            return
        origin = self.env.player_pos.copy()
        delta = np.array([inp.aim_x, inp.aim_y], dtype=np.float32) - origin
        distance = float(np.linalg.norm(delta))
        if distance < .1:
            return
        endpoint, guard = player_shot(self.env.grid,origin,origin+delta,self._active_guards(),self.equipment.weapon,self.rng)
        if not spend_energy(self.player_vitals, .08 if self.equipment.weapon in {'PISTOL','RIFLE'} else WEAPONS[self.equipment.weapon].energy):
            return
        self.equipment.ammo -= 1
        self.equipment.fire_cooldown = {'RIFLE':.2,'PISTOL':.45}.get(self.equipment.weapon,WEAPONS[self.equipment.weapon].cooldown)
        self._player_security_event("GUNFIRE", origin)
        if guard is not None:
            health, guard.armor, effect = projectile_damage(guard.vitals.health, guard.armor,
                guard.vitals, guard.has_shield, guard.angle, guard.pos, origin, self.equipment.weapon)
            if health <= 0:
                self._down_guard(guard.guard_id, lethal=True)
            else:
                guard.vitals.health = health
            self.message = effect
        self.shots.append({"from": _vec(origin), "to": _vec(endpoint), "ttl": .12})
        self.noise_events.append(NoiseEvent(self._new_id(), origin, 14., .72, "GUNSHOT"))

    def _observe_diversion_launch(self, device: ThrownDiversion) -> None:
        if self.spawn_protected:
            return
        guards = [g for g in self.aux_guards + [unit for _, unit in self.swat_units]
                  if g.active and not (self.takedown and self.takedown["guard_id"] == g.guard_id)]
        observers = [(g.guard_id, g.pos, g.angle) for g in guards]
        if self._primary_active() and not (self.takedown and self.takedown["guard_id"] == "G0"):
            observers.append(("G0", self.env.guard_pos, self.env.guard_angle))
        witnesses = [(gid, pos) for gid, pos, angle in observers
                     if can_see(self.env.grid, pos, angle, device.start,
                                GUARD_VISION_RANGE, GUARD_FOV, self.env.cfg.vision_immediate)]
        if not witnesses:
            return
        witnesses.sort(key=lambda item: (float(np.linalg.norm(item[1] - device.start)), item[0]))
        # Only the launch visible at this instant supplies a source estimate.
        # Later movement of the hidden thrower never updates this point.
        source = self._navigable_goal(device.start + self.rng.normal(0.0, 0.35, 2).astype(np.float32))
        self.security_compromised = True
        self._observed_launches[device.device_id] = {gid for gid, _ in witnesses}
        # Radio coordination makes the seen launch take priority over its pulses.
        self._diversion_assignments[device.device_id] = {}
        for index, (gid, pos) in enumerate(witnesses):
            direction = source - pos
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([1.0, 0.0], dtype=np.float32)
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            side = np.array([-direction[1], direction[0]], dtype=np.float32)
            goal = source if index == 0 else self._navigable_goal(
                source - direction * 1.5 + side * (1.5 if index % 2 else -1.5))
            self._launch_responses[gid] = {"goal": goal.copy(), "source": source.copy(),
                                          "device": device.device_id}
            self._searches.pop(gid, None)
            self._local_diversion_guards.discard(gid)
            self.guard_dialogue[gid] = ("Ho visto il lancio. Controllo da dove e partito."
                                       if index == 0 else "Ti copro sul fianco della sorgente.")
            if gid == "G0":
                self._primary_audio_goal = goal.copy()
                self._primary_audio_timer = 45.0
            else:
                guard = next(g for g in guards if g.guard_id == gid)
                if guard.state != "CHASE":
                    guard.state = "FLANK"
                    guard.target = goal.copy()
                    guard.pause_remaining = 0.0
                    guard.suspicion = max(guard.suspicion, 0.78)
        self._dialogue_timer = 7.0
        self.learning_events = getattr(self, "learning_events", [])
        self.learning_events.append({"type": "observed_launch", "device": device.device_id,
                                     "witnesses": [gid for gid, _ in witnesses],
                                     "estimated_source": source.tolist()})

    def _leave_blood_trail(self, old: np.ndarray) -> None:
        speed = float(np.linalg.norm(self.env.player_pos - old)) / self.game_dt
        loss = self.player_vitals.step_bleeding(self.game_dt, speed)
        self._wound_trail.deposit(self.blood, old, self.env.player_pos, loss, self._new_id)
        # Stepping through a dead enemy's blood stains shoes even if the kill
        # happened at range. Check the movement segment so a dive cannot skip it.
        delta = self.env.player_pos - old
        for body in self.bodies:
            if not body.lethal:
                continue
            t = np.clip(np.dot(body.pos - old, delta) / max(float(np.dot(delta, delta)), 1e-7), 0., 1.)
            nearest = old + delta * t
            if np.linalg.norm(nearest - body.pos) <= .85 and line_clear(self.env.grid, nearest, body.pos):
                if self.bloody_timer <= 0:
                    self.last_blood_pos = None
                self.bloody_timer = 22.0
                break
        if self.bloody_timer <= 0.0:
            return
        self.bloody_timer = max(0.0, self.bloody_timer - self.game_dt)
        moved = float(np.linalg.norm(self.env.player_pos - old))
        if moved <= 1e-5:
            return
        if (
            self.last_blood_pos is None
            or float(np.linalg.norm(self.env.player_pos - self.last_blood_pos)) >= 0.52
        ):
            mark = self.env.player_pos.copy()
            self.blood.append(BloodMark(self._new_id(), mark))
            self.last_blood_pos = mark

    def _player_grid(self):
        if self.level != 2 or self.archive_unlocked:
            return self.env.grid
        grid = self.env.grid.copy()
        for x, y in ARCHIVE_DOORS:
            grid[y, x] = 1
        return grid

    def _archive_access(self, inp):
        if self.level != 2 or self.archive_unlocked:
            return
        nearby = any(np.linalg.norm(self.env.player_pos - np.array(p)) <= 1.6 for p in ARCHIVE_DOORS)
        if nearby and inp.interact and not self._interact_latch:
            if self.equipment.security_card:
                self.archive_unlocked = True
                self.message = "ARCHIVIO SBLOCCATO · TESSERA ACCETTATA"
            else:
                self.message = "ARCHIVIO CHIUSO · SERVE LA TESSERA DELLA PRIMA MISSIONE"

    def _apply_player(self, inp: InputState) -> None:
        old = self.env.player_pos.copy()
        self._prepare_condition_input(inp)
        self.player_vitals.step_smoke(smoke_density(old, self.smoke_clouds), self.game_dt)
        self.player_actions.step(inp, old, self.game_dt)
        inp.aim_x = float(old[0] + math.cos(self.player_actions.angle) * 8.)
        inp.aim_y = float(old[1] + math.sin(self.player_actions.angle) * 8.)
        direction = inp.direction()
        self._archive_access(inp)
        player_grid = self._player_grid()

        moving = float(np.linalg.norm(direction)) > 0.0
        compliant = inp.compliant and not moving
        self.env.player_compliant = bool(compliant)

        self.player_actions.interaction(inp, old, self.game_dt, self.equipment, self.player_vitals,
            self.pickups, lambda a, b: line_clear(self.env.grid, a, b), busy=moving or inp.dive or inp.attack_held or self.player_actions.movement_locked,
            near_object=any(self.env.episode_has_intel[i] and not self.env.player_has_intel[i]
                and np.linalg.norm(old - point) <= 1.05 and line_clear(self.env.grid, old, point)
                for i, point in enumerate(self.env._intel)))
        self._player_extra_items(inp)
        if self.player_actions.station_id and self.player_vitals.treatment == 'MEDKIT':
            self.medical_watch.used.add(self.player_actions.station_id)
        self.equipment.medical_input(self.player_vitals, inp,
            busy=moving or inp.dive or self.takedown is not None or inp.fire or inp.choke or inp.lethal
                 or inp.throw or inp.craft or inp.use_flash or inp.use_smoke or inp.use_grenade or inp.use_snack
                 or self.player_actions.movement_locked)
        locked = self.takedown is not None or self.player_actions.movement_locked
        boost = self.player_vitals.boost_remaining > 0
        sprinting = self.player_vitals.step(inp.sprint and moving and not inp.crouch and not compliant and not locked,
                                           self.game_dt, drain=0.22 if boost else 0.16)
        if compliant or locked:
            speed = 0.0
        elif inp.crouch:
            speed = float(self.env.cfg.player_stealth_speed)
        elif sprinting:
            speed = float(self.env.cfg.player_sprint_speed)
            if boost:
                speed *= 1.65
        else:
            speed = float(self.env.cfg.player_walk_speed)
        speed *= self.player_vitals.speed_multiplier

        if self.player_flash_timer > 0:
            speed *= 0.28

        self.env.player_stance = (
            PlayerStance.CROUCHED if inp.crouch else PlayerStance.STANDING
        )
        self.env.player_pos = _move_player_grid(
            self.env.player_pos,
            direction,
            speed,
            lambda x, y: 0 <= x < self.env.cfg.width and 0 <= y < self.env.cfg.height and not player_grid[y, x],
        )
        self.env.player_pos = self.player_actions.dive(inp, self.env.player_pos, player_grid, self.player_vitals)
        if self.player_actions.dive_timer > .34:
            self.noise_events.append(NoiseEvent(self._new_id(), self.env.player_pos.copy(), 12., .9, "DIVE IMPACT"))
        self.env.player_velocity = (
            self.env.player_pos - old
        ) / max(self.env.cfg.dt, 1e-6)
        self.env.player_current_speed = float(
            np.linalg.norm(self.env.player_velocity)
        )

        if inp.interact and not self._interact_latch:
            self._collect_pickups()
            for idx, pedestal in enumerate(self.env._intel):
                if not self.env.episode_has_intel[idx] or self.env.player_has_intel[idx]:
                    continue
                if float(np.linalg.norm(self.env.player_pos - pedestal)) <= 1.05:
                    if not line_clear(player_grid, self.env.player_pos, pedestal):
                        continue
                    if self.level == 2 and idx == 0 and not self.archive_unlocked:
                        self.message = "SERVE LA TESSERA PER APRIRE L’ARCHIVIO"
                        break
                    if self.level == 3 and not (self.equipment.hazmat and self.equipment.sample_vial):
                        self.message = "SERVONO PROTEZIONE E PROVETTA STERILE"
                        break
                    self.env.player_has_intel[idx] = 1
                    if self.level == 1:
                        self.campaign_items.add('SECURITY_CARD')
                        self.equipment.security_card = True
                    if self.level == 3:
                        self.equipment.sample_vial = False
                    self.message = "OBIETTIVO RECUPERATO"
                    break
        self._interact_latch = inp.interact
        self._leave_blood_trail(old)
        used_station = self.player_actions.treatment_step(self.player_vitals, self.pickups, self.game_dt)
        if used_station:
            self.medical_watch.used.add(used_station)
        if self.player_vitals.bleeding <= 0:
            self._wound_trail.pending = 0.0
        self._craft_or_throw(inp)
        self._player_equipment_action(inp)
        self._step_pathogen()
        self._step_explosives()

    def _step_pathogen(self):
        self.pathogen_exposure = False
        if self.level != 3:
            return
        x,y = self.env.player_pos
        contact = 41.5 <= x <= 45.5 and 19.5 <= y <= 24.5
        if contact and not self.equipment.hazmat:
            self.pathogen_exposure = True
            self.player_vitals.health = max(0.,self.player_vitals.health - 4. * self.game_dt)
            self.player_vitals.stamina = 0.
            self.player_vitals.exhausted = True
            self.message = "CONTAMINAZIONE LETALE · PROTEZIONE ASSENTE"

    def _step_diversions(self) -> None:
        for event in self.noise_events:
            event.ttl -= self.game_dt
        self.noise_events = [event for event in self.noise_events if event.ttl > 0.0]

        active: list[ThrownDiversion] = []
        for device in self.diversions:
            if not device.landed:
                device.flight = min(1.0, device.flight + self.game_dt / 0.42)
                eased = 1.0 - (1.0 - device.flight) ** 2
                device.pos = (
                    device.start + (device.target - device.start) * eased
                ).astype(np.float32)
                if device.flight >= 1.0:
                    device.landed = True
                    device.pos = device.target.copy()
                    device.age = 0.0
                    device.next_pulse = 0.0
            else:
                device.age += self.game_dt
                if (
                    device.pulses_left > 0
                    and device.age >= device.next_pulse
                ):
                    self.noise_events.append(
                        NoiseEvent(
                            event_id=self._new_id(),
                            pos=device.pos.copy(),
                            radius=10.5,
                            ttl=0.72,
                            source="NOISEMAKER",
                            device_id=device.device_id,
                        )
                    )
                    device.pulses_left -= 1
                    device.next_pulse += 1.05
            if not device.landed or device.pulses_left > 0 or device.age < 3.2:
                active.append(device)
        self.diversions = active
        self._coordinate_diversions()

    def _step_cameras(self) -> None:
        player = self.env.player_pos
        moving = min(1.0, self.env.player_current_speed / 2.0)
        for camera in self.cameras:
            camera.step(self.game_dt)
            if not camera.online:
                continue
            visible = not self.spawn_protected and can_see(
                self.env.grid,
                camera.pos,
                camera.angle,
                player,
                camera.vision_range,
                camera.fov,
            )
            visible = visible and not self._smoke_blocks(camera.pos, player)
            if visible:
                distance = float(np.linalg.norm(player - camera.pos))
                posture = 0.45 if self.env.player_stance == PlayerStance.CROUCHED else 1.0
                rate = (0.38 + 0.95 * moving) * posture * (1.25 - 0.45 * distance / camera.vision_range)
                camera.confidence = min(1.0, camera.confidence + rate * self.game_dt)
                if distance <= 1.6:
                    camera.confidence = 1.0
                camera.tracking = camera.confidence >= 0.55
                if camera.confidence >= 1.0:
                    self._publish_contact(player, camera.camera_id)
            else:
                camera.confidence = max(0.0, camera.confidence - 0.32 * self.game_dt)
                camera.tracking = camera.confidence >= 0.55

    def _coordinate_diversions(self) -> None:
        if self.spawn_protected:
            return
        observers = [(g.guard_id, g.pos) for g in self.aux_guards
                     if g.active and g.state not in {"CHASE", "VERIFY", "RESPOND"}]
        if self._primary_active() and not self.env.sense.visual:
            observers.append(("G0", self.env.guard_pos))
        for event in self.noise_events:
            if event.source != "NOISEMAKER":
                continue
            key = event.device_id if event.device_id is not None else event.event_id
            if key in self._diversion_assignments:
                continue
            heard = sorted(((float(np.linalg.norm(pos - event.pos)), gid, pos)
                            for gid, pos in observers if np.linalg.norm(pos - event.pos) <= event.radius),
                           key=lambda item: (item[0], item[1]))
            if not heard:
                continue
            investigator = heard[0][1]
            self._diversion_counts[investigator] = self._diversion_counts.get(investigator, 0) + 1
            repeated = self._diversion_counts[investigator] >= 2
            if repeated:
                self.security_compromised = True
                self.guard_dialogue[investigator] = "C'è qualcosa di strano. Vieni a coprirmi."
            elif len(heard) >= 2:
                self.guard_dialogue[investigator] = (
                    "Controlliamo insieme. Coprimi sul fianco." if self.security_compromised
                    else "Hai sentito? Vado a controllare, tu resta di guardia.")
            else:
                self.guard_dialogue[investigator] = "Ho sentito qualcosa. Vado a controllare."
            assignments = {investigator: event.pos.copy()}
            partner = heard[1] if len(heard) >= 2 else None
            if repeated and partner is None:
                candidates = [(float(np.linalg.norm(pos - heard[0][2])), gid, pos)
                              for gid, pos in observers if gid != investigator]
                partner = min(candidates, default=None, key=lambda item: item[0])
            if self.security_compromised and partner is not None:
                direction = event.pos - heard[0][2]
                if np.linalg.norm(direction) < 1e-6:
                    direction = np.array([1.0, 0.0], dtype=np.float32)
                direction /= max(float(np.linalg.norm(direction)), 1e-6)
                side = np.array([-direction[1], direction[0]], dtype=np.float32)
                assignments[partner[1]] = self._navigable_goal(event.pos - direction * 1.5 + side * 1.5)
                self.guard_dialogue[partner[1]] = "Ricevuto. Ti copro."
            self.learning_events = getattr(self, "learning_events", [])
            self.learning_events.append({"type": "diversion", "device": key,
                                         "heard_by": [gid for _, gid, _ in heard],
                                         "assigned": list(assignments), "repeated": repeated,
                                         "high_guard": self.security_compromised})
            self._diversion_assignments[key] = assignments
            self._dialogue_timer = 7.0
            self._local_diversion_guards.update(assignments)
            self._searches.pop(investigator, None)

    def _audible_cue(self, observer: np.ndarray, observer_id: str = "G0") -> Optional[tuple[np.ndarray, str]]:
        if self.spawn_protected:
            return None
        heard: list[tuple[float, np.ndarray, str]] = []
        for event in self.noise_events:
            distance = float(np.linalg.norm(event.pos - observer))
            key = event.device_id if event.device_id is not None else event.event_id
            assigned = event.source == "NOISEMAKER" and observer_id in self._diversion_assignments.get(key, {})
            if event.pulse and (distance <= event.radius or assigned):
                if event.source == "NOISEMAKER":
                    key = event.device_id if event.device_id is not None else event.event_id
                    goal = self._diversion_assignments.get(key, {}).get(observer_id)
                    if goal is None:
                        continue
                    heard.append((distance, goal, event.source))
                else:
                    heard.append((distance, event.pos, event.source))

        if self.player_vitals.panting:
            distance = float(np.linalg.norm(self.env.player_pos - observer))
            if distance <= 5.0:
                heard.append((distance, self.env.player_pos, "HEAVY BREATHING"))
        if self.player_vitals.injured and np.linalg.norm(self.env.player_pos - observer) <= 5. and self.env.player_current_speed > .05:
            heard.append((float(np.linalg.norm(self.env.player_pos - observer)), self.env.player_pos, "LIMPING"))
        speed = self.env.player_current_speed
        if speed > 0.05 and not self.spawn_protected:
            if self.env.player_stance == PlayerStance.CROUCHED:
                radius = 1.5
            elif speed >= 1.9:
                radius = 7.5
            else:
                radius = 4.2
            distance = float(np.linalg.norm(self.env.player_pos - observer))
            if distance <= radius:
                heard.append((distance, self.env.player_pos, "FOOTSTEPS"))
        if not heard:
            return None
        distance, pos, source = min(heard, key=lambda item: item[0])
        sigma = 0.12 + min(1.2, distance * 0.08)
        noisy = pos + self.rng.normal(0.0, sigma, size=2).astype(np.float32)
        x, y = self.env._cell(noisy)
        if not self.env._free(x, y) or not line_clear(self.env.grid, pos, noisy):
            noisy = pos.copy()
        return noisy, source

    def _start_guard_search(self, guard: SecurityGuard, anchor: np.ndarray) -> None:
        anchor = self._navigable_goal(anchor)
        guard.state = "SEARCH"
        guard.last_seen = anchor.copy()
        guard.search_step = 0
        self._searches[guard.guard_id] = LocalSearch(
            local_search_points(self.env.grid, anchor, int(guard.guard_id[1:])))
        self._search_reports[guard.guard_id] = dict(point=anchor.copy(), reported=False)
        guard.target = anchor.copy()
        self._request_support(guard.guard_id, anchor)

    def _report_empty_target(self, guard):
        report = self._search_reports.get(guard.guard_id)
        if report is None or report['reported'] or guard.seeing_player:
            return
        report['reported'] = True
        if self._player_identified:
            message = "Radio: punto vuoto, bersaglio perso. Possibile furto: presidiamo intel e uscite."
        else:
            message = "Radio: punto vuoto, nessun intruso identificato. Controlliamo gli intel."
        self.guard_dialogue[guard.guard_id] = message
        self._dialogue_timer = 7.
        self.radio_updates += 1

    def _post_search_order(self, guard):
        self._report_empty_target(guard)
        if self._player_identified:
            self._guard_goal_from_radio(guard)
            return
        indices = [i for i in self.env.episode_intel_order if self.env.known_intel_present[i]]
        if indices:
            offset = int(guard.guard_id[1:]) % len(indices)
            indices = indices[offset:] + indices[:offset]
            self._intel_sweeps[guard.guard_id] = indices
            guard.state = "CHECK INTEL"
            guard.target = self.env._intel[indices[0]].copy()
        else:
            guard.state, guard.target = "PATROL", guard.patrol_target()

    def _step_casualty_reinforcements(self):
        pending = self._combat_casualties - self._reinforced_casualties
        if not pending or self.team_contact is None or self.team_contact_age <= 12.:
            return
        guards = self._active_guards()
        searching = {"CHASE", "REPOSITION", "RESPOND", "SEARCH", "VERIFY", "STACK", "CLEAR ROOM", "SUPPORT", "TRACK BLOOD", "INSPECT BODY", "INVESTIGATE", "FLANK"}
        if self.radio_timer is not None or any(g.seeing_player or g.state in searching for g in guards):
            return
        # A fresh squad is requested as soon as the unsuccessful search ends.
        # Departing members can coexist with the replacement squad.
        self._reinforced_casualties.update(pending)
        self._guard_dispatch = None
        self.radio_source = "COMBAT LOSSES"
        self.swat_phase, self.radio_timer = "DISPATCH", 1.0
        self.guard_dialogue['HQ'] = "Radio: ricerca conclusa, bersaglio non trovato e guardie cadute. Inviate rinforzi SWAT."
        self._dialogue_timer = 8.
        self.radio_updates += 1

    def _detect_evidence(self, guard: SecurityGuard) -> bool:
        if guard.blinded_timer > 0:
            return False
        for body in self.bodies:
            if body.discovered:
                continue
            if can_see(
                self.env.grid,
                guard.pos,
                guard.angle,
                body.pos,
                8.0,
                115.0,
            ):
                body.discovered = True
                guard.state = "INSPECT BODY"
                guard.target = body.pos.copy()
                guard.suspicion = 1.0
                self._raise_evidence_alert("BODY FOUND", body.pos, guard.guard_id)
                self.radio_updates += 1
                return True

        mark = observe_blood(self.blood, self.env.grid, guard.pos, guard.angle,
                             guard.guard_id, guard.evidence_id, self._smoke_blocks)
        if mark is not None:
            guard.evidence_id = mark.mark_id
            guard.state = "TRACK BLOOD"
            guard.target = mark.pos.copy()
            guard.suspicion = max(guard.suspicion, 0.85)
            self._raise_evidence_alert("BLOOD TRAIL")
            return True
        return False

    def _guard_goal_from_radio(self, guard: SecurityGuard) -> None:
        if self.team_contact is None:
            return
        # Split security between remaining objectives and the extraction route.
        points = [p for i, p in enumerate(self.env._intel)
                  if self.env.episode_has_intel[i] and self.env.known_intel_present[i]]
        points.append(self.env._exit)
        index = max(0, int(guard.guard_id[1:]) - 1) % len(points)
        if guard.target is None or not np.array_equal(guard.target, points[index]):
            guard.pause_angle = guard.angle
        guard.target = points[index].copy()
        guard.state = "SECURE EXFIL" if index == len(points) - 1 else "PROTECT INTEL"
        guard.last_seen = self.team_contact.copy()
        guard.last_seen_age = self.team_contact_age

    def _inspect_guard_intel(self, guard: SecurityGuard) -> None:
        if not guard.active or guard.blinded_timer > 0 or guard.vitals.hit_stun_remaining > 0:
            return
        for index, point in enumerate(self.env._intel):
            if (not self.env.episode_has_intel[index] or not self.env.player_has_intel[index]
                    or index in self._intel_checks or index in self._reported_missing_intel):
                continue
            if (np.linalg.norm(guard.pos - point) > 1.15
                    or not can_see(self.env.grid, guard.pos, guard.angle, point, 1.15, GUARD_FOV)
                    or self._smoke_blocks(guard.pos, point)):
                continue
            mission = MISSIONS[self.level]
            name = mission['names'][mission['indices'].index(index)]
            self._intel_checks[index] = {"source": guard.guard_id, "remaining": 1.2, "name": name}
            self.guard_dialogue[guard.guard_id] = f"{name} assente. Qualcuno di voi l'ha preso?"
            self._dialogue_timer = 5.
            self.radio_updates += 1

    def _step_intel_radio(self) -> None:
        guards = {guard.guard_id: guard for guard in self._active_guards()}
        for index, call in list(self._intel_checks.items()):
            source = guards.get(call['source'])
            if source is None:
                del self._intel_checks[index]
                continue
            if source.vitals.hit_stun_remaining > 0:
                continue
            call['remaining'] -= self.game_dt
            if call['remaining'] > 1e-6:
                continue
            responders = [guard for guard in guards.values() if guard is not source
                          and guard.vitals.hit_stun_remaining <= 0]
            if responders:
                for responder in responders:
                    self.guard_dialogue[responder.guard_id] = "Negativo, non l'ho preso."
                self.guard_dialogue[source.guard_id] = f"{call['name']} scomparso. Tutti in cautela."
            else:
                self.guard_dialogue[source.guard_id] = "Nessuna conferma di recupero. Intel assente, cautela."
            self._reported_missing_intel.add(index)
            self.env.known_intel_present[index] = 0
            self.security_compromised = True
            self.threat_level = max(self.threat_level, 1)
            self.env.alert = max(int(self.env.alert), int(FacilityAlert.SUSPICIOUS))
            self.env.personal_suspicion = max(self.env.personal_suspicion, .48)
            for guard in guards.values():
                guard.suspicion = max(guard.suspicion, .48)
            if not self.team_contact_source:
                self.team_contact_source = "MISSING INTEL"
            self._dialogue_timer = 7.
            self.radio_updates += 1
            del self._intel_checks[index]
        self._update_facility_alert()

    def _register_spot(self, source: str, confidence: float) -> None:
        if confidence < 0.15:
            self._spot_active.discard(source)
        elif confidence >= 0.8 and source not in self._spot_active:
            self._spot_active.add(source)
            self._strong_spots[source] = self._strong_spots.get(source, 0) + 1

    def _request_support(self, source: str, point: np.ndarray) -> None:
        tension = self.security_compromised or self._strong_spots.get(source, 0) >= 2
        if self.spawn_protected or source.startswith("S") or not tension:
            return
        if source in self._support_calls or self._support_cooldowns.get(source, 0.0) > 0:
            return
        self._support_calls[source] = {"remaining": 1.2, "point": self._navigable_goal(point)}
        self.guard_dialogue[source] = "Radio: movimento sospetto, chiedo supporto."
        self._dialogue_timer = 5.0

    def _step_support_radio(self) -> None:
        for source in self._support_cooldowns:
            self._support_cooldowns[source] = max(0.0, self._support_cooldowns[source] - self.game_dt)
        for source, call in list(self._support_calls.items()):
            active = self._primary_active() if source == "G0" else any(
                g.guard_id == source and g.active for g in self.aux_guards)
            if not active:
                del self._support_calls[source]
                continue
            call["remaining"] -= self.game_dt
            if call["remaining"] > 1e-6:
                continue
            if call.get("confirmed"):
                candidates = [g for g in self._active_guards() if g.guard_id == call["helper"]]
            else:
                candidates = [g for g in self.aux_guards if g.active and g.guard_id != source
                          and g.state in {"PATROL", "OBSERVE", "PROTECT INTEL", "SECURE EXFIL"} and g.visual_confidence < 0.25]
            if candidates:
                buddy = min(candidates, key=lambda g: float(np.linalg.norm(g.pos - call["point"])))
                if buddy.state != "CHASE":
                    buddy.state = "SUPPORT"
                    buddy.target = call["point"].copy()
                    buddy.last_seen, buddy.last_seen_age = call["point"].copy(), 0.0
                buddy.suspicion = max(buddy.suspicion, 0.25)
                self.guard_dialogue[buddy.guard_id] = "Ricevuto, arrivo a coprirti."
                if buddy.guard_id == "G0":
                    self.env.last_seen_pos = call["point"].copy()
                    self.env.last_seen_age = 0.0
                    self.env.intruder_known = bool(call.get("confirmed")) or self.env.intruder_known
                self.radio_updates += 1
            elif call.get("confirmed"):
                self._contact_helpers.pop(source, None)
                self._start_swat_response(source)
            self._support_cooldowns[source] = 12.0
            del self._support_calls[source]

    def _step_guard_dispatch(self) -> None:
        if self.radio_timer is not None or self.swat_units:
            self._guard_dispatch = None
            return
        if self._guard_dispatch is None or self.spawn_protected:
            return
        self._guard_dispatch["remaining"] -= self.game_dt
        if self._guard_dispatch["remaining"] > 1e-6:
            return
        point = self._guard_dispatch["point"]
        next_id = max([int(g.guard_id[1:]) for g in self.aux_guards] + [0]) + 1
        for i in range(2):
            guard = SecurityGuard(f"G{next_id+i}", np.array([46., 10.+i], dtype=np.float32),
                                  math.pi, ((46, 10), (40, 10), (40, 19), (46, 19)))
            guard.state = "SUPPORT"
            direction = np.array([0, 1.0 if i else -1.0], dtype=np.float32)
            guard.target = self._navigable_goal(point + direction * 1.4)
            guard.suspicion = 0.85
            self.aux_guards.append(guard)
            self.guard_dialogue[guard.guard_id] = "Pattuglia di rinforzo, copriamo il settore."
        self._regular_reinforcements += 2
        self.radio_updates += 1
        self._guard_dispatch = None

    def _guard_move(self, guard, goal, speed, turn_rate=2.0):
        if guard.vitals.hit_stun_remaining > 0:
            return guard.pos.copy(), guard.angle
        moving = goal is not None and np.linalg.norm(goal - guard.pos) > 0.2
        running = moving and (self.facility_alert != "CALM" or self.security_compromised
                              or guard.state in {"CHASE", "RESPOND", "FLANK", "SUPPORT", "CLEAR ROOM", "REPOSITION"})
        if guard.guard_id.startswith("S") and moving and guard.state == "SEARCH":
            running = True
        if running:
            patrol_speed = float(self.env.cfg.guard_speed) / self.game_dt
            if guard.guard_id.startswith("S"):
                speed = patrol_speed * 2.8 if guard.vitals.can_sprint else patrol_speed
            else:
                speed = max(speed, patrol_speed * 2.4) if guard.vitals.can_sprint else min(speed, patrol_speed)
        speed *= guard.vitals.speed_multiplier
        if guard.blinded_timer > 0:
            speed *= .35
        if guard.state == "REPOSITION" and goal is not None:
            pos = self.navigator.move(guard.guard_id, guard.pos, goal, speed * self.game_dt)
            pos = self._motion_recovery.step(self.navigator, guard.guard_id, guard.pos, pos,
                goal, speed, self.game_dt, True)
            target = guard.last_seen if guard.last_seen is not None else goal
            heading = math.atan2(float(target[1] - pos[1]), float(target[0] - pos[0]))
            guard.vitals.step(running and np.linalg.norm(pos - guard.pos) > 1e-6,
                              self.game_dt, drain=.14, fatigue_on_pause=False)
            return pos, turn_towards(guard.angle, heading, 4.0 * self.game_dt)
        motion = self._motions.setdefault(guard.guard_id, HumanMotion())
        moved, angle = motion.move(self.navigator, guard.guard_id, guard.pos, guard.angle,
                                   goal, speed, self.game_dt, turn_rate)
        moved = self._motion_recovery.step(self.navigator, guard.guard_id, guard.pos, moved, goal,
            speed, self.game_dt, goal is not None)
        guard.vitals.step(running and np.linalg.norm(moved - guard.pos) > 1e-6,
                          self.game_dt, drain=.14, fatigue_on_pause=False)
        return moved, angle

    def _step_guard_weapon(self, guard: SecurityGuard, confirmed: bool) -> None:
        guard.shot_cooldown = max(0.0, guard.shot_cooldown - self.game_dt)
        visible = confirmed and can_see(self.env.grid, guard.pos, guard.angle, self.env.player_pos,
                                       9.0 * (.85 if guard.has_shield else 1.), GUARD_FOV - (20. if guard.has_shield else 0.)) and not self._smoke_blocks(guard.pos, self.env.player_pos)
        recovering = self.player_actions.knockdown_remaining > 0 or self.player_actions.rise_timer > 0
        close_armed = (self.equipment.armed or recovering) and np.linalg.norm(guard.pos - self.env.player_pos) <= MELEE_RANGE
        melee = self._melee_attacks.setdefault(guard.guard_id, EnemyMelee())
        profile = MELEE_PROFILES['SHIELD' if guard.has_shield else 'SWAT' if guard.guard_id.startswith('S') else 'GUARD']
        ready = (guard.active and visible and not self.spawn_protected and not self.env.player_compliant
                 and guard.blinded_timer <= 0 and guard.state != "ESCAPE SMOKE"
                 and not (self.takedown and self.takedown['guard_id'] == guard.guard_id))
        hit = melee.step(ready and close_armed and self.player_actions.knockdown_remaining <= 0
                         and self.player_actions.rise_timer <= 0 and self.player_actions.dive_timer <= 0,
                         guard.vitals, self.game_dt, profile)
        if ready and close_armed:
            guard.aim_timer = 0.
            guard.fire_goal = None
            if hit:
                self._melee_impact(guard.guard_id, guard, profile)
            return
        if not guard.active or not visible or not guard.vitals.can_aim or guard.state == "ESCAPE SMOKE" or self.spawn_protected or self.env.player_compliant or guard.blinded_timer > 0 or guard.disarmed:
            guard.aim_timer = 0.0
            return
        allies = [g.pos for g in self._active_guards() if g.guard_id != guard.guard_id]
        if lane_blocked(guard.pos, self.env.player_pos, allies):
            guard.aim_timer = 0.0
            guard.fire_goal = firing_position(self.navigator, guard.pos, self.env.player_pos, allies,
                                               int(guard.guard_id[1:]))
            return
        guard.fire_goal = None
        if self._enemy_grenade(guard.guard_id, guard, guard.vitals):
            guard.shot_cooldown = 1.5
            return
        guard.aim_timer += self.game_dt
        aim_delay = 0.35 if self.threat_level >= 2 else 0.95
        if guard.aim_timer >= aim_delay and guard.shot_cooldown <= 0:
            if not self._enemy_shot(guard.guard_id, guard, guard.vitals):
                guard.aim_timer = 0.
                return
            guard.shot_cooldown = WEAPONS[guard.weapon].cooldown

    def _smoke_blocks(self, a: np.ndarray, b: np.ndarray) -> bool:
        return smoke_obscures(a, b, self.smoke_clouds)

    def _escape_guard_smoke(self, guard):
        if not smoke_density(guard.pos, self.smoke_clouds):
            guard.smoke_goal = None
            return False
        goal = getattr(guard, "smoke_goal", None)
        if goal is None or smoke_density(goal, self.smoke_clouds):
            goal = smoke_escape(self.navigator, guard.pos, self.smoke_clouds)
            guard.smoke_goal = goal
        guard.aim_timer = 0.
        guard.state = "ESCAPE SMOKE"
        guard.pos, guard.angle = self._guard_move(guard, goal,
            float(self.env.cfg.guard_speed) / self.game_dt, 4.)
        return True

    def _sense_primary(self):
        # Apply smoke to perception without changing collision/pathfinding.
        wall_clear = self.env._line_clear
        spot_rate = self.env._spot_rate
        self.env._line_clear = lambda a, b: wall_clear(a, b) and not self._smoke_blocks(a, b) and self._lead_patrol.blinded_timer <= 0
        self.env._spot_rate = lambda *args: spot_rate(*args) * (1.6 if self.threat_level >= 2 else 1.0)
        try:
            return self.env._sense()
        finally:
            self.env._line_clear = wall_clear
            self.env._spot_rate = spot_rate

    def _reported_room(self, point: np.ndarray):
        # Enclosures in build_soliton_grid; use radio memory, never player truth.
        for bounds in mission_room_bounds(self.level):
            x0, y0, x1, y1 = bounds
            if x0 < point[0] < x1 and y0 < point[1] < y1:
                return bounds
        return None

    def _step_tactical_entry(self, guard: SecurityGuard) -> bool:
        room = self._reported_room(self.team_contact) if self.team_contact is not None else None
        if room is None or guard.state in {"CHASE", "WITHDRAW", "LEAVING", "EXITED"} or not guard.guard_id.startswith("S"):
            return False
        if guard.guard_id in self._room_entries and self._room_entries[guard.guard_id]["room"] != room:
            self._room_entries.pop(guard.guard_id)
        plan = self._room_entries.get(guard.guard_id)
        # Finish an ongoing entry, but do not open a new one on an expired report.
        if plan is None:
            x0, y0, x1, y1 = room
            doors = [(np.array([x,y],dtype=np.float32), np.array(normal,dtype=np.float32))
                     for x,y,normal in
                     ([(x,y0,(0,1)) for x in range(x0+1,x1)] +
                      [(x,y1,(0,-1)) for x in range(x0+1,x1)] +
                      [(x0,y,(1,0)) for y in range(y0+1,y1)] +
                      [(x1,y,(-1,0)) for y in range(y0+1,y1)])
                     if self.env._free(x,y)]
            if not doors:
                return False
            teammates = [p for p in self._room_entries.values() if p["room"] == room]
            if teammates:
                door, inward = teammates[0]['door'].copy(), teammates[0]['inward'].copy()
            else:
                door, inward = min(doors, key=lambda p: self.navigator.path_length(guard.pos, p[0] - p[1]))
            index = int(guard.guard_id[1:]) % 3
            stage = self._navigable_goal(door - inward * (1 + index * .6))
            corners = [np.array(p, dtype=np.float32) for p in
                       ((x0+1, y0+1), (x1-1, y0+1), (x1-1, y1-1), (x0+1, y1-1))]
            corners = corners[index:] + corners[:index]
            plan = {"room": room, "stage": stage, "door": door, "inward": inward,
                    "search": LocalSearch([door.copy()] + corners, scan_seconds=0.85),
                    "phase": "STACK", "wait": 0.6}
            self._room_entries[guard.guard_id] = plan
        if plan["phase"] == "DONE":
            return False
        guard.state = plan["phase"]
        goal, look = plan["stage"], None
        if plan["phase"] == "STACK" and np.linalg.norm(guard.pos - goal) < 0.25:
            ready = all(
                unit.guard_id in self._room_entries
                and self._room_entries[unit.guard_id]["room"] == room
                and (self._room_entries[unit.guard_id]["phase"] != "STACK"
                     or np.linalg.norm(unit.pos - self._room_entries[unit.guard_id]["stage"]) < 0.25)
                for _, unit in self.swat_units if unit.active and unit.state not in {"CHASE", "WITHDRAW", "LEAVING", "EXITED"})
            if ready:
                plan["wait"] -= self.game_dt
            look = math.atan2(float(self.team_contact[1] - guard.pos[1]), float(self.team_contact[0] - guard.pos[0]))
            if plan["wait"] <= 0:
                # The squad breaches once, then enters after the device fires.
                squad = [unit for _, unit in self.swat_units if unit.active
                         and unit.state not in {"CHASE", "WITHDRAW", "LEAVING", "EXITED"}]
                plans = [self._room_entries[u.guard_id] for u in squad]
                inside = plan["door"] + plan['inward']
                blast = self.team_contact if line_clear(self.env.grid, inside, self.team_contact) else inside
                thrower = next((u for u in squad if not u.has_shield and u.flash_count and not u.disarmed), None)
                delay = .25
                if thrower is not None:
                    thrower.flash_count -= 1
                    self.flash_events.append({"pos": blast.copy(), "ttl": 0.45})
                    if (np.linalg.norm(self.env.player_pos - blast) <= 3.0
                            and line_clear(self.env.grid, blast, self.env.player_pos)):
                        self.player_flash_timer = 2.4
                else:
                    thrower = next((u for u in squad if not u.has_shield and u.grenade_count
                                    and not u.disarmed and u.vitals.can_aim), None)
                    if thrower is not None and all(np.linalg.norm(u.pos - blast) >= 3.5 for u in squad):
                        thrower.grenade_count -= 1
                        self._throw_grenade(inside, blast, thrower.guard_id)
                        delay = 1.6
                for entry in plans:
                    entry['phase'], entry['wait'] = 'BREACH WAIT', delay
        elif plan['phase'] == 'BREACH WAIT':
            plan['wait'] -= self.game_dt
            if plan['wait'] <= 1e-6:
                plan['phase'] = 'CLEAR ROOM'
        elif plan["phase"] == "CLEAR ROOM":
            goal, look, done = plan["search"].step(guard.pos, guard.angle, self.game_dt)
            if done:
                plan["phase"] = "DONE"
                self._start_guard_search(guard, self.team_contact)
                self._report_empty_target(guard)
        guard.target = goal
        guard.pos, guard.angle = self._guard_move(guard, goal, float(self.env.cfg.guard_speed) / self.game_dt * 1.65, 4.0)
        if look is not None:
            guard.angle = turn_towards(guard.angle, look, 4.0 * self.game_dt)
        return True

    def _step_guard(self, guard: SecurityGuard, policy_action=None) -> None:
        guard.seeing_player = False
        if not guard.active:
            return
        if guard.vitals.hit_stun_remaining > 0:
            guard.aim_timer = 0.
            guard.vitals.step(False, self.game_dt, fatigue_on_pause=False)
            return
        if self.takedown and self.takedown["guard_id"] == guard.guard_id:
            return
        if self._escape_guard_smoke(guard):
            return
        if guard.state == "ESCAPE SMOKE":
            guard.state = "SEARCH" if guard.last_seen is not None else "PATROL"
        if guard.state == "LEAVING":
            guard.angle = turn_towards(guard.angle, 0.0, 2.0 * self.game_dt)
            guard.pos[0] += float(self.env.cfg.guard_speed) * max(0.0, math.cos(guard.angle))
            if guard.pos[0] >= self.env.cfg.width:
                guard.state = "EXITED"
            return

        guard.radio_cooldown = max(0.0, guard.radio_cooldown - self.game_dt)
        guard.blinded_timer = max(0.0, guard.blinded_timer - self.game_dt)
        if guard.state in {"PATROL", "OBSERVE"}:
            guard.suspicion = max(0.0, guard.suspicion - 0.04 * self.game_dt)
        guard.last_seen_age += self.game_dt
        player = self.env.player_pos
        distance = float(np.linalg.norm(player - guard.pos))
        visible = not self.spawn_protected and can_see(
            self.env.grid,
            guard.pos,
            guard.angle,
            player,
            GUARD_VISION_RANGE * (.85 if guard.has_shield else 1.),
            GUARD_FOV - (20. if guard.has_shield else 0.),
            self.env.cfg.vision_immediate,
        )
        visible = visible and guard.blinded_timer <= 0 and not self._smoke_blocks(guard.pos, player)
        guard.seeing_player = visible
        if visible:
            motion = min(1.0, self.env.player_current_speed / 2.5)
            posture = 0.48 if self.env.player_stance == PlayerStance.CROUCHED else 1.0
            alert_mult = 2.4 if self.threat_level >= 2 else 1.55 if self.facility_alert != "CALM" else 1.0
            rate = (0.42 + 1.15 * motion) * posture * alert_mult * (1. + .15 * (self.level - 1))
            guard.visual_confidence = min(
                1.0,
                guard.visual_confidence + rate * self.game_dt,
            )
            if distance <= 1.35:
                guard.visual_confidence = 1.0
            self._register_spot(guard.guard_id, guard.visual_confidence)
            if guard.visual_confidence >= 1.0:
                guard.pause_remaining = 0.0
                self._searches.pop(guard.guard_id, None)
                guard.state = "CHASE"
                guard.target = player.copy()
                guard.last_seen = player.copy()
                guard.last_seen_age = 0.0
                guard.suspicion = 1.0
                self._publish_contact(player, guard.guard_id)
            elif guard.visual_confidence >= 0.25:
                guard.pause_remaining = 0.0
                if guard.state != "VERIFY" or guard.target is None or np.linalg.norm(guard.target - player) > 1.5:
                    uncertainty = (1.0 - guard.visual_confidence) * 1.2
                    guard.target = self._navigable_goal(player + self.rng.normal(
                        0.0, uncertainty, size=2
                    ).astype(np.float32))
                guard.state = "VERIFY"
                guard.suspicion = max(guard.suspicion, guard.visual_confidence)
                self._request_support(guard.guard_id, guard.target)
        else:
            guard.visual_confidence = max(
                0.0,
                guard.visual_confidence - 0.24 * self.game_dt,
            )
            self._register_spot(guard.guard_id, guard.visual_confidence)
            if guard.state in {"CHASE", "REPOSITION"} and guard.last_seen is not None:
                guard.fire_goal = None
                self._start_guard_search(guard, guard.last_seen)
            elif guard.state == "VERIFY" and guard.target is not None and np.linalg.norm(guard.pos - guard.target) < .35:
                self._start_guard_search(guard, guard.target.copy())

        if guard.visual_confidence < 1.0:
            cue = self._audible_cue(guard.pos, guard.guard_id)
            if cue is not None and guard.guard_id not in self._launch_responses and guard.state not in {"CHASE", "RESPOND", "VERIFY", "PROTECT INTEL", "SECURE EXFIL"}:
                if cue[1] != "NOISEMAKER":
                    self._local_diversion_guards.discard(guard.guard_id)
                previous_state = guard.state
                guard.pause_remaining = 0.0
                guard.state = "INVESTIGATE"
                if previous_state != "INVESTIGATE" or guard.target is None or np.linalg.norm(guard.target - cue[0]) > 1.0:
                    guard.target = cue[0]
                guard.suspicion = max(guard.suspicion, 0.48)
                if cue[1] == "BODY IMPACT":
                    guard.suspicion = max(guard.suspicion, 0.78)
                elif cue[1] == "GUNSHOT":
                    self._raise_evidence_alert("GUNFIRE")
                    self._request_support(guard.guard_id, guard.target)

        response = self._launch_responses.get(guard.guard_id)
        if response is not None:
            if guard.visual_confidence >= 0.25:
                self._launch_responses.pop(guard.guard_id, None)
            else:
                guard.state = "FLANK"
                guard.target = response["goal"].copy()

        if guard.state not in {"CHASE", "VERIFY"}:
            self._detect_evidence(guard)
            self._inspect_guard_intel(guard)

        if (
            self.team_contact is not None
            and guard.guard_id.startswith("G")
            and self.swat_phase in {"DISPATCH", "ACTIVE"}
            and guard.state in {"PATROL", "OBSERVE", "PROTECT INTEL", "SECURE EXFIL"}
            and guard.visual_confidence < 1.0
        ):
            self._guard_goal_from_radio(guard)

        # Every guard can execute the same intents. A policy chooses among
        # them; current perception and safety still override stale decisions.
        if policy_action is not None:
            self._apply_guard_intent(guard, policy_action, visible)

        medical_goal = self.medical_watch.goal(guard.guard_id, guard.pos, visible,
                                               self.navigator.path_length)
        medical_evidence = self.medical_watch.inspect(guard.guard_id,
            lambda p: can_see(self.env.grid,guard.pos,guard.angle,p,GUARD_VISION_RANGE,GUARD_FOV)
            and not self._smoke_blocks(guard.pos,p) and guard.blinded_timer <= 0)
        if medical_evidence is not None:
            self._raise_evidence_alert('MEDKIT USED', medical_evidence, guard.guard_id)
            self._start_guard_search(guard, medical_evidence)
            self.guard_dialogue[guard.guard_id] = "Ha usato il medikit. È passato da qui!"
            self._dialogue_timer = 7.
            medical_goal = None
        elif medical_goal is not None:
            guard.state = 'WATCH MEDKIT'
            guard.target = medical_goal

        if medical_goal is None and self._step_tactical_entry(guard):
            return
        if visible and guard.visual_confidence >= 1.0 and guard.fire_goal is not None:
            allies = [g.pos for g in self._active_guards() if g.guard_id != guard.guard_id]
            if lane_blocked(guard.pos, player, allies):
                guard.target = guard.fire_goal.copy()
                guard.state = "REPOSITION"
            else:
                guard.fire_goal = None
        if (guard.guard_id.startswith("S") and guard.state not in {"CHASE", "WATCH MEDKIT"}
                and self.team_contact is not None and self.team_contact_age > 0.5):
            trackers = [g for g in self.aux_guards if g.active and g.state in {"CHASE", "SEARCH"}
                        and g.last_seen is not None and g.last_seen_age < 8.0]
            if trackers:
                tracker = min(trackers, key=lambda g: g.last_seen_age)
                # Follow the searching guard's route until a new visual report.
                guide = tracker.target if tracker.target is not None else tracker.pos
                if guard.target is None or np.linalg.norm(guide - guard.target) > 1.0:
                    self._searches.pop(guard.guard_id, None)
                    guard.target = guide.copy()
                    guard.state = "RESPOND"
        scan_angle = None
        if guard.state == 'WATCH MEDKIT' and np.linalg.norm(guard.pos - guard.target) < 1.8:
            scan_angle = math.atan2(float(guard.target[1]-guard.pos[1]),float(guard.target[0]-guard.pos[0]))
            guard.target = guard.pos.copy()
        elif guard.state in {"PATROL", "OBSERVE"}:
            guard.target, scan_angle = self._patrol_step(guard)
        elif guard.state in {"PROTECT INTEL", "SECURE EXFIL"} and np.linalg.norm(guard.target - guard.pos) < 0.25:
            scan_angle = guard.pause_angle + 1.15 * math.sin(self.env.steps * self.game_dt * 0.8)
        elif guard.state == "SEARCH":
            if guard.guard_id not in self._searches:
                self._start_guard_search(guard, guard.last_seen if guard.last_seen is not None else guard.pos)
            search = self._searches[guard.guard_id]
            report = self._search_reports.get(guard.guard_id)
            if report is not None and np.linalg.norm(guard.pos - report['point']) <= 1.25:
                self._report_empty_target(guard)
            guard.target, scan_angle, done = search.step(guard.pos, guard.angle, self.game_dt)
            guard.search_step = search.index
            if done:
                self._searches.pop(guard.guard_id, None)
                if guard.guard_id.startswith("S"):
                    self._report_empty_target(guard)
                    guard.state = "WITHDRAW"
                    guard.target = guard.exit_pos if guard.exit_pos is not None else np.array([self.env.cfg.width - 1.0, 10.0], dtype=np.float32)
                else:
                    guard.suspicion = max(0.48 if self.security_compromised else 0.0, guard.suspicion * 0.5)
                    self._post_search_order(guard)
        elif guard.state == "CHECK INTEL" and np.linalg.norm(guard.pos - guard.target) < .25:
            indices = self._intel_sweeps.get(guard.guard_id, [])
            if indices:
                indices.pop(0)
            if indices:
                guard.target = self.env._intel[indices[0]].copy()
            else:
                self._intel_sweeps.pop(guard.guard_id, None)
                guard.state, guard.target = "PATROL", guard.patrol_target()
        elif guard.state == "TRACK BLOOD":
            mark = observe_blood(self.blood, self.env.grid, guard.pos, guard.angle,
                                 guard.guard_id, guard.evidence_id, self._smoke_blocks)
            if mark is not None:
                guard.evidence_id = mark.mark_id
                guard.target = mark.pos.copy()

        speed = float(self.env.cfg.guard_speed)
        if guard.guard_id.startswith("S"):
            speed *= 1.7
        if guard.state in {"CHASE", "RESPOND", "FLANK", "SUPPORT"}:
            speed *= 1.16
        elif guard.state in {"VERIFY", "SEARCH", "TRACK BLOOD", "INSPECT BODY"}:
            speed *= 0.82
        if guard.guard_id in self._launch_responses:
            speed = float(self.env.cfg.guard_speed) * 0.82
        if self.facility_alert == "SUSPICIOUS":
            speed *= 1.4
        if self.threat_level >= 2:
            speed *= 1.12

        if guard.target is not None:
            guard.target = self._navigable_goal(guard.target)
        guard.pos, guard.angle = self._guard_move(guard, guard.target, speed / self.game_dt,
                                                4.0 if guard.guard_id.startswith("S") or self.threat_level >= 2 else 3.0 if guard.state == "CHASE" else 2.0)
        if scan_angle is not None:
            guard.angle = turn_towards(guard.angle, scan_angle, 2.0 * self.game_dt)

        if guard.target is not None and float(np.linalg.norm(guard.target - guard.pos)) < 0.2:
            if guard.state == "WITHDRAW":
                guard.state = "LEAVING"
            elif guard.state in {"INVESTIGATE", "VERIFY", "RESPOND", "FLANK", "INSPECT BODY", "SUPPORT", "TRACK BLOOD"}:
                if guard.state != "VERIFY" or not visible:
                    response = self._launch_responses.pop(guard.guard_id, None)
                    self._start_guard_search(guard, response["source"] if response is not None else guard.target)

    def _step_aux_guard(self, guard: SecurityGuard) -> None:
        action = None if guard.policy_action is None else TacticalAction(guard.policy_action)
        self._step_guard(guard, action)

    def _apply_guard_intent(self, guard, action, visible):
        if self.spawn_protected or guard.state in {"ESCAPE SMOKE", "REPOSITION", "LEAVING", "EXITED"}:
            return
        point = guard.last_seen
        if action == TacticalAction.REQUEST_LOCAL_SUPPORT and guard.target is not None:
            self._request_support(guard.guard_id, guard.target)
        elif action in {TacticalAction.RADIO_REPORT, TacticalAction.CALL_REINFORCEMENTS}:
            if visible and guard.visual_confidence >= 1.:
                self._publish_contact(self.env.player_pos, guard.guard_id)
        elif action in {TacticalAction.FLANK_LEFT, TacticalAction.FLANK_RIGHT, TacticalAction.INTERCEPT} and point is not None and guard.visual_confidence >= 1.:
            allies = [g.pos for g in self._active_guards() if g.guard_id != guard.guard_id]
            side = 0 if action == TacticalAction.FLANK_LEFT else 1
            goal = firing_position(self.navigator, guard.pos, point, allies, side)
            if goal is not None:
                guard.state = "FLANK"
                guard.target = goal
        elif guard.state in {"PATROL", "OBSERVE", "PROTECT INTEL", "SECURE EXFIL"}:
            if action == TacticalAction.PROTECT_NEAREST_INTEL:
                points = [p for i, p in enumerate(self.env._intel) if self.env.episode_has_intel[i] and self.env.known_intel_present[i]]
                if points:
                    guard.target = min(points, key=lambda p: self.navigator.path_length(guard.pos, p)).copy()
                    guard.pause_angle = guard.angle
                    guard.state = "PROTECT INTEL"
            elif action in {TacticalAction.GO_LAST_SEEN, TacticalAction.PURSUE, TacticalAction.SEARCH_LAST_SEEN} and point is not None:
                self._start_guard_search(guard, point)

    def _step_aux_guards(self) -> None:
        for guard in self.aux_guards:
            self._step_aux_guard(guard)

    def _inject_primary_audio(self) -> None:
        if not self._primary_active() or self.env.sense.sound:
            return
        cue = self._audible_cue(self.env.guard_pos)
        if cue is None:
            return
        if cue[1] != "NOISEMAKER":
            self._local_diversion_guards.discard("G0")
        relative = cue[0] - self.env.guard_pos
        self.env.sense.sound = True
        self.env.sense.sound_rel = (float(relative[0]), float(relative[1]))
        self.env.sense.sound_confidence = 0.74
        self.env.personal_suspicion = max(self.env.personal_suspicion, 0.48)
        if cue[1] == "GUNSHOT":
            self._raise_evidence_alert("GUNFIRE")
            self._request_support("G0", cue[0])

    def _primary_evidence(self) -> None:
        if not self._primary_active() or self._lead_patrol.blinded_timer > 0:
            return
        for body in self.bodies:
            if body.guard_id == "G0" or body.discovered:
                continue
            if can_see(
                self.env.grid,
                self.env.guard_pos,
                self.env.guard_angle,
                body.pos,
                8.0,
                112.0,
            ):
                body.discovered = True
                self._raise_evidence_alert("BODY FOUND", body.pos, "G0")
                self.env.personal_suspicion = 1.0
                self.env.last_seen_pos = body.pos.copy()
                self.env.last_seen_age = 0.0
                self.radio_updates += 1
                return
        if self.env.sense.visual:
            return
        lead = self._lead_patrol
        mark = observe_blood(self.blood, self.env.grid, self.env.guard_pos,
                             self.env.guard_angle, "G0", lead.evidence_id, self._smoke_blocks)
        if mark is not None:
            lead.evidence_id = mark.mark_id
            lead.state = "TRACK BLOOD"
            lead.target = mark.pos.copy()
            self._searches.pop("G0", None)
            self._raise_evidence_alert("BLOOD TRAIL")
            self.env.personal_suspicion = max(self.env.personal_suspicion, 0.86)
            self.env.last_seen_pos = mark.pos.copy()
            self.env.last_seen_age = 0.0

    def tick(self, inp: InputState) -> None:
        self.learning_decisions = []
        self.learning_events = []
        if self.status != "RUNNING":
            return

        self.player_flash_timer = max(0.0, self.player_flash_timer - self.game_dt)
        self.shots = [dict(shot, ttl=shot["ttl"] - self.game_dt) for shot in self.shots if shot["ttl"] > self.game_dt]
        self._lead_vitals_stepped = False
        for effect in self.smoke_clouds + self.flash_events:
            effect["ttl"] -= self.game_dt
        self.smoke_clouds = [c for c in self.smoke_clouds if c["ttl"] > 0]
        for guard in self._active_guards():
            guard.vitals.step_smoke(smoke_density(guard.pos, self.smoke_clouds), self.game_dt)
        self.flash_events = [f for f in self.flash_events if f["ttl"] > 0]

        self._dialogue_timer = max(0.0, self._dialogue_timer - self.game_dt)
        if self._dialogue_timer == 0.0:
            self.guard_dialogue.clear()

        inp = replace(inp)
        self._apply_player(inp)
        if not self._inside_spawn(self.env.player_pos):
            self.spawn_safe = False
        self._process_takedown(inp)
        self._step_diversions()
        self._step_cameras()

        self.env.steps += 1
        self.env.search_completed_event = False
        self.env.reinforcements_spawned_event = False
        self.env.reinforcement_capture_event = False
        self.env.reinforcement_radio_contact = False

        decision = self.env._decision_context()
        self.last_phase = self.env._tactical_phase(decision)

        primary_grappled = bool(
            self.takedown and self.takedown.get("guard_id") == "G0"
        )
        if self._primary_active() and not primary_grappled:
            lead = self._lead_patrol
            lead.pos, lead.angle = self.env.guard_pos.copy(), self.env.guard_angle
            lead.visual_confidence = self.env.visual_confidence
            if self.env.last_seen_pos is not None and self.env.last_seen_age < lead.last_seen_age:
                lead.last_seen = self.env.last_seen_pos.copy()
                lead.last_seen_age = self.env.last_seen_age
                if lead.state in {"PATROL", "OBSERVE"}:
                    self._start_guard_search(lead, lead.last_seen)
            learning_observation = self.env._observation().copy()
            if self.policy_tick % self.policy_every == 0:
                action, _ = self.model.predict(learning_observation, deterministic=True)
                self._primary_policy_action = TacticalAction(int(action))
            policy_action = getattr(self, "_primary_policy_action", TacticalAction.PATROL)
            self._step_guard(lead, policy_action)
            self._lead_vitals_stepped = True
            self.env.guard_pos, self.env.guard_angle = lead.pos.copy(), lead.angle
            self._primary_patrol_action = lead.state
            action_enum = {
                "CHASE": TacticalAction.PURSUE, "VERIFY": TacticalAction.INVESTIGATE_VISUAL,
                "SEARCH": TacticalAction.SEARCH_LAST_SEEN, "TRACK BLOOD": TacticalAction.GO_LAST_SEEN,
                "FLANK": TacticalAction.FLANK_LEFT, "SUPPORT": TacticalAction.GO_LAST_SEEN,
                "RESPOND": TacticalAction.GO_LAST_SEEN, "PROTECT INTEL": TacticalAction.PROTECT_NEAREST_INTEL,
                "INVESTIGATE": TacticalAction.INVESTIGATE_SOUND, "PATROL": TacticalAction.PATROL,
            }.get(lead.state, TacticalAction.HOLD)
            self.last_action = action_enum
            if self.policy_tick % self.policy_every == 0 and not self.spawn_protected:
                self.learning_decisions.append((learning_observation, int(policy_action), "G0"))
            if self.spawn_protected:
                self.env.sense = self.env.sense.__class__()
                lead.visual_confidence = 0.
            else:
                self.env.sense = self._sense_primary()
            # The common sensor owns playable spotting. The environment remains
            # an observation/knowledge adapter for the learned policy.
            self.env.visual_confidence = lead.visual_confidence
            self.env.sense.visual = lead.seeing_player and lead.visual_confidence >= 1. and not self.spawn_protected
            self.env.personal_suspicion = lead.suspicion
            self.env.last_seen_pos = None if lead.last_seen is None else lead.last_seen.copy()
            self.env.last_seen_age = lead.last_seen_age
            self._inject_primary_audio()
            self.env._update_knowledge(inspect_intel=False)
            self.env.last_action = action_enum
            response = self._launch_responses.get("G0")
            if response is not None:
                self._primary_audio_goal = response["goal"].copy()

        else:
            action_enum = TacticalAction.HOLD
            self.last_action = action_enum
            self.env.sense = self.env.sense.__class__()
            if not primary_grappled:
                self.env.last_seen_age += self.game_dt

        self.policy_tick += 1
        self._step_aux_guards()
        self._step_support_radio()
        self._step_guard_dispatch()
        self._step_radio()

        for _, unit in self.swat_units:
            self._step_aux_guard(unit)
        self._step_swat_departures()
        self._step_intel_radio()
        if not self._lead_vitals_stepped:
            self._lead_patrol.vitals.step(False, self.game_dt, fatigue_on_pause=False)
        if self._primary_active():
            self._lead_patrol.pos = self.env.guard_pos.copy()
            self._lead_patrol.angle = self.env.guard_angle
            self._step_guard_weapon(self._lead_patrol, self.env.sense.visual)
        for guard in self.aux_guards + [u for _, u in self.swat_units]:
            if guard.active and not (self.takedown and self.takedown["guard_id"] == guard.guard_id):
                self._step_guard_weapon(guard, guard.visual_confidence >= 1.0)

        self.team_contact_age += self.game_dt
        self._step_casualty_reinforcements()
        self._update_facility_alert()

        for body in self.bodies:
            body.fall = max(0.0, body.fall - self.game_dt * (2.6 if body.lethal else 0.7))
        for mark in self.blood:
            mark.age += self.game_dt

        captured = self.player_vitals.health <= 0
        if not self.spawn_protected:
            if self._primary_active() and self._lead_patrol.vitals.can_aim:
                _, grabbed = self.env._terminal_events(action_enum, decision)
                if ((self.equipment.armed and not self.env.player_compliant
                     or self.player_actions.knockdown_remaining > 0 or self.player_actions.rise_timer > 0)
                        and np.linalg.norm(self.env.guard_pos - self.env.player_pos) <= MELEE_RANGE):
                    grabbed = False
                captured = captured or grabbed
            if not captured:
                captured = any(
                    guard.active
                    and (not self.equipment.armed or self.env.player_compliant)
                    and self.player_actions.knockdown_remaining <= 0 and self.player_actions.rise_timer <= 0
                    and guard.vitals.can_aim
                    and guard.seeing_player
                    and guard.visual_confidence >= 1.0
                    and float(np.linalg.norm(guard.pos - self.env.player_pos)) <= 0.72
                    and line_clear(self.env.grid, guard.pos, self.env.player_pos)
                    for guard in self.aux_guards + [u for _, u in self.swat_units]
                )
        self.spawn_grace = max(0.0, self.spawn_grace - self.game_dt)

        if captured:
            if self.player_vitals.health <= 0 and self.pathogen_exposure:
                self.status = "NEUTRALIZED"
                self.message = "ESPOSIZIONE BIOLOGICA LETALE"
            else:
                self.status = "CAPTURED"
                self.message = "SECURITY CAPTURED YOU"
        elif (
            float(np.linalg.norm(self.env.player_pos - self.env._exit))
            <= self.env.cfg.escape_distance
            and np.all(self.env.player_has_intel[self.env.episode_has_intel.astype(bool)])
        ):
            self.status = "ESCAPED"
            self.message = "YOU REACHED THE EXIT"

    def _player_observes(self, point):
        return (can_see(self.env.grid, self.env.player_pos, self.player_actions.angle, point, 12., 90., 0.)
                and not self._smoke_blocks(self.env.player_pos, point) and self.player_flash_timer <= 0)

    def _guard_state(self, guard: SecurityGuard) -> dict[str, Any]:
        return {
            **guard.vitals.snapshot(),
            "armor": guard.armor, "shield": guard.has_shield,
            "weapon": guard.weapon, "disarmed": guard.disarmed, "grenades": guard.grenade_count,
            "observed": self._player_observes(guard.pos),
            "aiming": guard.aim_timer > 0,
            "aim_progress": min(1.0, guard.aim_timer / (0.35 if self.threat_level >= 2 else 0.95)),
            "ammo": guard.ammo,
            "melee_windup": self._melee_attacks.get(guard.guard_id, EnemyMelee()).windup,
            "id": guard.guard_id,
            "dialogue": self.guard_dialogue.get(guard.guard_id, ""),
            "x": _f(guard.pos[0]),
            "y": _f(guard.pos[1]),
            "angle": _f(guard.angle),
            "action": "MELEE" if self._melee_attacks.get(guard.guard_id, EnemyMelee()).windup > 0 else guard.state,
            "alive": bool(guard.alive),
            "unconscious": bool(guard.unconscious),
            "visual": guard.visual_confidence >= 1.0,
            "confidence": _f(guard.visual_confidence),
            "suspicion": _f(guard.suspicion),
            "last_seen": _vec(guard.last_seen),
            "vision_range": GUARD_VISION_RANGE * (.85 if guard.has_shield else 1.) * (1. - .95 * smoke_density(guard.pos, self.smoke_clouds)),
            "fov": GUARD_FOV - (20. if guard.has_shield else 0.),
            "vision_status": self._vision_status(guard.state, guard.visual_confidence >= 1.0, guard.suspicion),
            "combat_alert": self.team_contact_source == "GUNFIRE" and self.facility_alert in {"CONTACT", "SEARCH"},
        }

    def state(self, include_static: bool = True) -> dict[str, Any]:
        env = self.env
        last_seen = _vec(env.last_seen_pos)
        intel = [
            {
                "id": idx,
                "x": _f(pos[0]),
                "y": _f(pos[1]),
                "stolen": bool(env.player_has_intel[idx]),
                "guard_knows_present": bool(env.known_intel_present[idx]),
                "name": MISSIONS[self.level]['names'][MISSIONS[self.level]['indices'].index(idx)],
            }
            for idx, pos in enumerate(env._intel) if env.episode_has_intel[idx]
        ]
        primary = {
            **self._lead_patrol.vitals.snapshot(),
            "weapon": self._lead_patrol.weapon, "disarmed": self._lead_patrol.disarmed,
            "observed": self._player_observes(env.guard_pos),
            "aiming": self._lead_patrol.aim_timer > 0,
            "melee_windup": self._melee_attacks.get("G0", EnemyMelee()).windup,
            "aim_progress": min(1.0, self._lead_patrol.aim_timer / (0.35 if self.threat_level >= 2 else 0.95)),
            "id": "G0",
            "x": _f(env.guard_pos[0]),
            "y": _f(env.guard_pos[1]),
            "angle": _f(env.guard_angle),
            "action": ("MELEE" if self._melee_attacks.get("G0", EnemyMelee()).windup > 0 else
                       self._primary_patrol_action or self.last_action.name) if self._primary_active() else "DOWN",
            "phase": self.last_phase,
            "alert": int(env.alert),
            "alert_name": self.facility_alert,
            "visual": bool(env.sense.visual) if self._primary_active() else False,
            "candidate": bool(env.sense.visual_candidate) if self._primary_active() else False,
            "confidence": _f(env.visual_confidence) if self._primary_active() else 0.0,
            "sound": bool(env.sense.sound) if self._primary_active() else False,
            "suspicion": _f(env.personal_suspicion),
            "last_seen": last_seen,
            "last_seen_age": None if env.last_seen_pos is None else _f(env.last_seen_age),
            "vision_range": _f(env.cfg.vision_outer) * (1. - .95 * smoke_density(env.guard_pos, self.smoke_clouds)),
            "fov": GUARD_FOV,
            "vision_status": self._vision_status(self.last_action.name, env.sense.visual, env.personal_suspicion),
            "combat_alert": self.team_contact_source == "GUNFIRE" and self.facility_alert in {"CONTACT", "SEARCH"},
            "alive": bool(self.lead_alive),
            "unconscious": bool(self.lead_unconscious),
        }
        guards = [primary] + [self._guard_state(guard) for guard in self.aux_guards]
        candidate = self._nearest_takedown_target()

        return {
            "mode": self.MODE,
            "scenario": env.training_scenario,
            "scenario_options": self.scenarios,
            "model": str(self.model_path),
            "status": self.status,
            "mission": dict(level=self.level,title=MISSIONS[self.level]['title'],
                            objective=MISSIONS[self.level]['objective'],
                            requirements=[dict(name="Protezione biologica",ready=self.equipment.hazmat),
                                          dict(name="Provetta sterile",ready=self.equipment.sample_vial or bool(self.env.player_has_intel[2]))] if self.level == 3 else
                                         [dict(name="Tessera della prima missione · E alla porta dell’archivio",ready=self.equipment.security_card or self.archive_unlocked)] if self.level == 2 else []),
            "message": self.message,
            "map": {
                "width": int(env.cfg.width),
                "height": int(env.cfg.height),
                "grid": env.grid.astype(int).tolist() if include_static else None,
                "exit": _vec(env._exit),
                "intel": intel,
                "semantic": [] if include_static else None,
                "zones": [dict(name=name,x=x,y=y) for name,x,y in self._mission_zones],
                "hazards": [dict(x=41.5,y=19.5,width=4.,height=5.,name="CONTAMINAZIONE BIOLOGICA")] if self.level == 3 else [],
                "doors": [dict(x=x,y=y,locked=not self.archive_unlocked,name="ARCHIVIO · E / TESSERA") for x,y in ARCHIVE_DOORS] if self.level == 2 else [],
            },
            "player": {
                "x": _f(env.player_pos[0]),
                "y": _f(env.player_pos[1]),
                "vx": _f(env.player_velocity[0]),
                "vy": _f(env.player_velocity[1]),
                "speed": _f(env.player_current_speed),
                "crouched": env.player_stance == PlayerStance.CROUCHED,
                "compliant": bool(env.player_compliant),
                **self.player_vitals.snapshot(),
                **self.player_actions.snapshot(),
                **self.equipment.snapshot(),
                "stolen_count": int(np.sum(env.player_has_intel)),
                "pathogen_exposure": getattr(self,'pathogen_exposure',False),
                "diversions": int(self.diversion_inventory),
                "scrap": int(self.scrap),
                "bloody": self.bloody_timer > 0.0 or self.player_vitals.bleeding > 0.0,
                "flashed": self.player_flash_timer > 0.0,
            },
            "guard": primary,
            "guards": guards,
            "cameras": [
                {
                    "id": camera.camera_id,
                    "x": _f(camera.pos[0]),
                    "y": _f(camera.pos[1]),
                    "angle": _f(camera.angle),
                    "vision_range": _f(camera.vision_range),
                    "fov": _f(camera.fov),
                    "confidence": _f(camera.confidence),
                    "tracking": bool(camera.tracking),
                    "online": bool(camera.online),
                }
                for camera in self.cameras
            ],
            "bodies": [
                {
                    "id": body.body_id,
                    "guard_id": body.guard_id,
                    "x": _f(body.pos[0]),
                    "y": _f(body.pos[1]),
                    "angle": _f(body.angle),
                    "lethal": bool(body.lethal),
                    "fall": _f(body.fall),
                    "discovered": bool(body.discovered),
                }
                for body in self.bodies
            ],
            "blood": [
                {"id": mark.mark_id, "x": _f(mark.pos[0]), "y": _f(mark.pos[1]), "age": _f(mark.age)}
                for mark in self.blood
            ],
            "diversions": [
                {
                    "id": device.device_id,
                    "x": _f(device.pos[0]),
                    "y": _f(device.pos[1]),
                    "landed": bool(device.landed),
                    "pulses": int(device.pulses_left),
                }
                for device in self.diversions
            ],
            "noise": [
                {
                    "id": event.event_id,
                    "x": _f(event.pos[0]),
                    "y": _f(event.pos[1]),
                    "radius": _f(event.radius),
                    "ttl": _f(event.ttl),
                    "source": event.source,
                }
                for event in self.noise_events
            ],
            "security": {
                "medical_evidence": len(self.medical_watch.discovered),
                "swat_phase": self.swat_phase,
                "radio_remaining": self.radio_timer,
                "alert": self.facility_alert,
                "high_guard": self.security_compromised,
                "threat_level": self.threat_level,
                "contact": _vec(self.team_contact),
                "contact_age": _f(self.team_contact_age),
                "source": self.team_contact_source,
                "radio_updates": int(self.radio_updates),
                "dialogue": dict(self.guard_dialogue),
                "guards_active": sum(1 for guard in guards if guard["alive"] and not guard["unconscious"]),
                "guards_down": sum(1 for guard in guards if not guard["alive"] or guard["unconscious"]),
                "cameras_tracking": sum(1 for camera in self.cameras if camera.tracking),
            },
            "takedown": self.takedown,
            "grenades": [g.snapshot() for g in self.explosives], "blasts": self.blasts,
            "takedown_candidate": None if candidate is None else {
                "guard_id": candidate[0],
                "can_lethal": self.equipment.knife,
                "can_stun": True,
                "distance": candidate[2],
            },
            "buddy": None,
            "reinforcements": [],
            "swat": [dict(self._guard_state(unit), role=role, local_visual=unit.visual_confidence >= 1.0) for role, unit in self.swat_units],
            "shots": self.shots,
            "pickups": [dict(id=item["id"], kind=item["kind"], amount=item["amount"], source=item["source"], fixed=item.get("fixed", False), zone=item.get("zone", ""), **_vec(item["pos"])) for item in self.pickups],
            "blackboard": None,
            "smoke": [dict(_vec(c["pos"]), radius=c["radius"], ttl=c["ttl"]) for c in self.smoke_clouds],
            "flashes": [dict(_vec(f["pos"]), ttl=f["ttl"]) for f in self.flash_events],
            "gameplay": {
                "dt": self.game_dt,
                "walk_speed": _f(self.env.cfg.player_walk_speed / self.game_dt),
                "sprint_speed": _f(self.env.cfg.player_sprint_speed / self.game_dt),
                "crouch_speed": _f(self.env.cfg.player_stealth_speed / self.game_dt),
                "spawn_grace": _f(self.spawn_grace),
                "spawn_protected": self.spawn_protected,
            },
        }

    def _legacy_apply_player(self, inp: InputState) -> None:
        old = self.env.player_pos.copy()
        direction = inp.direction()

        moving = float(np.linalg.norm(direction)) > 0.0
        compliant = inp.compliant and not moving
        self.env.player_compliant = bool(compliant)

        if compliant:
            speed = 0.0
        elif inp.crouch:
            speed = float(self.env.cfg.player_stealth_speed)
        elif inp.sprint:
            speed = float(self.env.cfg.player_sprint_speed)
        else:
            speed = float(self.env.cfg.player_walk_speed)

        self.env.player_stance = (
            PlayerStance.CROUCHED if inp.crouch else PlayerStance.STANDING
        )
        self.env.player_pos = _move_player_grid(
            self.env.player_pos,
            direction,
            speed,
            self.env._free,
        )
        self.env.player_velocity = (
            self.env.player_pos - old
        ) / max(self.env.cfg.dt, 1e-6)
        self.env.player_current_speed = float(
            np.linalg.norm(self.env.player_velocity)
        )

        if inp.interact and not self._interact_latch:
            for idx, pedestal in enumerate(self.env._intel):
                if self.env.player_has_intel[idx]:
                    continue
                if float(np.linalg.norm(self.env.player_pos - pedestal)) <= 1.05:
                    self.env.player_has_intel[idx] = 1
                    self.message = f"INTEL {idx + 1} STOLEN"
                    break
        self._interact_latch = inp.interact

    def _legacy_tick(self, inp: InputState) -> None:
        if self.status != "RUNNING":
            return

        self._apply_player(inp)

        self.env.steps += 1
        self.env.search_completed_event = False
        self.env.reinforcements_spawned_event = False
        self.env.reinforcement_capture_event = False
        self.env.reinforcement_radio_contact = False

        decision = self.env._decision_context()
        self.last_phase = self.env._tactical_phase(decision)

        # Tactical brain runs at a lower rate than the physical executor.
        # This keeps movement responsive without paying a PPO inference every
        # render/network frame.
        if self.policy_tick % self.policy_every == 0:
            obs = self.env._observation()
            action, _ = self.model.predict(obs, deterministic=True)
            self.last_action = TacticalAction(int(action))

        action_enum = self.last_action
        self.policy_tick += 1
        self.env._apply_action(action_enum)

        # The human replaces the built-in training player bot.
        self.env.sense = self.env._sense()
        self.env._update_knowledge()
        self.env.last_action = action_enum
        self.env._step_buddy()
        self.env._step_reinforcements()

        captured = False
        if self.spawn_grace > 0.0:
            self.spawn_grace = max(
                0.0,
                self.spawn_grace - self.game_dt,
            )
        else:
            _, captured = self.env._terminal_events(action_enum, decision)

        self.last_action = action_enum

        if captured:
            self.status = "CAPTURED"
            self.message = "SECURITY CAPTURED YOU"
        elif (
            float(np.linalg.norm(self.env.player_pos - self.env._exit))
            <= self.env.cfg.escape_distance
        ):
            self.status = "ESCAPED"
            self.message = "YOU REACHED THE EXIT"

    def _legacy_state(self, include_static: bool = True) -> dict[str, Any]:
        env = self.env
        last_seen = _vec(env.last_seen_pos)
        intel = []
        for idx, p in enumerate(env._intel):
            intel.append(
                {
                    "id": idx,
                    "x": _f(p[0]),
                    "y": _f(p[1]),
                    "stolen": bool(env.player_has_intel[idx]),
                    "guard_knows_present": bool(env.known_intel_present[idx]),
                }
            )

        reinforcements = []
        for idx, pos in enumerate(env.reinforcement_positions):
            reinforcements.append(
                {
                    "id": idx,
                    "x": _f(pos[0]),
                    "y": _f(pos[1]),
                    "angle": _f(
                        env.reinforcement_angles[idx]
                        if idx < len(env.reinforcement_angles)
                        else 0.0
                    ),
                }
            )

        return {
            "mode": self.MODE,
            "scenario": env.training_scenario,
            "scenario_options": self.scenarios,
            "model": str(self.model_path),
            "status": self.status,
            "message": self.message,
            "map": {
                "width": int(env.cfg.width),
                "height": int(env.cfg.height),
                "grid": env.grid.astype(int).tolist() if include_static else None,
                "exit": _vec(env._exit),
                "intel": intel,
                "semantic": [] if include_static else None,
            },
            "player": {
                "x": _f(env.player_pos[0]),
                "y": _f(env.player_pos[1]),
                "vx": _f(env.player_velocity[0]),
                "vy": _f(env.player_velocity[1]),
                "speed": _f(env.player_current_speed),
                "crouched": env.player_stance == PlayerStance.CROUCHED,
                "compliant": bool(env.player_compliant),
                "health": 1.0,
                "stolen_count": int(np.sum(env.player_has_intel)),
            },
            "guard": {
                "x": _f(env.guard_pos[0]),
                "y": _f(env.guard_pos[1]),
                "angle": _f(env.guard_angle),
                "action": self.last_action.name,
                "phase": self.last_phase,
                "alert": int(env.alert),
                "alert_name": FacilityAlert(int(env.alert)).name,
                "visual": bool(env.sense.visual),
                "candidate": bool(env.sense.visual_candidate),
                "confidence": _f(env.sense.visual_confidence),
                "sound": bool(env.sense.sound),
                "suspicion": _f(env.personal_suspicion),
                "last_seen": last_seen,
                "last_seen_age": (
                    None if env.last_seen_pos is None else _f(env.last_seen_age)
                ),
                "vision_range": _f(env.cfg.vision_outer),
                "fov": _f(env.cfg.peripheral_fov_degrees),
            },
            "buddy": (
                None
                if env.buddy_pos is None
                else {
                    "x": _f(env.buddy_pos[0]),
                    "y": _f(env.buddy_pos[1]),
                    "supporting": bool(env.buddy_supporting),
                }
            ),
            "reinforcements": reinforcements,
            "swat": [],
            "blackboard": None,
            "smoke": [],
            "gameplay": {
                "dt": self.game_dt,
                "walk_speed": _f(self.env.cfg.player_walk_speed / self.game_dt),
                "sprint_speed": _f(self.env.cfg.player_sprint_speed / self.game_dt),
                "crouch_speed": _f(self.env.cfg.player_stealth_speed / self.game_dt),
                "spawn_grace": _f(self.spawn_grace),
            },
        }


class InteractiveSwatRuntime(RuntimeCombatSystems):
    MODE = "swat"

    def __init__(self, model_path: Path, seed: int = 177) -> None:
        self.model_path = Path(model_path)
        self.model = PPO.load(self.model_path, device="cpu")
        self.core = SwatTeamCore(config=SwatConfig(width=48, height=30), seed=seed)
        # Covered entry alcove, confined to the human-playable map.
        self.core.map.grid[:] = build_research_grid(48, 30, compact=True)
        self.core.map.door_cells = {(10,7), (21,7), (22,7), (20,13), (9,13)}
        self.core.map.semantic_points = self.core.map._build_semantics()
        self.navigator = PlayableNavigation(
            self.core.map.grid, self.core.map.cell, self.core.map.free,
            self.core.map.astar, self.core.map.nearest_free_cell)
        self.core._move_unit = self._move_swat_unit
        self.core._unit_can_see_player = self._swat_can_see_player
        self.core._sense_all = self._sense_swat_units
        self.core.cfg.vision_range = GUARD_VISION_RANGE
        self.core.cfg.fov_degrees = GUARD_FOV
        self.seed = seed
        self.status = "RUNNING"
        self.message = ""

        # SWAT executor already multiplies physical speeds by cfg.dt, so here
        # we only set the actual playable physics timestep.
        self.game_dt = 1.0 / 30.0
        self.core.cfg.dt = self.game_dt
        self.core.cfg.player_walk_speed = 1.7 * PLAYER_SPEED_SCALE
        self.core.cfg.player_sprint_speed = 2.5 * PLAYER_SPEED_SCALE
        self.policy_every = max(1, round(0.20 / self.game_dt))
        self.policy_tick = 0
        self.spawn_grace = 0.0
        self.cached_actions: dict[str, SwatAction] = {}
        self.actions: dict[str, str] = {}
        self._interact_latch = False
        self.reset("entry_clear")

    @property
    def scenarios(self) -> list[str]:
        return [
            "entry_clear",
            "corner_clear",
            "lost_contact",
            "confirmed_visual",
            "utility",
            "arrest",
            "objective_breach",
        ]

    def reset(self, scenario: str = "confirmed_visual") -> None:
        if scenario not in self.scenarios:
            scenario = "confirmed_visual"
        self.seed += 1
        self._departed: set[str] = set()
        self._withdrawals: set[str] = set()
        self._equipment_alert = None
        self.player_vitals = PlayableVitals()
        self._gunshot_cue = None
        self.core.reset_core(seed=self.seed, scenario=scenario)
        for unit in self.core.units.values():
            unit.pos = self.navigator.goal(unit.pos)
        self.player_vitals = PlayableVitals()
        self._wound_trail = BloodTrail()
        self.blood = []
        self._blood_id = 0
        self._blood_targets = {}
        self._blood_progress = {}
        self._blood_searches = {}
        self._unit_vitals = {uid: PlayableVitals(health=u.health, stamina_capacity=SWAT_STAMINA_CAPACITY)
                             for uid, u in self.core.units.items()}
        self.rng = np.random.default_rng(self.seed + 991)
        self._melee_attacks = {}
        self._unit_vitals_stepped = set()
        self._unit_armor = {uid: 1.0 for uid in self.core.units}
        self.core._unit_can_see_player = self._swat_can_see_player
        self.takedown = None
        self.equipment = Equipment(knife=True)
        self.player_actions = PlayerActions()
        self._motion_recovery = MotionRecovery()
        self.pickups = map_pickups(self.navigator, "swat")
        self._reset_combat_systems()
        for unit in self.core.units.values():
            unit.weapon = swat_loadout(unit.role.name, self.seed)
            unit.disarmed = False
            unit.grenade_count = 0 if unit.profile.has_shield else 1
            unit.ammo = unit.max_ammo = 8 if unit.weapon == 'SHOTGUN' else 24
        self.shots = []
        self.flash_events = []
        self._fire_goals = {}
        self._gunshot_cue = None
        self._thrown = []
        self._confirmed_gap = 999.0
        self.core.player_pos = np.array([2.0, 2.0], dtype=np.float32)
        self.core.player_velocity[:] = 0.0
        self.spawn_safe = True
        self.navigator.paths.clear()
        self._searches: dict[str, LocalSearch] = {}
        self._search_anchor = None
        self._searched_contact = None
        self._motions: dict[str, HumanMotion] = {}

        # Runtime-only insertion formation: all centers are placed in free
        # cells directly in front of the right-side doorway, not behind the
        # x=26 wall used by the training map.
        formation = [
            np.array([30.0, 7.4], dtype=np.float32),  # SHIELD
            np.array([29.0, 7.4], dtype=np.float32),  # ASSAULT
            np.array([30.0, 8.6], dtype=np.float32),  # UTILITY
            np.array([29.0, 8.6], dtype=np.float32),  # HEAVY
            np.array([28.0, 8.0], dtype=np.float32),  # LEADER
        ]
        for uid, desired in zip(self.core.agent_ids, formation):
            unit = self.core.units[uid]
            x, y = self.core.map.cell(desired)
            if not self.core.map.free(x, y):
                x, y = self.core.map.nearest_free_cell(x, y)
                desired = np.array([float(x), float(y)], dtype=np.float32)
            unit.pos = desired.copy()
            unit.angle = math.pi

        self.core._assign_semantic_targets()
        self.core._sense_all()
        # A reported location belongs to the scenario; it is never the new spawn.
        for unit in self.core.units.values():
            unit.local_visual = False
            unit.local_visual_confidence = 0.0
            unit.local_visual_rel[:] = 0.0
            unit.local_sound = False
            unit.last_seen_pos = None
            unit.last_seen_age = 999.0

        self.spawn_grace = 2.5
        self.status = "RUNNING"
        self.message = ""
        self.policy_tick = 0
        self.cached_actions = {
            uid: SwatAction.HOLD for uid in self.core.agent_ids
        }
        self.actions = {uid: "HOLD" for uid in self.core.agent_ids}
        self._interact_latch = False

    def _inside_spawn(self, pos: np.ndarray) -> bool:
        return float(pos[0]) < 4.5 and float(pos[1]) < 3.5

    @property
    def spawn_protected(self) -> bool:
        return self.spawn_grace > 1e-6 or (self.spawn_safe and self._inside_spawn(self.core.player_pos))

    def _move_swat_unit(self, unit, target, multiplier: float = 1.0) -> None:
        if target is None or not unit.alive:
            return
        if self._unit_vitals[unit.uid].hit_stun_remaining > 0:
            return
        # Tactical intent changes the destination, not the unit's running pace.
        speed = 0.95 * unit.profile.move_multiplier
        vitals = self._unit_vitals[unit.uid]
        vitals.health = unit.health
        running = np.linalg.norm(target - unit.pos) > 0.2
        speed *= vitals.speed_multiplier
        if not vitals.can_sprint:
            speed = min(speed, 0.4 * unit.profile.move_multiplier)
        motion = self._motions.setdefault(unit.uid, HumanMotion())
        old = unit.pos.copy()
        unit.pos, unit.angle = motion.move(self.navigator, unit.uid, unit.pos, unit.angle,
                                           target, speed, self.game_dt, 4.0)
        # Recover all moving intents, including loss of contact and smoke exits.
        unit.pos = self._motion_recovery.step(self.navigator, unit.uid, old, unit.pos,
            target, speed, self.game_dt, True)
        if unit.uid not in self._unit_vitals_stepped:
            vitals.step(running and np.linalg.norm(unit.pos - old) > 1e-6,
                        self.game_dt, drain=.14, fatigue_on_pause=False)
            self._unit_vitals_stepped.add(unit.uid)

    def _swat_can_see_player(self, unit) -> bool:
        if not unit.alive or unit.blinded_timer > 0 or unit.uid in self._departed:
            return False
        return (can_see(self.core.map.grid, unit.pos, unit.angle, self.core.player_pos,
                        GUARD_VISION_RANGE * (.85 if unit.profile.has_shield else 1.),
                        GUARD_FOV - (20. if unit.profile.has_shield else 0.), self.core.cfg.close_identification_range)
                and not smoke_obscures(unit.pos, self.core.player_pos, self.core.smoke_clouds))

    def _sense_swat_units(self):
        cue = getattr(self, "_gunshot_cue", None)
        for uid, unit in self.core.units.items():
            if uid not in self._departed:
                self.core._sense_unit(unit)
                if (not unit.local_visual and cue is not None
                        and np.linalg.norm(unit.pos - cue["pos"]) <= cue.get("radius", 14)):
                    unit.local_sound = True
                    unit.local_sound_rel = cue["pos"] - unit.pos
                    unit.local_sound_confidence = .8
                if (not unit.local_visual and self.player_vitals.panting
                        and np.linalg.norm(unit.pos - self.core.player_pos) <= 5.):
                    unit.local_sound = True
                    unit.local_sound_rel = self.core.player_pos - unit.pos
                    unit.local_sound_confidence = .65
                if (not unit.local_visual and self.player_vitals.injured
                        and np.linalg.norm(self.core.player_velocity) > .05
                        and np.linalg.norm(unit.pos-self.core.player_pos) <= 5.):
                    unit.local_sound = True
                    unit.local_sound_rel = self.core.player_pos-unit.pos
                    unit.local_sound_confidence = .7

    def _step_swat_medical_watch(self, uid):
        unit = self.core.units[uid]
        goal = self.medical_watch.goal(uid,unit.pos,unit.local_visual,self.navigator.path_length)
        evidence = self.medical_watch.inspect(uid,
            lambda p: can_see(self.core.map.grid,unit.pos,unit.angle,p,GUARD_VISION_RANGE,GUARD_FOV)
            and not smoke_obscures(unit.pos,p,self.core.smoke_clouds) and unit.blinded_timer <= 0)
        if evidence is not None:
            self.message = 'MEDIKIT USATO — LA SWAT HA TROVATO LE TRACCE'
            self._searches[uid] = LocalSearch(local_search_points(self.core.map.grid,evidence,0),scan_seconds=1.2)
            self._equipment_alert = 'SUSPICIOUS'
            self._search_anchor = evidence.copy()
            self._withdrawals.discard(uid)
            return False
        if goal is None:
            return False
        self.actions[uid] = 'WATCH MEDKIT'
        self._withdrawals.discard(uid)
        if np.linalg.norm(unit.pos-goal) > 1.8:
            self._move_swat_unit(unit,goal)
        else:
            unit.angle = turn_towards(unit.angle,math.atan2(float(goal[1]-unit.pos[1]),float(goal[0]-unit.pos[0])),3.*self.game_dt)
        return True

    def _step_swat_withdrawal(self, uid):
        bb = self.core.blackboard
        if bb.contact_pos is not None and bb.contact_age <= 0.5:
            self._withdrawals.clear()
            return False
        if uid not in self._withdrawals and self._searched_contact is None:
            return False
        self._withdrawals.add(uid)
        unit = self.core.units[uid]
        self.actions[uid] = "WITHDRAW"
        exit_pos = np.array([self.core.cfg.width - 1., 7. + self.core.agent_ids.index(uid) * .4], dtype=np.float32)
        if unit.pos[0] >= self.core.cfg.width - 1.08 and abs(unit.pos[1] - exit_pos[1]) < .1:
            unit.angle = turn_towards(unit.angle, 0.0, 2.0 * self.game_dt)
            unit.pos[0] += .62 * self.game_dt * max(0.0, math.cos(unit.angle))
            if unit.pos[0] >= self.core.cfg.width:
                self._departed.add(uid)
                unit.local_visual = False
                unit.local_sound = False
                self.actions[uid] = "EXITED"
        elif unit.blinded_timer <= 0:
            self._move_swat_unit(unit, exit_pos)
        return True

    def _player_observes(self, point):
        return (can_see(self.core.map.grid, self.core.player_pos, self.player_actions.angle, point, 12., 90., 0.)
                and self.core.map.line_clear(self.core.player_pos, point) and not smoke_obscures(self.core.player_pos, point, self.core.smoke_clouds, 12.)
                and self.core.player_flash_timer <= 0)

    def _down_swat_player_target(self, unit, lethal):
        unit.alive = False
        if lethal:
            unit.health = 0.
        self.actions[unit.uid] = "DOWN" if lethal else "STUNNED"
        for kind, amount in ((unit.weapon, unit.ammo), ("FLASH", unit.flash_count), ("SMOKE", unit.smoke_count), ("GRENADE", unit.grenade_count), ("MAGAZINE", 1)):
            if kind and (amount or kind in FIREARMS):
                self.pickups.append(dict(id=f"{unit.uid}-{kind}", kind=kind, amount=amount, pos=unit.pos.copy(), source=unit.uid))
        unit.ammo = unit.flash_count = unit.smoke_count = 0
        unit.grenade_count = 0
        self.takedown = None

    def _swat_melee(self, inp):
        if self.player_actions.movement_locked:
            return
        targets = [u for uid, u in self.core.units.items() if u.alive and uid not in self._departed
                   and np.linalg.norm(u.pos - self.core.player_pos) <= 2.0
                   and self.core.map.line_clear(self.core.player_pos, u.pos)]
        target = min(targets, key=lambda u: np.linalg.norm(u.pos - self.core.player_pos)) if targets else None
        if self.takedown:
            unit = self.core.units[self.takedown["guard_id"]]
            if (not inp.choke or np.linalg.norm(inp.direction()) > .1 or not unit.alive or not is_behind_guard(self.core.player_pos, unit.pos, unit.angle, 2.0)
                    or not spend_energy(self.player_vitals, .20 * self.game_dt)):
                self.takedown = None
                return
            self.takedown["progress"] += self.game_dt
            if self.takedown["progress"] >= 3.:
                self._down_swat_player_target(unit, False)
            return
        if inp.lethal and self.equipment.knife:
            swing = self.player_actions.knife_swing(self.core.player_pos, self.core.map.grid, self.player_vitals,
                                                     None if target is None else target.pos)
            if swing is None:
                return
            if swing["impact"]:
                self._gunshot_cue = {"pos": swing["pos"], "ttl": .7, "radius": 6., "source": "KNIFE IMPACT"}
            if target is None:
                return
            if is_behind_guard(self.core.player_pos, target.pos, target.angle, 2.0):
                self._down_swat_player_target(target, True)
            else:
                attack = self._melee_attacks.get(target.uid)
                if target.profile.has_shield and attack is not None and attack.windup > 0:
                    self.message = "SCUDO — IL COLPO NON È DISARMABILE"
                    return
                self._counter_disarm(target.uid, target, target.profile.has_shield)
                target.health = max(0., target.health - .25)
                self._unit_vitals[target.uid].hit_stun_remaining = .3
                target.aim_timer = 0.
                target.angle = math.atan2(float(self.core.player_pos[1] - target.pos[1]), float(self.core.player_pos[0] - target.pos[0]))
                target.last_seen_pos = self.core.player_pos.copy()
                if target.health <= 0:
                    self._down_swat_player_target(target, True)
        elif target is not None and inp.choke and is_behind_guard(self.core.player_pos, target.pos, target.angle, 2.0) and spend_energy(self.player_vitals, .15):
            self.takedown = dict(guard_id=target.uid, kind="CHOKE", progress=self.game_dt, duration=3.)

    def _new_blood_id(self):
        self._blood_id += 1
        return self._blood_id

    def _step_swat_blood(self, uid):
        unit = self.core.units[uid]
        if unit.local_visual:
            self._blood_targets.pop(uid, None)
            self._blood_searches.pop(uid, None)
            return False
        if unit.blinded_timer > 0:
            return False
        mark = observe_blood(self.blood, self.core.map.grid, unit.pos, unit.angle,
                             uid, self._blood_progress.get(uid, -1),
                             lambda a, b: not self.core.map.line_clear(a, b, self.core.smoke_clouds))
        if mark is not None:
            self._blood_progress[uid] = mark.mark_id
            self._blood_targets[uid] = mark.pos.copy()
            self._withdrawals.discard(uid)
            self._searched_contact = None
            self._searches.pop(uid, None)
            self._blood_searches.pop(uid, None)
        target = self._blood_targets.get(uid)
        if target is None:
            search = self._blood_searches.get(uid)
            if search is not None:
                goal, look, done = search.step(unit.pos, unit.angle, self.game_dt)
                if done:
                    self._blood_searches.pop(uid, None)
                    return False
                self.actions[uid] = "SEARCH BLOOD TRAIL"
                if look is not None:
                    unit.angle = turn_towards(unit.angle, look, 2.0 * self.game_dt)
                else:
                    self._move_swat_unit(unit, goal, .82)
                return True
            return False
        self.actions[uid] = "TRACK BLOOD"
        self._move_swat_unit(unit, target, .82)
        if np.linalg.norm(unit.pos - target) < .3:
            self._blood_targets.pop(uid, None)
            self._blood_searches[uid] = LocalSearch(local_search_points(self.core.map.grid, target, self.core.agent_ids.index(uid)))
        return True

    def _step_swat_search(self, uid: str) -> bool:
        unit = self.core.units[uid]
        bb = self.core.blackboard
        if unit.local_visual or (bb.contact_pos is not None and bb.contact_age <= 0.5):
            self._searches.clear()
            self._search_anchor = None
            self._searched_contact = None
            return False
        if bb.contact_pos is None and self._search_anchor is None:
            return False
        # Keep the search already assigned after radio memory expires, so the
        # journey to a distant last-seen area can actually reach its inspections.
        anchor = self._search_anchor if bb.contact_pos is None else self.navigator.goal(bb.contact_pos)
        if self._search_anchor is None or np.linalg.norm(anchor - self._search_anchor) > 1.0:
            self._search_anchor = anchor.copy()
            self._searches.clear()
            self._searched_contact = None
        if self._searched_contact is not None and np.linalg.norm(anchor - self._searched_contact) <= 1.0:
            return False
        if uid not in self._searches:
            self._searches[uid] = LocalSearch(local_search_points(
                self.core.map.grid, anchor, self.core.agent_ids.index(uid)))
        search = self._searches[uid]
        target, look, done = search.step(unit.pos, unit.angle, self.game_dt)
        if done:
            self.actions[uid] = "SEARCH COMPLETE"
            living = [key for key, other in self.core.units.items() if other.alive and key not in self._departed]
            if all(key in self._searches and self._searches[key].index >= len(self._searches[key].points)
                   for key in living):
                self._searched_contact = anchor.copy()
            return True
        self.actions[uid] = "CHECK CORNER" if look is not None else "SEARCH LAST SEEN"
        if unit.blinded_timer <= 0.0:
            if look is not None:
                unit.angle = turn_towards(unit.angle, look, 2.0 * self.game_dt)
            else:
                self._move_swat_unit(unit, target, 0.82)
        return True

    def _step_swat_firing_position(self, uid):
        unit = self.core.units[uid]
        if not unit.local_visual or unit.last_seen_pos is None:
            self._fire_goals.pop(uid, None)
            return False
        target = unit.last_seen_pos
        allies = [u.pos for key, u in self.core.units.items() if key != uid and u.alive and key not in self._departed]
        if not lane_blocked(unit.pos, target, allies):
            self._fire_goals.pop(uid, None)
            return False
        goal = self._fire_goals.get(uid)
        if goal is None or np.linalg.norm(unit.pos - goal) < .2:
            goal = firing_position(self.navigator, unit.pos, target, allies, self.core.agent_ids.index(uid))
            if goal is not None:
                self._fire_goals[uid] = goal
        self.actions[uid] = "REPOSITION" if goal is not None else "HOLD FIRE"
        if goal is not None and unit.blinded_timer <= 0:
            vitals = self._unit_vitals[uid]
            speed = (.8 if vitals.can_sprint else .4) * vitals.speed_multiplier
            old = unit.pos.copy()
            unit.pos = self.navigator.move(uid, unit.pos, goal, speed * self.game_dt)
            vitals.step(np.linalg.norm(unit.pos - old) > 1e-6, self.game_dt,
                        drain=.14, fatigue_on_pause=False)
            self._unit_vitals_stepped.add(uid)
            heading = math.atan2(float(target[1] - unit.pos[1]), float(target[0] - unit.pos[0]))
            unit.angle = turn_towards(unit.angle, heading, 4.0 * self.game_dt)
        return True

    def _collect_swat_pickups(self):
        if self.player_actions.knockdown_remaining > 0 or self.player_actions.rise_timer > 0:
            return
        collected = []
        for item in self.pickups:
            if item.get("fixed"):
                continue
            if np.linalg.norm(item["pos"] - self.core.player_pos) > 1.2 or not self.core.map.line_clear(self.core.player_pos, item["pos"]):
                continue
            kind = item["kind"]
            if not self.equipment.collect(kind, item["amount"]):
                continue
            collected.append(item["id"])
        self.pickups = [item for item in self.pickups if item["id"] not in collected]

    def _swat_security_event(self, source, point):
        self.spawn_safe = False
        self.spawn_grace = 0.
        if source == "SMOKE":
            if self._equipment_alert != "CONTACT":
                self._equipment_alert = "SUSPICIOUS"
            return
        self._equipment_alert = "CONTACT"
        bb = self.core.blackboard
        bb.alert = FacilityAlert.CONFIRMED
        bb.confirmed_intruder = True
        bb.contact_pos = point.copy()
        bb.contact_velocity[:] = 0.
        bb.contact_age = 0.
        bb.contact_reported = True
        bb.radio_updates += 1
        self._gunshot_cue = {"pos": point.copy(), "ttl": .72,
                            "radius": float(np.hypot(self.core.cfg.width, self.core.cfg.height))}
        self._withdrawals.clear()
        self._searches.clear()
        self._searched_contact = None

    def _swat_player_equipment(self, inp):
        for device in self._thrown:
            device.flight = min(1., device.flight + self.game_dt / .42)
            device.pos = device.start + (device.target - device.start) * device.flight
            if device.flight >= 1. and not device.landed:
                device.landed = True
                self._gunshot_cue = {"pos": device.target.copy(), "ttl": 1.2}
            if device.landed:
                device.age += self.game_dt
        self._thrown = [d for d in self._thrown if d.age < 1.5]
        if inp.throw and self.equipment.take_diversion():
            origin = self.core.player_pos.copy()
            aim = np.array([inp.aim_x, inp.aim_y], dtype=np.float32)
            endpoint, _ = trace_shot(self.core.map.grid, origin, aim, [], reach=min(8., float(np.linalg.norm(aim - origin))))
            self._thrown.append(ThrownDiversion(self._new_blood_id(), origin, endpoint, origin.copy(), pulses_left=1))
        self.equipment.fire_cooldown = max(0.0, self.equipment.fire_cooldown - self.game_dt)
        origin = self.core.player_pos
        aim = np.array([inp.aim_x, inp.aim_y], dtype=np.float32)
        endpoint, _ = trace_shot(self.core.map.grid, origin, aim, [], reach=min(8.0, float(np.linalg.norm(aim - origin))))
        if inp.use_smoke and self.equipment.smoke > 0 and spend_energy(self.player_vitals, .12):
            self.equipment.smoke -= 1
            self.core.smoke_clouds.append(SmokeCloud(tuple(float(v) for v in endpoint), radius=3.0, ttl=8.0))
            self._swat_security_event("SMOKE", endpoint)
        if inp.use_flash and self.equipment.flash > 0 and spend_energy(self.player_vitals, .12):
            self.equipment.flash -= 1
            self.flash_events.append({"pos": endpoint.copy(), "ttl": .45})
            for unit in self.core.units.values():
                if unit.alive and np.linalg.norm(unit.pos - endpoint) < 3 and self.core.map.line_clear(unit.pos, endpoint):
                    unit.blinded_timer = 2.4
        if (not inp.fire or not self.player_vitals.can_aim or self.equipment.fire_cooldown > 0
                or not self.equipment.weapon or self.equipment.ammo <= 0
                or self.player_actions.knockdown_remaining > 0 or self.player_actions.rise_timer > 0):
            return
        if np.linalg.norm(aim - origin) < .1:
            return
        units = [u for u in self.core.units.values() if u.alive and u.uid not in self._departed]
        endpoint, unit = player_shot(self.core.map.grid,origin,aim,units,self.equipment.weapon,self.rng)
        if not spend_energy(self.player_vitals, .08 if self.equipment.weapon in {'PISTOL','RIFLE'} else WEAPONS[self.equipment.weapon].energy):
            return
        self.equipment.ammo -= 1
        self.equipment.fire_cooldown = {'RIFLE':.2,'PISTOL':.45}.get(self.equipment.weapon,WEAPONS[self.equipment.weapon].cooldown)
        if unit is not None:
            unit.health, self._unit_armor[unit.uid], effect = projectile_damage(unit.health,
                self._unit_armor[unit.uid], self._unit_vitals[unit.uid], unit.profile.has_shield,
                unit.angle, unit.pos, origin, self.equipment.weapon)
            self.message = effect
            if unit.health <= 0:
                self._down_swat_player_target(unit,True)
        self.shots.append({"from": _vec(origin), "to": _vec(endpoint), "ttl": .12})
        self._swat_security_event("GUNFIRE", origin)

    def _apply_player(self, inp: InputState) -> None:
        core = self.core
        old = core.player_pos.copy()
        self.player_vitals.health = core.player_health
        self._prepare_condition_input(inp)
        direction = inp.direction()
        moving = float(np.linalg.norm(direction)) > 0.0

        core.player_compliant = bool(inp.compliant and not moving)
        self.player_vitals.health = core.player_health
        self.player_vitals.step_smoke(smoke_density(old, core.smoke_clouds), self.game_dt)
        self.player_actions.step(inp, old, self.game_dt)
        inp.aim_x = float(old[0] + math.cos(self.player_actions.angle) * 8.)
        inp.aim_y = float(old[1] + math.sin(self.player_actions.angle) * 8.)
        self.player_actions.interaction(inp, old, self.game_dt, self.equipment, self.player_vitals,
            self.pickups, core.map.line_clear, busy=moving or inp.dive or inp.attack_held or self.player_actions.movement_locked)
        self._player_extra_items(inp)
        if self.player_actions.station_id and self.player_vitals.treatment == 'MEDKIT':
            self.medical_watch.used.add(self.player_actions.station_id)
        self._swat_melee(inp)
        self.equipment.medical_input(self.player_vitals, inp,
            busy=moving or inp.fire or inp.choke or inp.lethal or inp.throw or inp.craft
                 or inp.use_flash or inp.use_smoke or inp.use_grenade or inp.use_snack
                 or self.player_actions.movement_locked)
        boost = self.player_vitals.boost_remaining > 0
        sprinting = self.player_vitals.step(inp.sprint and moving and not inp.crouch and not core.player_compliant
                                           and not self.player_actions.movement_locked and self.takedown is None,
                                           self.game_dt, drain=0.22 if boost else 0.16)
        if core.player_compliant or self.player_actions.movement_locked or self.takedown is not None:
            speed = 0.0
        elif inp.crouch:
            speed = core.cfg.player_walk_speed * 0.55
        elif sprinting:
            speed = core.cfg.player_sprint_speed
            if boost:
                speed *= 1.65
        else:
            speed = core.cfg.player_walk_speed

        speed *= self.player_vitals.speed_multiplier
        if core.player_flash_timer > 0:
            speed *= core.cfg.player_flash_speed_multiplier

        core.player_pos = _move_player_grid(
            core.player_pos,
            direction,
            float(speed) * self.game_dt,
            core.map.free,
        )
        core.player_pos = self.player_actions.dive(inp, core.player_pos, core.map.grid, self.player_vitals)
        if self.player_actions.dive_timer > .34:
            self._gunshot_cue = {"pos": core.player_pos.copy(), "ttl": .9}
        core.player_velocity = (
            core.player_pos - old
        ) / max(core.cfg.dt, 1e-6)
        if inp.interact and not self._interact_latch:
            self._collect_swat_pickups()
        self._interact_latch = inp.interact
        loss = self.player_vitals.step_bleeding(self.game_dt, float(np.linalg.norm(core.player_pos - old)) / self.game_dt)
        used_station = self.player_actions.treatment_step(self.player_vitals, self.pickups, self.game_dt)
        if used_station:
            self.medical_watch.used.add(used_station)
        core.player_health = self.player_vitals.health
        self._wound_trail.deposit(self.blood, old, core.player_pos, loss, self._new_blood_id)
        if self.player_vitals.bleeding <= 0:
            self._wound_trail.pending = 0.0
        for mark in self.blood:
            mark.age += self.game_dt
        if core.player_health <= core.cfg.player_neutralized_health:
            core.player_neutralized = True
        self._swat_player_equipment(inp)
        self._step_explosives()

        x, y = core.map.cell(core.player_pos)
        if x <= 1 and y >= core.cfg.height - 3:
            core.player_escaped = True

    def _world_after_ai(self) -> None:
        core = self.core
        core.world_ticks += 1
        core.contact_event = False
        core.capture_event = False
        core.feature_clear_event = False
        core.utility_effect_event = False

        core.player_flash_timer = max(
            0.0,
            core.player_flash_timer - core.cfg.dt,
        )
        for cloud in core.smoke_clouds:
            cloud.ttl -= core.cfg.dt
        core.smoke_clouds = [c for c in core.smoke_clouds if c.ttl > 0]

        for unit in core.units.values():
            unit.blinded_timer = max(0.0, unit.blinded_timer - core.cfg.dt)

        core.blackboard.tick(
            core.cfg.dt,
            core.cfg.radio_memory_seconds,
        )
        # Keep every physical unit on navigable geometry. This is a runtime
        # safety net; the tactical policy still chooses the movement intent.
        for unit in core.units.values():
            if not unit.alive or unit.uid in self._departed or (unit.uid in self._withdrawals and unit.pos[0] > core.cfg.width - 1.):
                continue
            x, y = core.map.cell(unit.pos)
            if not core.map.free(x, y):
                x, y = core.map.nearest_free_cell(x, y)
                unit.pos = np.array([float(x), float(y)], dtype=np.float32)

        core._sense_all()
        if self.takedown:
            core.units[self.takedown["guard_id"]].local_visual = False

        if any(u.alive and u.local_visual for u in core.units.values()):
            if self._confirmed_gap > 12.0:
                self.player_vitals.boost_remaining = 2.5
            self._confirmed_gap = 0.0
        else:
            self._confirmed_gap += self.game_dt
        for uid, vitals in self._unit_vitals.items():
            vitals.health = core.units[uid].health
            if uid not in self._unit_vitals_stepped:
                vitals.step(False, self.game_dt, fatigue_on_pause=False)

        if not self.spawn_protected:
            melee_blocked = set()
            for uid, unit in core.units.items():
                recovering = self.player_actions.knockdown_remaining > 0 or self.player_actions.rise_timer > 0
                close_armed = (self.equipment.armed or recovering) and np.linalg.norm(unit.pos - core.player_pos) <= MELEE_RANGE
                ready = (unit.alive and uid not in self._departed and unit.local_visual
                         and not core.player_compliant and unit.blinded_timer <= 0
                         and not smoke_density(unit.pos, core.smoke_clouds)
                         and not (self.takedown and self.takedown['guard_id'] == uid))
                melee = self._melee_attacks.setdefault(uid, EnemyMelee())
                profile = MELEE_PROFILES['SHIELD' if unit.profile.has_shield else 'SWAT']
                hit = melee.step(ready and close_armed and self.player_actions.knockdown_remaining <= 0
                                 and self.player_actions.rise_timer <= 0 and self.player_actions.dive_timer <= 0,
                                 self._unit_vitals[uid], self.game_dt, profile)
                if close_armed:
                    melee_blocked.add(uid)
                if hit:
                    self._melee_impact(uid, unit, profile)
            for uid, unit in core.units.items():
                timer = max(0., self._enemy_fire_timers.get(uid, 0.) - self.game_dt)
                self._enemy_fire_timers[uid] = timer
                if not unit.alive or uid in self._departed or not unit.local_visual or uid in melee_blocked:
                    continue
                vitals = self._unit_vitals[uid]
                if not vitals.can_aim or unit.blinded_timer > 0 or smoke_density(unit.pos, core.smoke_clouds):
                    continue
                distance = float(np.linalg.norm(unit.pos - core.player_pos))
                if core.player_compliant or not self.equipment.armed:
                    if distance <= (1.2 if core.player_compliant else .62):
                        core.player_captured = core.capture_event = True
                        break
                allies = [u.pos for key,u in core.units.items() if key != uid and u.alive and key not in self._departed]
                if core.player_compliant or unit.disarmed or lane_blocked(unit.pos,core.player_pos,allies) or timer > 0:
                    continue
                if self._enemy_grenade(uid, unit, vitals):
                    self._enemy_fire_timers[uid] = 1.5
                elif self._enemy_shot(uid, unit, vitals):
                    self._enemy_fire_timers[uid] = WEAPONS[unit.weapon].cooldown
            if core.player_health <= core.cfg.player_neutralized_health:
                core.player_neutralized = True

    def tick(self, inp: InputState) -> None:
        self.learning_decisions = []
        self.learning_events = []
        if self.status != "RUNNING":
            return

        inp = replace(inp)
        self._apply_player(inp)
        if not self._inside_spawn(self.core.player_pos):
            self.spawn_safe = False
        if self.spawn_protected:
            self.spawn_grace = max(0.0, self.spawn_grace - self.game_dt)
            return

        self._unit_vitals_stepped.clear()
        for uid, unit in self.core.units.items():
            vitals = self._unit_vitals[uid]
            vitals.health = unit.health
            vitals.step_smoke(smoke_density(unit.pos, self.core.smoke_clouds), self.game_dt)
            unit.health = vitals.health
        if self._gunshot_cue is not None:
            self._gunshot_cue["ttl"] -= self.game_dt
            if self._gunshot_cue["ttl"] <= 0:
                self._gunshot_cue = None
        self.shots = [dict(shot, ttl=shot["ttl"] - self.game_dt) for shot in self.shots if shot["ttl"] > self.game_dt]
        self.flash_events = [dict(f, ttl=f["ttl"] - self.game_dt) for f in self.flash_events if f["ttl"] > self.game_dt]

        # One batched PPO call for all living agents. This is substantially
        # cheaper than five separate PyTorch forward passes.
        living = [
            uid for uid in self.core.agent_ids
            if self.core.units[uid].alive and uid not in self._departed
        ]

        refresh_policy = (self.policy_tick % self.policy_every == 0)
        if refresh_policy and living:
            batch = np.stack(
                [self.core.observation(uid) for uid in living],
                axis=0,
            )
            batch_actions, _ = self.model.predict(
                batch,
                deterministic=True,
            )
            for uid, action in zip(living, batch_actions):
                self.cached_actions[uid] = SwatAction(int(action))

        one_shot = {
            SwatAction.USE_FLASH,
            SwatAction.USE_SMOKE,
            SwatAction.RELOAD,
        }

        for uid in self.core.agent_ids:
            unit = self.core.units[uid]
            if uid in self._departed:
                continue
            if self.takedown and self.takedown["guard_id"] == uid:
                self.actions[uid] = "GRAPPLED"
                unit.local_visual = False
                continue
            if not unit.alive:
                self.actions[uid] = "DOWN"
                continue
            if self._unit_vitals[uid].hit_stun_remaining > 0:
                self.actions[uid] = "HIT STUN"
                unit.aim_timer = 0.
                self._unit_vitals[uid].step(False, self.game_dt, fatigue_on_pause=False)
                self._unit_vitals_stepped.add(uid)
                continue
            if smoke_density(unit.pos, self.core.smoke_clouds):
                goal = getattr(unit, "smoke_goal", None)
                if goal is None or smoke_density(goal, self.core.smoke_clouds):
                    goal = smoke_escape(self.navigator, unit.pos, self.core.smoke_clouds)
                    unit.smoke_goal = goal
                unit.aim_timer = 0.
                self.actions[uid] = "ESCAPE SMOKE"
                self._move_swat_unit(unit, goal)
                continue
            unit.smoke_goal = None
            if self._step_swat_medical_watch(uid) or self._step_swat_firing_position(uid) or self._step_swat_blood(uid) or self._step_swat_withdrawal(uid) or self._step_swat_search(uid):
                continue

            action_enum = self.cached_actions.get(uid, SwatAction.HOLD)
            if unit.disarmed and action_enum == SwatAction.RELOAD:
                action_enum = SwatAction.PURSUE
            if unit.local_visual and action_enum in {SwatAction.HOLD, SwatAction.WATCH_ENTRY, SwatAction.REGROUP}:
                action_enum = SwatAction.PURSUE
            elif unit.local_sound and action_enum == SwatAction.HOLD:
                action_enum = SwatAction.ADVANCE
            self.actions[uid] = action_enum.name

            # Physical movement intents persist between neural decisions.
            # Consumable actions fire only on a policy-refresh frame.
            if refresh_policy or action_enum not in one_shot:
                if refresh_policy:
                    self.learning_decisions.append((self.core.observation(uid).copy(), int(action_enum), uid))
                before = unit.pos.copy()
                self.core.apply_action(uid, action_enum)
                movement = unit.pos - before
                vitals = self._unit_vitals[uid]
                moving = np.linalg.norm(movement) > 1e-6
                if uid not in self._unit_vitals_stepped:
                    vitals.step(moving, self.game_dt, drain=.14, fatigue_on_pause=False)
                    self._unit_vitals_stepped.add(uid)
                    scale = vitals.speed_multiplier * (.5 if moving and not vitals.can_sprint else 1.)
                    unit.pos = before + movement * scale

        self.policy_tick += 1
        self._world_after_ai()

        if self.core.player_captured:
            self.status = "CAPTURED"
            self.message = "SWAT ARRESTED YOU"
        elif self.core.player_neutralized:
            self.status = "NEUTRALIZED"
            self.message = "SWAT NEUTRALIZED YOU"
        elif self.core.player_escaped:
            self.status = "ESCAPED"
            self.message = "YOU ESCAPED"

    def state(self, include_static: bool = True) -> dict[str, Any]:
        core = self.core
        units = []
        for uid in core.agent_ids:
            if uid in self._departed:
                continue
            unit = core.units[uid]
            units.append(
                {
                    "id": uid,
                    **self._unit_vitals[uid].snapshot(),
                    "armor": self._unit_armor[uid], "shield": unit.profile.has_shield,
                    "weapon": unit.weapon, "disarmed": unit.disarmed, "grenades": unit.grenade_count,
                    "observed": self._player_observes(unit.pos),
                    "role": unit.role.name,
                    "melee_windup": self._melee_attacks.get(uid, EnemyMelee()).windup,
                    "x": _f(unit.pos[0]),
                    "y": _f(unit.pos[1]),
                    "angle": _f(unit.angle),
                    "health": _f(unit.health),
                    "ammo": int(unit.ammo),
                    "max_ammo": int(unit.max_ammo),
                    "smoke": int(unit.smoke_count),
                    "flash": int(unit.flash_count),
                    "alive": bool(unit.alive),
                    "action": "MELEE" if self._melee_attacks.get(uid, EnemyMelee()).windup > 0 else self.actions.get(uid, "HOLD"),
                    "local_visual": bool(unit.local_visual),
                    "visual_confidence": _f(unit.local_visual_confidence),
                    "local_sound": bool(unit.local_sound),
                    "last_seen": _vec(unit.last_seen_pos),
                    "last_seen_age": (
                        None
                        if unit.last_seen_pos is None
                        else _f(unit.last_seen_age)
                    ),
                    "vision_range": GUARD_VISION_RANGE * (.85 if unit.profile.has_shield else 1.) * (1. - .95 * smoke_density(unit.pos, core.smoke_clouds)),
                    "fov": GUARD_FOV - (20. if unit.profile.has_shield else 0.),
                    "vision_status": "ALERT" if unit.local_visual else "SUSPICIOUS",
                    "combat_alert": self._equipment_alert == "CONTACT",
                    "assigned_sector": int(unit.assigned_sector),
                    "assigned_feature_id": unit.assigned_feature_id,
                }
            )

        semantic = []
        if include_static:
            semantic = [
                {
                    "id": p.feature_id,
                    "type": p.feature.name,
                    "x": _f(p.position[0]),
                    "y": _f(p.position[1]),
                    "sector": int(p.sector_id),
                    "cleared": p.feature_id in core.blackboard.cleared_feature_ids,
                }
                for p in core.map.semantic_points
            ]

        bb = core.blackboard
        return {
            "mode": self.MODE,
            "scenario": core.scenario,
            "scenario_options": self.scenarios,
            "model": str(self.model_path),
            "status": self.status,
            "message": self.message,
            "map": {
                "width": int(core.cfg.width),
                "height": int(core.cfg.height),
                "grid": core.map.grid.astype(int).tolist() if include_static else None,
                "exit": {"x": 1.0, "y": float(core.cfg.height - 2)},
                "intel": [],
                "zones": facility_zones(compact=True),
                "semantic": semantic if include_static else None,
            },
            "player": {
                "x": _f(core.player_pos[0]),
                "y": _f(core.player_pos[1]),
                "vx": _f(core.player_velocity[0]),
                "vy": _f(core.player_velocity[1]),
                "speed": _f(np.linalg.norm(core.player_velocity)),
                "crouched": self.player_actions.crouched,
                "compliant": bool(core.player_compliant),
                **self.player_vitals.snapshot(),
                **self.player_actions.snapshot(),
                "health": _f(core.player_health),
                "bloody": self.player_vitals.bleeding > 0,
                **self.equipment.snapshot(),
                "stolen_count": 0,
                "flashed": core.player_flash_timer > 0.0,
            },
            "guard": None,
            "buddy": None,
            "reinforcements": [],
            "swat": units,
            "takedown": self.takedown,
            "diversions": [dict(id=d.device_id, landed=d.landed, pulses=0, **_vec(d.pos)) for d in self._thrown],
            "blood": [dict(id=m.mark_id, age=m.age, **_vec(m.pos)) for m in self.blood],
            "shots": self.shots,
            "grenades": [g.snapshot() for g in self.explosives], "blasts": self.blasts,
            "pickups": [dict(id=item["id"], kind=item["kind"], amount=item["amount"], source=item["source"], fixed=item.get("fixed", False), zone=item.get("zone", ""), **_vec(item["pos"])) for item in self.pickups],
            "flashes": [dict(_vec(f["pos"]), ttl=f["ttl"]) for f in self.flash_events],
            "blackboard": {
                "contact": _vec(bb.contact_pos),
                "contact_age": _f(bb.contact_age),
                "reported": bool(bb.contact_reported),
                "compliant_known": bool(bb.player_compliant_known),
                "theft_known": bool(bb.theft_known),
                "radio_updates": int(bb.radio_updates),
                "cleared_count": len(bb.cleared_feature_ids),
            },
            "security": {
                "alert": "CONTACT" if self._equipment_alert == "CONTACT" else "CONTACT" if bb.contact_pos is not None and bb.contact_age <= .5 else "SEARCH" if any(search.index < len(search.points) for search in self._searches.values()) else "SUSPICIOUS",
                "contact": _vec(bb.contact_pos) or _vec(self._search_anchor),
                "high_guard": True,
                "swat_phase": "WITHDRAWN" if len(self._departed) == len(core.agent_ids) else "RETURNING" if self._withdrawals else "ACTIVE",
            },
            "smoke": [
                {
                    "x": _f(c.center[0]),
                    "y": _f(c.center[1]),
                    "radius": _f(c.radius),
                    "ttl": _f(c.ttl),
                }
                for c in core.smoke_clouds
            ],
            "gameplay": {
                "dt": self.game_dt,
                "walk_speed": _f(core.cfg.player_walk_speed),
                "sprint_speed": _f(core.cfg.player_sprint_speed),
                "crouch_speed": _f(core.cfg.player_walk_speed * 0.55),
                "spawn_grace": _f(self.spawn_grace),
                "spawn_protected": self.spawn_protected,
            },
        }
