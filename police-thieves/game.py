"""
Police & Thieves – game logic
2D tile-based hide-and-seek:
  - 1 police vs N thieves on a 20×20 map
  - Hiding phase (30 s) → thieves move freely, police is frozen
  - Active phase (round timer) → police hunts; thieves can move & counter-capture
  - Capture: proximity < CAPTURE_RADIUS world units (3D distance from raw_x/raw_z)
  - Win: police catches all thieves | a thief catches police | timer expires
"""

import asyncio
import random
from math import sqrt
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ── Constants ──────────────────────────────────────────────────────────────────

MAP_W = 20
MAP_H = 20
# Capture radii in raw 3D world units.
# City: BLOCK_GAP=22 wu/block, character capsule radius ≈ 0.4 wu.
# A "touch" is roughly 1–2 wu apart.
CAPTURE_RADIUS   = 3.5   # police catches thief  (generous tag range)
COUNTER_RADIUS   = 3.5   # thief catches police   (same range; thief wins if in range first)
HIDE_TIME = 30              # seconds
DEFAULT_ROUND_TIME = 120    # seconds
TICK_RATE = 0.05            # 20 ticks/s
MAX_PLAYERS = 6

# ── Tile map: 0=floor, 1=wall, 2=obstacle ─────────────────────────────────────
# A hand-crafted 20×20 map with walls on borders + internal obstacles.

_RAW_MAP: List[List[int]] = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,2,0,0,0,0,0,0,0,0,2,0,0,0,0,1],
    [1,0,1,1,0,0,0,1,0,2,2,0,1,0,0,0,1,1,0,1],
    [1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1],
    [1,0,0,0,2,0,0,0,0,1,1,0,0,0,2,0,0,0,0,1],
    [1,2,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,2,1],
    [1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1],
    [1,0,0,1,0,0,0,0,2,0,0,2,0,0,0,0,1,0,0,1],
    [1,0,0,0,0,2,0,0,0,0,0,0,0,0,2,0,0,0,0,1],
    [1,0,1,0,0,0,0,2,0,0,0,0,2,0,0,0,0,1,0,1],
    [1,0,1,0,0,0,0,2,0,0,0,0,2,0,0,0,0,1,0,1],
    [1,0,0,0,0,2,0,0,0,0,0,0,0,0,2,0,0,0,0,1],
    [1,0,0,1,0,0,0,0,2,0,0,2,0,0,0,0,1,0,0,1],
    [1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1],
    [1,2,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,2,1],
    [1,0,0,0,2,0,0,0,0,1,1,0,0,0,2,0,0,0,0,1],
    [1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1],
    [1,0,1,1,0,0,0,1,0,2,2,0,1,0,0,0,1,1,0,1],
    [1,0,0,0,0,2,0,0,0,0,0,0,0,0,2,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

TILE_MAP: List[List[int]] = _RAW_MAP   # shared read-only

# Spawn points: safe floor tiles spread across the map
_POLICE_SPAWN: Tuple[float, float] = (1.5, 1.5)
_THIEF_SPAWNS: List[Tuple[float, float]] = [
    (18.5, 18.5), (18.5, 1.5), (1.5, 18.5),
    (10.5, 10.5), (5.5, 15.5), (15.5, 5.5),
]


# ── Enums / dataclasses ────────────────────────────────────────────────────────

class Phase(str, Enum):
    LOBBY   = "lobby"
    HIDING  = "hiding"
    ACTIVE  = "active"
    ENDED   = "ended"


class Role(str, Enum):
    POLICE = "police"
    THIEF  = "thief"


@dataclass
class Player:
    id: str
    name: str
    ws: object = None           # WebSocket (set at runtime)
    role: Optional[Role] = None
    x: float = 1.5             # tile coord
    y: float = 1.5             # tile coord
    alive: bool = True          # False = spectating (captured thief)
    is_connected: bool = True
    # Raw 3D coordinates reported by the Three.js client
    raw_x: float = 0.0
    raw_z: float = 0.0
    raw_rot: float = 0.0

    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_dict(self, viewer_role: Optional[Role] = None) -> dict:
        return {
            "id":           self.id,
            "name":         self.name,
            "role":         self.role,
            "x":            round(self.x, 2),
            "y":            round(self.y, 2),
            "raw_x":        round(self.raw_x, 1),
            "raw_z":        round(self.raw_z, 1),
            "raw_rot":      round(self.raw_rot, 3),
            "alive":        self.alive,
            "is_connected": self.is_connected,
        }


# ── Helper ─────────────────────────────────────────────────────────────────────

def _is_walkable(x: float, y: float) -> bool:
    tx, ty = int(x), int(y)
    if tx < 0 or ty < 0 or tx >= MAP_W or ty >= MAP_H:
        return False
    return TILE_MAP[ty][tx] == 0


def _distance(a: Player, b: Player) -> float:
    """Distance in raw 3D world units (raw_x / raw_z reported by Three.js)."""
    return sqrt((a.raw_x - b.raw_x) ** 2 + (a.raw_z - b.raw_z) ** 2)


def _thief_is_behind_police(police: "Player", thief: "Player") -> bool:
    """
    True if the thief is in the BACK half of the police.
    Police forward = (sin(rot), cos(rot)).  If dot(forward, police→thief) < 0,
    the thief is behind the police.
    """
    import math
    fwd_x = math.sin(police.raw_rot)
    fwd_z = math.cos(police.raw_rot)
    dx = thief.raw_x - police.raw_x
    dz = thief.raw_z - police.raw_z
    return (fwd_x * dx + fwd_z * dz) < 0


# ── Room ───────────────────────────────────────────────────────────────────────

class Room:
    def __init__(self, room_id: str, host_id: str, round_time: int = DEFAULT_ROUND_TIME,
                 hide_time: int = HIDE_TIME, total_rounds: int = 1):
        self.room_id      = room_id
        self.host_id      = host_id
        self.phase        = Phase.LOBBY
        self.round_time   = round_time
        self.hide_time    = hide_time
        self.total_rounds = total_rounds
        self.current_round = 1
        self.time_left    = round_time
        self.players:   Dict[str, Player] = {}
        self.police_id: Optional[str]     = None
        self.winner:    Optional[str]     = None   # "police" | "thieves"
        self.first_caught_id: Optional[str] = None  # Rule 6: first thief caught this round
        self.scores:    Dict[str, int]    = {}      # player_id → total score
        self._timer_task: Optional[asyncio.Task]   = None
        self._tick_task:  Optional[asyncio.Task]   = None

    def award_round_points(self):
        """Award points for the just-finished round and return per-player score snapshot."""
        if self.winner == "police":
            # Police gets 2 pts; uncaught thieves get 0; caught thieves get 0
            police = self.players.get(self.police_id)
            if police:
                self.scores[police.id] = self.scores.get(police.id, 0) + 2
        elif self.winner == "thieves":
            # Each alive thief gets 2 pts; the thief who caught police gets +1 bonus
            for p in self.players.values():
                if p.role == Role.THIEF and p.alive:
                    self.scores[p.id] = self.scores.get(p.id, 0) + 2
        # Everyone who participated gets 1 participation point
        for p in self.players.values():
            self.scores.setdefault(p.id, 0)
        return self.score_snapshot()

    def score_snapshot(self) -> list:
        return sorted([
            {"id": p.id, "name": p.name, "score": self.scores.get(p.id, 0)}
            for p in self.players.values()
        ], key=lambda x: -x["score"])

    # ── Player management ──────────────────────────────────────────────────

    def add_player(self, pid: str, name: str, ws) -> Player:
        p = Player(id=pid, name=name, ws=ws)
        self.players[pid] = p
        return p

    def remove_player(self, pid: str):
        self.players.pop(pid, None)

    def connected_players(self) -> List[Player]:
        return [p for p in self.players.values() if p.is_connected]

    # ── Game lifecycle ─────────────────────────────────────────────────────

    def assign_roles(self, next_police_id: Optional[str] = None):
        """Assign roles. If next_police_id given, that player becomes police (Rule 6)."""
        active = self.connected_players()
        if next_police_id and any(p.id == next_police_id for p in active):
            # Put designated police first
            active = sorted(active, key=lambda p: 0 if p.id == next_police_id else 1)
        else:
            random.shuffle(active)
        self.police_id = active[0].id
        active[0].role = Role.POLICE
        active[0].x, active[0].y = _POLICE_SPAWN
        for i, p in enumerate(active[1:]):
            p.role    = Role.THIEF
            p.x, p.y = _THIEF_SPAWNS[i % len(_THIEF_SPAWNS)]
            p.alive   = True

    def start_hiding(self, next_police_id: Optional[str] = None):
        self.phase          = Phase.HIDING
        self.time_left      = self.hide_time
        self.winner         = None
        self.first_caught_id = None
        self.assign_roles(next_police_id)

    def start_active(self):
        self.phase     = Phase.ACTIVE
        self.time_left = self.round_time

    def move_player(self, pid: str, dx: float, dy: float):
        """Move player by (dx, dy); clamp to walkable tiles."""
        p = self.players.get(pid)
        if not p or not p.alive:
            return
        # Police cannot move during hiding phase
        if self.phase == Phase.HIDING and p.role == Role.POLICE:
            return

        speed = 0.15
        nx = p.x + dx * speed
        ny = p.y + dy * speed
        if _is_walkable(nx, p.y):
            p.x = nx
        if _is_walkable(p.x, ny):
            p.y = ny

    def tick_captures(self) -> Optional[str]:
        """Called each game tick. Returns winner or None."""
        if self.phase != Phase.ACTIVE:
            return None

        police = self.players.get(self.police_id)
        if not police:
            return None

        alive_thieves = [
            p for p in self.players.values()
            if p.role == Role.THIEF and p.alive
        ]

        # Rule 5: Thief touches police from BEHIND → thieves win (sneak attack).
        # Thief in FRONT of police → police can see them → fall through to police-capture.
        for thief in alive_thieves:
            if _distance(police, thief) < COUNTER_RADIUS:
                if _thief_is_behind_police(police, thief):
                    self.winner = "thieves"
                    self.phase  = Phase.ENDED
                    return "thieves"

        # Police catches thieves (any direction)
        for thief in alive_thieves:
            if _distance(police, thief) < CAPTURE_RADIUS:
                thief.alive = False
                # Rule 6: track first thief caught this round
                if self.first_caught_id is None:
                    self.first_caught_id = thief.id

        remaining = [p for p in self.players.values() if p.role == Role.THIEF and p.alive]
        if not remaining:
            self.winner = "police"
            self.phase  = Phase.ENDED
            return "police"

        return None

    def public_state(self) -> dict:
        return {
            "type":          "state",
            "room_id":       self.room_id,
            "host_id":       self.host_id,
            "phase":         self.phase,
            "time_left":     self.time_left,
            "police_id":     self.police_id,
            "winner":        self.winner,
            "players":       [p.to_dict() for p in self.players.values()],
            "scores":        self.score_snapshot(),
            "settings": {
                "hide_time":     self.hide_time,
                "round_time":    self.round_time,
                "total_rounds":  self.total_rounds,
                "current_round": self.current_round,
            },
        }

    def cancel_tasks(self):
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        if self._tick_task and not self._tick_task.done():
            self._tick_task.cancel()
        self._timer_task = None
        self._tick_task  = None


# ── Room registry ──────────────────────────────────────────────────────────────

class PTRoomManager:
    def __init__(self):
        self._rooms: Dict[str, Room] = {}

    def create(self, room_id: str, host_id: str,
               round_time: int = DEFAULT_ROUND_TIME,
               hide_time: int = HIDE_TIME,
               total_rounds: int = 1) -> Room:
        room = Room(room_id, host_id, round_time=round_time,
                    hide_time=hide_time, total_rounds=total_rounds)
        self._rooms[room_id] = room
        return room

    def get(self, room_id: str) -> Optional[Room]:
        return self._rooms.get(room_id)

    def delete(self, room_id: str):
        room = self._rooms.pop(room_id, None)
        if room:
            room.cancel_tasks()

    def all_rooms(self) -> List[dict]:
        return [
            {
                "room_id":      r.room_id,
                "phase":        r.phase,
                "player_count": len(r.connected_players()),
            }
            for r in self._rooms.values()
        ]


pt_rooms = PTRoomManager()
