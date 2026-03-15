"""
Police & Thieves – FastAPI router
WebSocket: /police-thieves/ws/{room_id}/{player_id}
HTTP:
  GET  /police-thieves/rooms          – list rooms
  POST /police-thieves/rooms          – create room  ?host_name=…&round_time=…
  GET  /police-thieves/rooms/{room_id} – room info
"""

import asyncio
import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import Dict

from game import pt_rooms, Phase, HIDE_TIME, MAX_PLAYERS

router = APIRouter(prefix="/police-thieves", tags=["police-thieves"])

# room_id → { player_id → WebSocket }
_ws_map: Dict[str, Dict[str, WebSocket]] = {}


# ── Broadcast helpers ─────────────────────────────────────────────────────────

async def _send(ws: WebSocket, msg: dict):
    try:
        await ws.send_json(msg)
    except Exception:
        pass


async def _broadcast(room_id: str, msg: dict):
    for ws in list(_ws_map.get(room_id, {}).values()):
        await _send(ws, msg)


async def _broadcast_state(room_id: str):
    room = pt_rooms.get(room_id)
    if not room:
        return
    state = room.public_state()
    await _broadcast(room_id, state)


# ── Game loop helpers ─────────────────────────────────────────────────────────

async def _run_hide_timer(room_id: str):
    """Count down hiding phase, then start active phase."""
    room = pt_rooms.get(room_id)
    if not room:
        return
    try:
        while room.time_left > 0 and room.phase == Phase.HIDING:
            await asyncio.sleep(1)
            room.time_left -= 1
            await _broadcast(room_id, {"type": "timer", "seconds": room.time_left, "phase": room.phase})
        if room.phase == Phase.HIDING:
            room.start_active()
            await _broadcast_state(room_id)
            room._timer_task = asyncio.create_task(_run_round_timer(room_id))
            room._tick_task  = asyncio.create_task(_run_tick(room_id))
    except asyncio.CancelledError:
        pass


async def _finish_round(room_id: str, winner: str, reason: str):
    """Award points, broadcast round_over, then auto-start next round or end game."""
    room = pt_rooms.get(room_id)
    if not room:
        return
    scores = room.award_round_points()
    more_rounds = room.current_round < room.total_rounds
    await _broadcast(room_id, {
        "type":          "round_over",
        "winner":        winner,
        "reason":        reason,
        "round":         room.current_round,
        "total_rounds":  room.total_rounds,
        "scores":        scores,
        "more_rounds":   more_rounds,
        "next_in":       7 if more_rounds else 0,  # seconds until next round
    })
    if more_rounds:
        # Countdown to next round, then auto-start
        for i in range(7, 0, -1):
            await asyncio.sleep(1)
            await _broadcast(room_id, {"type": "next_round_countdown", "seconds": i - 1})
        room.current_round += 1
        # Rule 6: determine next police
        next_police_id = getattr(room, "_next_police_id", None)
        room._next_police_id = None
        room.cancel_tasks()
        room.start_hiding(next_police_id)
        await _broadcast_state(room_id)
        room._timer_task = asyncio.create_task(_run_hide_timer(room_id))
    else:
        # All rounds done — final game over
        await _broadcast(room_id, {"type": "game_over", "winner": winner, "reason": reason,
                                   "final": True, "scores": scores})
        await _broadcast_state(room_id)


async def _run_round_timer(room_id: str):
    """Count down active phase; thieves win on timeout. round_time==0 means no limit."""
    room = pt_rooms.get(room_id)
    if not room:
        return
    try:
        if room.round_time == 0:
            # No time limit — just wait until the phase ends via captures
            while room.phase == Phase.ACTIVE:
                await asyncio.sleep(1)
            return
        while room.time_left > 0 and room.phase == Phase.ACTIVE:
            await asyncio.sleep(1)
            room.time_left -= 1
            await _broadcast(room_id, {"type": "timer", "seconds": room.time_left, "phase": room.phase})
        if room.phase == Phase.ACTIVE:
            room.winner = "thieves"
            room.phase  = Phase.ENDED
            asyncio.create_task(_finish_round(room_id, "thieves", "timeout"))
    except asyncio.CancelledError:
        pass


async def _run_tick(room_id: str):
    """Collision / capture tick loop at ~20 TPS."""
    room = pt_rooms.get(room_id)
    if not room:
        return
    try:
        while room.phase == Phase.ACTIVE:
            await asyncio.sleep(0.05)
            winner = room.tick_captures()
            if winner:
                reason = "all_caught" if winner == "police" else "police_caught"
                asyncio.create_task(_finish_round(room_id, winner, reason))
                break
            # Send lightweight position-only update every tick
            positions = [
                {
                    "id":    p.id,
                    "x":     round(p.raw_x, 1),
                    "z":     round(p.raw_z, 1),
                    "rotY":  round(p.raw_rot, 3),
                    "alive": p.alive,
                }
                for p in room.players.values()
            ]
            await _broadcast(room_id, {"type": "positions", "players": positions})
    except asyncio.CancelledError:
        pass


# ── HTTP endpoints ─────────────────────────────────────────────────────────────

@router.get("/rooms")
def list_rooms():
    return pt_rooms.all_rooms()


@router.post("/rooms")
def create_room(
    host_name:    str = Query("Host"),
    round_time:   int = Query(120, ge=0, le=600),
    hide_time:    int = Query(30,  ge=10, le=120),
    total_rounds: int = Query(1,   ge=1,  le=10),
):
    room_id = str(uuid.uuid4())[:8].upper()
    host_id = str(uuid.uuid4())
    room    = pt_rooms.create(room_id, host_id,
                              round_time=round_time,
                              hide_time=hide_time,
                              total_rounds=total_rounds)
    room.add_player(host_id, host_name, ws=None)
    return {"room_id": room_id, "player_id": host_id}


@router.get("/rooms/{room_id}")
def get_room(room_id: str):
    room = pt_rooms.get(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room.public_state()


# ── WebSocket ─────────────────────────────────────────────────────────────────

@router.websocket("/ws/{room_id}")
async def ws_endpoint(ws: WebSocket, room_id: str):
    """
    Client connects, then sends a JSON join message:
      { "type": "join", "player_id": "...", "name": "..." }
    player_id is the one returned by POST /police-thieves/rooms (host)
    or a new uuid4 for other players.
    """
    room = pt_rooms.get(room_id)

    # Accept the WebSocket first (required before any close/send)
    await ws.accept()

    if not room:
        await ws.close(code=4004)
        return

    # Wait for join message
    try:
        raw  = await ws.receive_text()
        data = json.loads(raw)
    except Exception:
        await ws.close(code=4002)
        return

    if data.get("type") != "join":
        await ws.close(code=4002)
        return

    player_id = str(data.get("player_id") or uuid.uuid4())
    name      = str(data.get("name", "Player"))[:40].strip() or "Player"

    # Reconnect or new player
    player = room.players.get(player_id)
    if player:
        player.ws           = ws
        player.is_connected = True
    else:
        if room.phase != Phase.LOBBY:
            await _send(ws, {"type": "error", "message": "Game already in progress."})
            await ws.close(code=4003)
            return
        if len(room.players) >= MAX_PLAYERS:
            await _send(ws, {"type": "error", "message": f"Room is full (max {MAX_PLAYERS} players)."})
            await ws.close(code=4003)
            return
        player = room.add_player(player_id, name, ws)

    _ws_map.setdefault(room_id, {})[player_id] = ws

    await _send(ws, {"type": "connected", "player_id": player_id, "room_id": room_id})
    await _broadcast_state(room_id)

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")

            # ── Start game (host only) ────────────────────────────────────────
            if mtype == "start_game":
                if player_id != room.host_id:
                    await _send(ws, {"type": "error", "message": "Only the host can start."})
                    continue
                if room.phase != Phase.LOBBY:
                    await _send(ws, {"type": "error", "message": "Game already started."})
                    continue
                if len(room.connected_players()) < 2:
                    await _send(ws, {"type": "error", "message": "Need at least 2 players."})
                    continue

                room.cancel_tasks()
                room.current_round = 1
                room.scores = {}
                room.start_hiding(getattr(room, "_next_police_id", None))
                room._next_police_id = None
                await _broadcast_state(room_id)
                room._timer_task = asyncio.create_task(_run_hide_timer(room_id))

            # ── Player movement (tile-based, keyboard/dpad) ───────────────────────
            elif mtype == "move":
                try:
                    dx = float(msg.get("dx", 0))
                    dy = float(msg.get("dy", 0))
                except (ValueError, TypeError):
                    continue
                dx = max(-1.0, min(1.0, dx))
                dy = max(-1.0, min(1.0, dy))
                room.move_player(player_id, dx, dy)

            # ── Position update from 3-D client ──────────────────────────────
            elif mtype == "position_update":
                try:
                    raw_x   = float(msg.get("x", 0))
                    raw_z   = float(msg.get("z", 0))
                    raw_rot = float(msg.get("rotY", 0))
                except (ValueError, TypeError):
                    continue
                p = room.players.get(player_id)
                if p and p.alive and room.phase in (Phase.HIDING, Phase.ACTIVE):
                    if room.phase == Phase.HIDING and p.role and p.role.value == "police":
                        pass  # police frozen during hiding (Rule 4)
                    else:
                        p.raw_x   = raw_x
                        p.raw_z   = raw_z
                        p.raw_rot = raw_rot
                        # Convert to tile coords for server-side capture detection
                        p.x = max(0.5, min(19.5, raw_x / 11.0 + 10.0))
                        p.y = max(0.5, min(19.5, raw_z / 11.0 + 10.0))
                        await _broadcast(room_id, {
                            "type":      "player_moved",
                            "player_id": player_id,
                            "x":         round(raw_x, 1),
                            "z":         round(raw_z, 1),
                            "rotY":      round(raw_rot, 3),
                            "alive":     p.alive,
                        })
                # Positions are broadcast by the tick loop; no extra send needed

            # ── Restart (host only, after ENDED) ──────────────────────────────
            elif mtype == "restart":
                if player_id != room.host_id:
                    continue
                if room.phase != Phase.ENDED:
                    continue
                room.cancel_tasks()
                # Rule 6: determine next police based on who won last round
                # Police won → first caught thief becomes next police
                # Thieves won → current police stays as police
                if room.winner == "police" and room.first_caught_id and room.first_caught_id in room.players:
                    room._next_police_id = room.first_caught_id
                elif room.winner == "thieves" and room.police_id and room.police_id in room.players:
                    room._next_police_id = room.police_id
                else:
                    room._next_police_id = None  # fall back to random
                room.phase     = Phase.LOBBY
                room.winner    = None
                room.police_id = None
                for p in room.players.values():
                    p.role  = None  # type: ignore[assignment]
                    p.alive = True
                await _broadcast_state(room_id)

            # ── Hint (thief only) — broadcast position to all ─────────────────
            elif mtype == "hint":
                p = room.players.get(player_id)
                if p and p.role and p.role.value == "thief" and room.phase in (Phase.HIDING, Phase.ACTIVE):
                    try:
                        hx = float(msg.get("x", p.raw_x))
                        hz = float(msg.get("z", p.raw_z))
                    except (ValueError, TypeError):
                        hx, hz = p.raw_x, p.raw_z
                    await _broadcast(room_id, {
                        "type":      "hint",
                        "player_id": player_id,
                        "name":      p.name,
                        "x":         round(hx, 1),
                        "z":         round(hz, 1),
                    })

            # ── Chat ──────────────────────────────────────────────────────────
            elif mtype == "chat":
                text = str(msg.get("text", "")).strip()[:200]
                if text:
                    await _broadcast(room_id, {
                        "type":       "chat",
                        "player_id":  player_id,
                        "name":       player.name,
                        "text":       text,
                    })

    except WebSocketDisconnect:
        _ws_map.get(room_id, {}).pop(player_id, None)
        if player_id in room.players:
            room.players[player_id].is_connected = False
            room.players[player_id].ws = None

        await _broadcast(room_id, {"type": "player_left", "player_id": player_id})

        # Only delete room if it's still in lobby with nobody left.
        # During an active/hiding game, keep the room alive so players can
        # reconnect (they disconnect transiently when navigating lobby → game).
        if not room.connected_players():
            if room.phase == Phase.LOBBY:
                pt_rooms.delete(room_id)
                _ws_map.pop(room_id, None)
            # else: room survives; players will reconnect from the game page
        else:
            await _broadcast_state(room_id)
