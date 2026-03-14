import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from room_manager import RoomManager

# Police-Thieves mode (folder name has a hyphen, so we add it to the path)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "police-thieves"))
from router import router as pt_router  # noqa: E402

load_dotenv()

ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app = FastAPI(title="Drawing Together — WebSocket Server")
app.include_router(pt_router)

_allow_credentials = "*" not in ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = RoomManager()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
@app.head("/")
def health_check():
    return {"status": "ok", "message": "Drawing Together backend is running"}


# ---------------------------------------------------------------------------
# Room info endpoint
# ---------------------------------------------------------------------------

@app.get("/room/{room_id}/info")
def room_info(room_id: str):
    if not manager.room_exists(room_id):
        return {"room_id": room_id, "players": 0, "full": False}
    count = manager.player_count(room_id)
    return {"room_id": room_id, "players": count, "full": manager.is_full(room_id)}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

VALID_DRAW_TOOLS = {
    "pen", "neon", "rainbow", "spray", "mirror",
    "glitter", "chalk", "fire", "bubble", "zigzag",
    "kaleidoscope", "lightning", "fur", "splatter",
    "ribbon", "confetti", "watercolor", "mosaic",
    "fill", "star", "heart", "circle", "eraser",
}


def _validate_draw(data: dict) -> dict | None:
    """Return a validated draw dict or None on bad data."""
    if "x" not in data or "y" not in data:
        return None
    try:
        out: dict = {"type": "draw", "x": float(data["x"]), "y": float(data["y"])}
        if "fromX" in data:
            out["fromX"] = float(data["fromX"])
        if "fromY" in data:
            out["fromY"] = float(data["fromY"])
        if "color" in data:
            out["color"] = str(data["color"])[:20]
        if "size" in data:
            s = float(data["size"])
            if s < 1 or s > 100:
                return None
            out["size"] = s
        if "tool" in data:
            t = str(data["tool"])
            if t in VALID_DRAW_TOOLS:
                out["tool"] = t
        if "hue" in data:
            out["hue"] = float(data["hue"]) % 360
        if "lineStart" in data:
            out["lineStart"] = bool(data["lineStart"])
        return out
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    # Origin check
    if "*" not in ALLOWED_ORIGINS:
        origin = websocket.headers.get("origin")
        if origin and origin not in ALLOWED_ORIGINS:
            await websocket.close(code=1008)
            return

    await websocket.accept()

    # Wait for first message to get player name
    try:
        first_raw = await websocket.receive_text()
        first_data = json.loads(first_raw)
        player_name = str(first_data.get("name", "Player"))[:40].strip() or "Player"
    except Exception:
        player_name = "Player"

    # Check if this is a reconnection (name-based) before capacity check
    # connect() handles reconnection internally — it only fails if truly full
    player = manager.connect(room_id, websocket, player_name)
    if not player:
        await websocket.send_text(json.dumps({"type": "error", "message": "Room is full"}))
        await websocket.close(code=4000)
        return

    # Tell this player their id (may be their old id on reconnect)
    await websocket.send_text(json.dumps({
        "type": "connected",
        "player": player.id,
    }))

    # Send game state sync if a game is in progress (reconnection)
    sync = manager.get_game_sync(room_id, player)
    if sync:
        await websocket.send_text(json.dumps(sync))
    elif room_id in manager.collab_history and manager.collab_history[room_id]:
        # Collab mode reconnect — send saved draw history
        await websocket.send_text(json.dumps({
            "type": "draw_history_sync",
            "history": manager.collab_history[room_id],
        }))

    # Tell everyone (including this player) updated player list
    await _broadcast_player_list(room_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            event_type = data.get("type")

            # ── Drawing relay ─────────────────────────────────────────
            if event_type == "draw":
                validated = _validate_draw(data)
                if not validated:
                    continue
                game = manager.games.get(room_id)
                if game and game.phase == "drawing":
                    game.draw_history.append(validated)
                elif not game:
                    manager.collab_history.setdefault(room_id, []).append(validated)
                await manager.broadcast(room_id, json.dumps(validated), sender=websocket)

            elif event_type == "fill":
                try:
                    msg = {
                        "type": "fill",
                        "x": float(data["x"]),
                        "y": float(data["y"]),
                        "color": str(data.get("color", "#000000"))[:20],
                    }
                except (ValueError, TypeError, KeyError):
                    continue
                game = manager.games.get(room_id)
                if game and game.phase == "drawing":
                    game.draw_history.append(msg)
                elif not game:
                    manager.collab_history.setdefault(room_id, []).append(msg)
                await manager.broadcast(room_id, json.dumps(msg), sender=websocket)

            elif event_type == "end":
                end_msg = {"type": "end"}
                game = manager.games.get(room_id)
                if game and game.phase == "drawing":
                    game.draw_history.append(end_msg)
                elif not game:
                    manager.collab_history.setdefault(room_id, []).append(end_msg)
                await manager.broadcast(room_id, json.dumps(end_msg), sender=websocket)

            elif event_type == "clear":
                clear_msg = {"type": "clear"}
                game = manager.games.get(room_id)
                if game and game.phase == "drawing":
                    game.draw_history = [clear_msg]  # clear wipes everything before it
                elif not game:
                    manager.collab_history[room_id] = [clear_msg]
                await manager.broadcast(room_id, json.dumps(clear_msg), sender=websocket)

            # ── Stamp relay ───────────────────────────────────────────
            elif event_type == "stamp":
                try:
                    valid_stamps = {"star", "heart", "circle", "diamond", "triangle", "arrow", "spiral"}
                    st = str(data.get("tool", ""))
                    if st not in valid_stamps:
                        continue
                    msg = {
                        "type": "stamp",
                        "x": float(data["x"]),
                        "y": float(data["y"]),
                        "color": str(data.get("color", "#000000"))[:20],
                        "size": max(1.0, min(float(data.get("size", 4)), 100.0)),
                        "tool": st,
                    }
                except (ValueError, TypeError, KeyError):
                    continue
                game = manager.games.get(room_id)
                if game and game.phase == "drawing":
                    game.draw_history.append(msg)
                elif not game:
                    manager.collab_history.setdefault(room_id, []).append(msg)
                await manager.broadcast(room_id, json.dumps(msg), sender=websocket)

            # ── Start game (from lobby host) ──────────────────────────
            elif event_type == "start_game":
                mode = str(data.get("mode", "guess"))[:10]
                rounds = max(1, min(int(data.get("rounds", 5)), 20))
                time_val = max(30, min(int(data.get("time", 70)), 300))

                # Relay to other lobby clients so they navigate
                relay: dict = {"type": "start_game", "mode": mode, "rounds": rounds, "time": time_val}
                await manager.broadcast(room_id, json.dumps(relay), sender=websocket)

                # Initialize game state
                if mode == "guess":
                    game = manager.get_or_create_game(room_id, rounds, time_val)
                    game.total_rounds = rounds
                    game.round_time = time_val
                    game.current_round = 0
                    game.drawer_index = -1
                    game.phase = "lobby"
                    game.used_words.clear()
                    # Reset scores
                    for p in manager.get_players(room_id):
                        p.score = 0

            # ── Player ready (sent when guess page mounts) ────────────
            elif event_type == "ready":
                game = manager.games.get(room_id)
                # Create game if it doesn't exist (e.g. server restarted)
                if not game:
                    rounds = max(1, min(int(data.get("rounds", 5)), 20))
                    time_val = max(30, min(int(data.get("time", 70)), 300))
                    game = manager.get_or_create_game(room_id, rounds, time_val)

                if game.phase in ("lobby", "roundover") and game.current_round < game.total_rounds:
                    # Mark as "starting" to prevent other ready messages from double-triggering
                    game.phase = "starting"

                    async def _delayed_start(rid: str, g: object) -> None:
                        await asyncio.sleep(2)  # let remaining players connect
                        players = manager.get_players(rid)
                        if len(players) >= 2:
                            await manager.start_round(rid)
                        else:
                            # Not enough players yet — revert so next ready can try
                            g.phase = "lobby"  # type: ignore[attr-defined]

                    asyncio.create_task(_delayed_start(room_id, game))

            # ── Chat / guess ──────────────────────────────────────────
            elif event_type == "chat":
                try:
                    text = str(data.get("text", "")).strip()[:200]
                    if not text:
                        continue
                except (ValueError, TypeError):
                    continue

                me = manager.get_player(room_id, websocket)
                if not me:
                    continue

                # Check if it's a correct guess
                guessed = await manager.check_guess(room_id, me, text)

                if not guessed:
                    # Relay as normal chat (but don't show the exact word to others)
                    game = manager.games.get(room_id)
                    if game and game.phase == "drawing" and text.strip().lower() == game.secret_word.lower():
                        # Close guess — don't broadcast to prevent hints
                        continue
                    chat_msg = {
                        "type": "chat",
                        "text": text,
                        "playerId": me.id,
                        "playerName": me.name,
                    }
                    await manager.broadcast(room_id, json.dumps(chat_msg), sender=websocket)

            # ── Unknown → ignore ──────────────────────────────────────

    except WebSocketDisconnect:
        manager.disconnect(room_id, websocket)
        remaining = manager.player_count(room_id)
        if remaining > 0:
            await _broadcast_player_list(room_id)
            # If the drawer left mid-round, end the round
            game = manager.games.get(room_id)
            if game and game.phase == "drawing":
                players = manager.get_players(room_id)
                if game.drawer_index >= len(players):
                    await manager.end_round(room_id)


async def _broadcast_player_list(room_id: str):
    """Notify all clients of current players with names and scores."""
    players = manager.get_players(room_id)
    player_list = [{"id": p.id, "name": p.name, "score": p.score} for p in players]
    msg = json.dumps({
        "type": "player_joined",
        "players": len([p for p in players if p.ws is not None]),
        "playerList": player_list,
    })
    await manager.broadcast_all(room_id, msg)


# ---------------------------------------------------------------------------
# Server startup (when run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("UVICORN_PORT", "8000")))
    import uvicorn
    uvicorn.run("main:app", host=host, port=port)
