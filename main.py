import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from room_manager import RoomManager

load_dotenv()

ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app = FastAPI(title="Drawing Together — WebSocket Server")

# allow_credentials=True is incompatible with allow_origins=["*"] per the CORS
# spec — browsers reject the combination. Only enable credentials when a
# specific origin list is provided.
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
# Room info endpoint (optional utility)
# ---------------------------------------------------------------------------

@app.get("/room/{room_id}/info")
def room_info(room_id: str):
    if not manager.room_exists(room_id):
        return {"room_id": room_id, "players": 0, "full": False}
    count = manager.player_count(room_id)
    return {"room_id": room_id, "players": count, "full": manager.is_full(room_id)}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    # Origin check — skip if wildcard "*" is in the allowed list
    if "*" not in ALLOWED_ORIGINS:
        origin = websocket.headers.get("origin")
        if origin and origin not in ALLOWED_ORIGINS:
            print(f"WebSocket rejected: origin '{origin}' not in {ALLOWED_ORIGINS}")
            await websocket.close(code=1008)
            return

    # Reject if room is full before accepting the connection
    if manager.is_full(room_id):
        await websocket.close(code=4000)
        return

    await websocket.accept()

    joined = manager.connect(room_id, websocket)
    if not joined:
        # Race condition — room filled between the check and accept
        await websocket.send_text(
            json.dumps({"type": "error", "message": "Room is full"})
        )
        await websocket.close(code=4000)
        return

    player_number = manager.player_count(room_id)

    # Notify the joining player of their number
    await websocket.send_text(
        json.dumps({"type": "connected", "player": player_number})
    )

    # Notify the other player that a partner joined
    await manager.broadcast(
        room_id,
        json.dumps({"type": "player_joined", "players": player_number}),
        sender=websocket,
    )

    try:
        while True:
            raw = await websocket.receive_text()

            # Validate that the message is proper JSON
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid message format"})
                )
                continue

            # Validate that required fields are present for draw events
            event_type = data.get("type")

            if event_type == "draw":
                if "x" not in data or "y" not in data:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "Missing draw fields"})
                    )
                    continue

                try:
                    validated: dict = {
                        "type": "draw",
                        "x": float(data["x"]),
                        "y": float(data["y"]),
                    }

                    # Optional brush fields
                    if "fromX" in data:
                        validated["fromX"] = float(data["fromX"])
                    if "fromY" in data:
                        validated["fromY"] = float(data["fromY"])
                    if "color" in data:
                        validated["color"] = str(data["color"])[:20]
                    if "size" in data:
                        size = float(data["size"])
                        if size < 1 or size > 100:
                            raise ValueError("size out of range")
                        validated["size"] = size
                    if "tool" in data:
                        valid_tools = {
                            "pen", "neon", "rainbow", "spray", "mirror",
                            "glitter", "fill", "star", "heart", "circle", "eraser",
                        }
                        tool = str(data["tool"])
                        if tool in valid_tools:
                            validated["tool"] = tool
                    if "hue" in data:
                        validated["hue"] = float(data["hue"]) % 360
                    if "lineStart" in data:
                        validated["lineStart"] = bool(data["lineStart"])

                except (ValueError, TypeError):
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "Invalid draw data"})
                    )
                    continue

                await manager.broadcast(room_id, json.dumps(validated), sender=websocket)

            elif event_type == "fill":
                try:
                    validated = {
                        "type": "fill",
                        "x": float(data["x"]),
                        "y": float(data["y"]),
                        "color": str(data.get("color", "#000000"))[:20],
                    }
                except (ValueError, TypeError, KeyError):
                    continue
                await manager.broadcast(room_id, json.dumps(validated), sender=websocket)

            elif event_type == "stamp":
                try:
                    valid_stamp_tools = {"star", "heart", "circle"}
                    stamp_tool = str(data.get("tool", ""))
                    if stamp_tool not in valid_stamp_tools:
                        continue
                    stamp_size = float(data.get("size", 4))
                    stamp_size = max(1.0, min(stamp_size, 100.0))
                    validated = {
                        "type": "stamp",
                        "x": float(data["x"]),
                        "y": float(data["y"]),
                        "color": str(data.get("color", "#000000"))[:20],
                        "size": stamp_size,
                        "tool": stamp_tool,
                    }
                except (ValueError, TypeError, KeyError):
                    continue
                await manager.broadcast(room_id, json.dumps(validated), sender=websocket)

            elif event_type == "end":
                await manager.broadcast(room_id, json.dumps({"type": "end"}), sender=websocket)

            elif event_type == "clear":
                await manager.broadcast(
                    room_id, json.dumps({"type": "clear"}), sender=websocket
                )

            # Unknown event types are silently ignored

    except WebSocketDisconnect:
        manager.disconnect(room_id, websocket)

        remaining = manager.player_count(room_id)
        if remaining > 0:
            await manager.broadcast(
                room_id,
                json.dumps({"type": "player_left", "players": remaining}),
                sender=websocket,
            )


# ---------------------------------------------------------------------------
# Server startup (when run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # The hosting platform (Render, Heroku, etc.) typically sets PORT and maybe HOST
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("UVICORN_PORT", "8000")))
    # uvicorn may not be imported until runtime to avoid affecting FastAPI imports
    import uvicorn

    uvicorn.run("main:app", host=host, port=port)
