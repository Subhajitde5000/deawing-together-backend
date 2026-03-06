from fastapi import WebSocket
from typing import Dict, List


class RoomManager:
    def __init__(self):
        # room_id -> list of connected WebSocket players
        self.rooms: Dict[str, List[WebSocket]] = {}

    def room_exists(self, room_id: str) -> bool:
        return room_id in self.rooms

    def is_full(self, room_id: str) -> bool:
        return len(self.rooms.get(room_id, [])) >= 2

    def connect(self, room_id: str, websocket: WebSocket) -> bool:
        """
        Add a player to a room.
        Returns True on success, False if room is full.
        """
        if self.is_full(room_id):
            return False

        if room_id not in self.rooms:
            self.rooms[room_id] = []

        self.rooms[room_id].append(websocket)
        return True

    def disconnect(self, room_id: str, websocket: WebSocket):
        """Remove a player from a room and clean up if empty."""
        if room_id in self.rooms:
            try:
                self.rooms[room_id].remove(websocket)
            except ValueError:
                pass

            if not self.rooms[room_id]:
                del self.rooms[room_id]

    async def broadcast(self, room_id: str, message: str, sender: WebSocket):
        """Send a message to all players in the room except the sender."""
        for player in self.rooms.get(room_id, []):
            if player is not sender:
                await player.send_text(message)

    def player_count(self, room_id: str) -> int:
        return len(self.rooms.get(room_id, []))
