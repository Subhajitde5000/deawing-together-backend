import asyncio
import json
import os
import random

from dotenv import load_dotenv
from fastapi import WebSocket
from typing import Dict, List, Optional

load_dotenv()

MAX_PLAYERS: int = int(os.getenv("MAX_PLAYERS_PER_ROOM", "6"))

# ── Word bank ──────────────────────────────────────────────────────────────
WORD_LIST = [
    "tree", "house", "cat", "dog", "sun", "moon", "star", "bird", "fish",
    "apple", "book", "car", "boat", "cloud", "fire", "flower", "guitar",
    "heart", "kite", "lamp", "lion", "mountain", "ocean", "pizza", "rain",
    "rainbow", "robot", "rocket", "snake", "snowman", "spider", "sword",
    "tiger", "train", "umbrella", "whale", "witch", "wolf", "zebra",
    "balloon", "banana", "bridge", "butterfly", "candle", "castle",
    "cherry", "clock", "compass", "crown", "diamond", "dinosaur",
    "dolphin", "dragon", "drum", "eagle", "elephant", "ghost",
    "glasses", "hammer", "helicopter", "island", "jellyfish", "key",
    "ladder", "lighthouse", "mushroom", "octopus", "panda", "penguin",
    "piano", "pirate", "popcorn", "pumpkin", "satellite", "skateboard",
    "skull", "snowflake", "spaceship", "telescope", "tornado", "treasure",
    "trophy", "turtle", "unicorn", "volcano", "waterfall", "windmill",
]


class PlayerInfo:
    __slots__ = ("ws", "id", "name", "score")

    def __init__(self, ws: Optional[WebSocket], player_id: int, name: str = "Player"):
        self.ws: Optional[WebSocket] = ws
        self.id = player_id
        self.name = name
        self.score = 0


class GameState:
    """Per-room game state for the guess-the-word mode."""

    def __init__(self, total_rounds: int = 5, round_time: int = 70):
        self.total_rounds = total_rounds
        self.round_time = round_time
        self.current_round = 0
        self.phase: str = "lobby"          # lobby | drawing | roundover
        self.drawer_index = -1             # index into room's player list
        self.secret_word = ""
        self.correct_guessers: List[str] = []
        self.timer_task: Optional[asyncio.Task] = None
        self.hint_task: Optional[asyncio.Task] = None
        self.time_left = round_time
        self.revealed_indices: List[int] = []
        self.used_words: List[str] = []
        self.draw_history: List[dict] = []   # recorded draw commands for reconnect replay


class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, List[PlayerInfo]] = {}
        self.games: Dict[str, GameState] = {}
        self.max_players: int = MAX_PLAYERS
        self._next_id: Dict[str, int] = {}   # room_id -> next player id
        self.collab_history: Dict[str, List[dict]] = {}  # room-level draw history for collab mode

    # ── Room helpers ──────────────────────────────────────────────────────

    def room_exists(self, room_id: str) -> bool:
        return room_id in self.rooms

    def is_full(self, room_id: str) -> bool:
        active = [p for p in self.rooms.get(room_id, []) if p.ws is not None]
        return len(active) >= self.max_players

    def connect(self, room_id: str, ws: WebSocket, name: str = "Player") -> Optional[PlayerInfo]:
        if room_id not in self.rooms:
            self.rooms[room_id] = []
            self._next_id[room_id] = 1

        # Check if this is a reconnection (same name, ghost player with ws=None)
        game = self.games.get(room_id)
        if game and game.phase != "lobby":
            for p in self.rooms[room_id]:
                if p.name == name:
                    p.ws = ws
                    return p
        else:
            # Collab mode (no game) — also allow reconnect by name if ghost exists
            for p in self.rooms[room_id]:
                if p.name == name and p.ws is None:
                    p.ws = ws
                    return p

        # New player — check capacity (count only active connections)
        active = [p for p in self.rooms.get(room_id, []) if p.ws is not None]
        if len(active) >= self.max_players:
            return None

        pid = self._next_id[room_id]
        self._next_id[room_id] = pid + 1
        p = PlayerInfo(ws, pid, name)
        self.rooms[room_id].append(p)
        return p

    def disconnect(self, room_id: str, ws: WebSocket):
        if room_id not in self.rooms:
            return
        game = self.games.get(room_id)
        # Keep the player as a ghost (None ws) so they can reconnect.
        # This applies during active games AND collab mode rooms with history.
        keep_ghost = (game and game.phase not in ("lobby",)) or (room_id in self.collab_history)
        if keep_ghost:
            for p in self.rooms[room_id]:
                if p.ws is ws:
                    p.ws = None
                    break
        else:
            self.rooms[room_id] = [p for p in self.rooms[room_id] if p.ws is not ws]

        # If no player has an active connection, clean up
        active = [p for p in self.rooms.get(room_id, []) if p.ws is not None]
        if not active:
            del self.rooms[room_id]
            self._next_id.pop(room_id, None)
            self.collab_history.pop(room_id, None)
            game = self.games.get(room_id)
            if game:
                self._cancel_tasks(game)

    def player_count(self, room_id: str) -> int:
        return len([p for p in self.rooms.get(room_id, []) if p.ws is not None])

    def get_player(self, room_id: str, ws: WebSocket) -> Optional[PlayerInfo]:
        for p in self.rooms.get(room_id, []):
            if p.ws is ws:
                return p
        return None

    def get_players(self, room_id: str) -> List[PlayerInfo]:
        return self.rooms.get(room_id, [])

    # ── Broadcast helpers ────────────────────────────────────────────────

    async def broadcast(self, room_id: str, message: str, sender: Optional[WebSocket] = None):
        for p in self.rooms.get(room_id, []):
            if p.ws is not None and p.ws is not sender:
                try:
                    await p.ws.send_text(message)
                except Exception:
                    pass

    async def broadcast_all(self, room_id: str, message: str):
        """Send to every player including the one who triggered it."""
        for p in self.rooms.get(room_id, []):
            if p.ws is not None:
                try:
                    await p.ws.send_text(message)
                except Exception:
                    pass

    async def send_to(self, ws: Optional[WebSocket], message: str):
        if ws is None:
            return
        try:
            await ws.send_text(message)
        except Exception:
            pass

    # ── Game state helpers ───────────────────────────────────────────────

    def get_game_sync(self, room_id: str, player: PlayerInfo) -> Optional[dict]:
        """Build a full game-state snapshot for a reconnecting player."""
        game = self.games.get(room_id)
        if not game or game.phase == "lobby":
            return None

        players = self.get_players(room_id)
        drawer = players[game.drawer_index] if game.drawer_index < len(players) else None
        is_drawer = drawer is not None and drawer.id == player.id

        msg: dict = {
            "type": "game_state",
            "phase": game.phase,
            "round": game.current_round,
            "totalRounds": game.total_rounds,
            "roundTime": game.round_time,
            "timeLeft": game.time_left,
            "drawer": drawer.id if drawer else None,
            "drawerName": drawer.name if drawer else "?",
            "scores": [{"id": p.id, "name": p.name, "score": p.score} for p in players],
            "correctGuessers": list(game.correct_guessers),
        }

        if game.phase == "drawing":
            if is_drawer:
                msg["word"] = game.secret_word
            else:
                msg["mask"] = self.mask_word(game.secret_word, game.revealed_indices)
            msg["drawHistory"] = game.draw_history

        return msg

    def get_or_create_game(self, room_id: str, rounds: int = 5, time: int = 70) -> GameState:
        if room_id not in self.games:
            self.games[room_id] = GameState(rounds, time)
        return self.games[room_id]

    def pick_word(self, game: GameState) -> str:
        available = [w for w in WORD_LIST if w not in game.used_words]
        if not available:
            available = list(WORD_LIST)
            game.used_words.clear()
        word = random.choice(available)
        game.used_words.append(word)
        return word

    @staticmethod
    def mask_word(word: str, revealed: List[int]) -> str:
        return " ".join(
            c.upper() if i in revealed else "_"
            for i, c in enumerate(word)
            if c != " "
        )

    # ── Round lifecycle ──────────────────────────────────────────────────

    async def start_round(self, room_id: str):
        game = self.games.get(room_id)
        players = self.get_players(room_id)
        if not game or len(players) < 2:
            return
        # Guard against double-start
        if game.phase == "drawing":
            return

        game.current_round += 1
        game.phase = "drawing"
        game.correct_guessers = []
        game.revealed_indices = []
        game.time_left = game.round_time
        game.draw_history = []

        # Rotate drawer
        game.drawer_index = (game.drawer_index + 1) % len(players)
        drawer = players[game.drawer_index]

        # Pick word
        word = self.pick_word(game)
        game.secret_word = word

        # Reveal at least 1 letter from the start so guessers get a hint
        letter_indices = [i for i, c in enumerate(word) if c != " "]
        initial_reveal = random.sample(letter_indices, min(1, len(letter_indices)))
        game.revealed_indices = list(initial_reveal)

        mask = self.mask_word(word, game.revealed_indices)

        # Send round_start — drawer gets the word, guessers get mask
        for p in players:
            msg: dict = {
                "type": "round_start",
                "round": game.current_round,
                "drawer": drawer.id,
                "drawerName": drawer.name,
                "mask": mask,
            }
            if p.ws is drawer.ws:
                msg["word"] = word
            await self.send_to(p.ws, json.dumps(msg))

        # Start timer and hint tasks
        self._cancel_tasks(game)
        game.timer_task = asyncio.create_task(self._run_timer(room_id))
        game.hint_task = asyncio.create_task(self._run_hints(room_id))

    async def end_round(self, room_id: str):
        game = self.games.get(room_id)
        players = self.get_players(room_id)
        if not game:
            return

        self._cancel_tasks(game)
        game.phase = "roundover"

        # Award drawer bonus (5 pts per correct guesser, max 20)
        drawer = players[game.drawer_index] if game.drawer_index < len(players) else None
        if drawer and game.correct_guessers:
            drawer.score += min(len(game.correct_guessers) * 5, 20)

        is_final = game.current_round >= game.total_rounds
        scores = [{"name": p.name, "score": p.score} for p in players]
        await self.broadcast_all(room_id, json.dumps({
            "type": "round_over",
            "word": game.secret_word,
            "correctGuessers": list(game.correct_guessers),
            "scores": scores,
            "isFinal": is_final,
            "round": game.current_round,
            "totalRounds": game.total_rounds,
        }))

        # Auto-start next round after a short delay (no client ready needed)
        if not is_final:
            async def _auto_next(rid: str) -> None:
                await asyncio.sleep(3)
                g = self.games.get(rid)
                if g and g.phase == "roundover":
                    await self.start_round(rid)
            asyncio.create_task(_auto_next(room_id))

    async def check_guess(self, room_id: str, player: PlayerInfo, text: str) -> bool:
        game = self.games.get(room_id)
        if not game or game.phase != "drawing":
            return False

        # Don't let the drawer guess
        players = self.get_players(room_id)
        if game.drawer_index < len(players) and players[game.drawer_index].ws is player.ws:
            return False

        # Already guessed correctly
        if player.name in game.correct_guessers:
            return False

        if text.strip().lower() == game.secret_word.lower():
            # Calculate points — earlier = more points
            remaining_pct = game.time_left / game.round_time if game.round_time else 0
            points = max(5, int(10 + 90 * remaining_pct))

            player.score += points
            game.correct_guessers.append(player.name)

            await self.broadcast_all(room_id, json.dumps({
                "type": "correct_guess",
                "playerName": player.name,
                "playerId": player.id,
                "points": points,
            }))

            # Check if all guessers got it → end round early
            non_drawer_count = len(players) - 1
            if len(game.correct_guessers) >= non_drawer_count:
                await self.end_round(room_id)

            return True
        return False

    # ── Timer ────────────────────────────────────────────────────────────

    async def _run_timer(self, room_id: str):
        game = self.games.get(room_id)
        if not game:
            return
        try:
            while game.time_left > 0 and game.phase == "drawing":
                await asyncio.sleep(1)
                game.time_left -= 1
                await self.broadcast_all(room_id, json.dumps({
                    "type": "timer",
                    "seconds": game.time_left,
                }))
            if game.phase == "drawing":
                await self.end_round(room_id)
        except asyncio.CancelledError:
            pass

    # ── Hint reveals ─────────────────────────────────────────────────────

    async def _run_hints(self, room_id: str):
        game = self.games.get(room_id)
        if not game:
            return
        word = game.secret_word
        letter_indices = [i for i, c in enumerate(word) if c != " "]
        total = len(letter_indices)
        max_reveals = max(1, total // 2)  # reveal up to half the letters
        interval = game.round_time / (max_reveals + 1)

        try:
            for _ in range(max_reveals):
                await asyncio.sleep(interval)
                if game.phase != "drawing":
                    break
                unrevealed = [i for i in letter_indices if i not in game.revealed_indices]
                if not unrevealed:
                    break
                idx = random.choice(unrevealed)
                game.revealed_indices.append(idx)
                mask = self.mask_word(word, game.revealed_indices)
                # Only send hints to guessers, not the drawer
                players = self.get_players(room_id)
                drawer_ws = players[game.drawer_index].ws if game.drawer_index < len(players) else None
                hint_msg = json.dumps({"type": "hint", "mask": mask})
                for p in players:
                    if p.ws is not drawer_ws:
                        await self.send_to(p.ws, hint_msg)
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _cancel_tasks(game: GameState):
        if game.timer_task and not game.timer_task.done():
            game.timer_task.cancel()
        if game.hint_task and not game.hint_task.done():
            game.hint_task.cancel()
        game.timer_task = None
        game.hint_task = None
