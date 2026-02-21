"""Parse natural language commands into structured robot commands.

Uses fast rule-based parsing for common patterns, with LLM fallback via Ollama
for ambiguous commands.
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import ollama


log = logging.getLogger(__name__)

# Direction keywords that signal locomotion (not navigation to an object)
_DIRECTIONS = {'forward', 'forwards', 'backward', 'backwards', 'back',
               'left', 'right', 'sideways'}

# Filler words/phrases Whisper commonly adds
_FILLER_RE = re.compile(
    r'\b(please|can you|could you|would you|i want you to|i need you to'
    r'|go ahead and|just|okay|hey|robot|reachy)\b', re.I)

# --- Stop / halt / freeze ---
_STOP_RE = re.compile(
    r'^(?:stop|halt|freeze|stay|don\'?t move|wait)$', re.I)

# --- Locomotion: turn/rotate/spin <dir> [angle] ---
_TURN_RE = re.compile(
    r'(?:turn|rotate|spin)\s+(left|right)'
    r'(?:\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:degrees?|deg)?)?$', re.I)

# --- Locomotion: move/turn/spin around (180 degree turn) ---
_AROUND_RE = re.compile(
    r'(?:move|go|turn|spin)\s+around$', re.I)

# --- Locomotion: move/go/shift/step/walk forward/back/left/right [distance] ---
_MOVE_DIR_RE = re.compile(
    r'(?:move|go|shift|step|walk|slide|strafe)\s+(?:to\s+the\s+)?'
    r'(forward|forwards|backward|backwards|back|left|right)'
    r'(?:\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(centimeters?|cm|meters?|m))?$', re.I)

# --- Navigation: go to / navigate to / move to / ... <object> ---
_NAV_RE = re.compile(
    r'(?:go\s+to|navigate\s+to|move\s+to|head\s+to|walk\s+to|drive\s+to'
    r'|come\s+to|take\s+me\s+to|bring\s+me\s+to|go\s+near|approach'
    r'|find|go\s+towards|head\s+towards|walk\s+towards)\s+(?:the\s+)?(.+)$', re.I)

# --- Manipulation: <action> [the] <object> ---
_MANIP_RE = re.compile(
    r'(pick\s+up|grab|take|catch|fetch|get|bring|hold|lift|raise|lower'
    r'|press|click|push|pull|tap|touch|poke|squeeze'
    r'|put|place|set|drop|release|throw|toss'
    r'|open|close|shut|lock|unlock'
    r'|turn\s+on|turn\s+off|switch\s+on|switch\s+off|toggle'
    r'|flip|twist|rotate|screw|unscrew'
    r'|hand\s+me|give\s+me|pass\s+me|show\s+me'
    r'|point\s+at|point\s+to|look\s+at'
    r'|wipe|clean|sweep|pour|fill|empty|stir|mix'
    r'|plug\s+in|unplug|connect|disconnect)\s+(?:the\s+)?(.+)$', re.I)

SYSTEM_PROMPT = """You parse spoken robot commands into structured JSON.

Return a JSON object with these fields:
- "category": one of "navigation", "manipulation", or "locomotion"
- "object": the physical thing being acted on (no articles). Empty string for locomotion.
- "instruction": the action to perform (verb + modifiers, no filler)
- "dx": forward displacement in metres (only for locomotion, 0.0 otherwise)
- "dy": leftward displacement in metres (only for locomotion, 0.0 otherwise)
- "dyaw": counter-clockwise rotation in degrees (only for locomotion, 0.0 otherwise)

Category rules:
- "navigation": the robot should move its base to reach a named object or location (e.g. "go to the bottle", "navigate to the table")
- "manipulation": the robot should use its arm to interact with an object (e.g. "pick up the cup", "grab the ball", "put the box on the shelf", "press the button")
- "locomotion": the robot should move its base by a relative amount without a target object (e.g. "move left 10cm", "go forward 50cm", "turn right 90 degrees")

For locomotion, convert distances to metres and directions to the robot body frame:
- forward = +dx, backward = -dx
- left = +dy, right = -dy
- turn left = +dyaw, turn right = -dyaw

Examples:
User: "go to the bottle"
{"category": "navigation", "object": "bottle", "instruction": "go to", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "pick up the red cup"
{"category": "manipulation", "object": "red cup", "instruction": "pick up", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "grab the ball from the table"
{"category": "manipulation", "object": "ball", "instruction": "grab from the table", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "navigate to the chair"
{"category": "navigation", "object": "chair", "instruction": "navigate to", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "move forward 50 centimeters"
{"category": "locomotion", "object": "", "instruction": "move forward 50cm", "dx": 0.5, "dy": 0.0, "dyaw": 0.0}

User: "go left by 10 centimeters"
{"category": "locomotion", "object": "", "instruction": "move left 10cm", "dx": 0.0, "dy": 0.1, "dyaw": 0.0}

User: "move to the right 30 centimeters"
{"category": "locomotion", "object": "", "instruction": "move right 30cm", "dx": 0.0, "dy": -0.3, "dyaw": 0.0}

User: "turn left 90 degrees"
{"category": "locomotion", "object": "", "instruction": "turn left 90 degrees", "dx": 0.0, "dy": 0.0, "dyaw": 90.0}

User: "rotate right 45 degrees"
{"category": "locomotion", "object": "", "instruction": "turn right 45 degrees", "dx": 0.0, "dy": 0.0, "dyaw": -45.0}

User: "go back 20 centimeters"
{"category": "locomotion", "object": "", "instruction": "move backward 20cm", "dx": -0.2, "dy": 0.0, "dyaw": 0.0}

User: "put the bottle on the shelf"
{"category": "manipulation", "object": "bottle", "instruction": "put on the shelf", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "press the light switch"
{"category": "manipulation", "object": "light switch", "instruction": "press", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}"""


@dataclass
class ParsedCommand:
    category: str  # "navigation", "manipulation", or "locomotion"
    object: str
    instruction: str
    dx: float = 0.0
    dy: float = 0.0
    dyaw: float = 0.0  # degrees from LLM, converted to radians in property

    @property
    def dyaw_rad(self) -> float:
        """Yaw delta in radians."""
        return math.radians(self.dyaw)


class CommandParser:
    """Extracts structured commands from spoken text.

    Tries fast rule-based parsing first, falls back to Ollama LLM.
    """

    def __init__(self, model: str = 'llama3.2'):
        self._model = model

    # ------------------------------------------------------------------ #
    #  Rule-based (instant)
    # ------------------------------------------------------------------ #
    def _try_rules(self, text: str) -> Optional[ParsedCommand]:
        t = text.strip()
        # Strip trailing punctuation (Whisper often adds periods)
        t = t.rstrip('.,!?;:')
        # Strip filler words/phrases
        t = _FILLER_RE.sub('', t).strip()
        t = re.sub(r'\s+', ' ', t)

        if not t:
            return None

        # --- Stop / halt / freeze ---
        if _STOP_RE.match(t):
            return ParsedCommand('locomotion', '', 'stop')

        # --- Turn / rotate / spin ---
        m = _TURN_RE.search(t)
        if m:
            direction = m.group(1).lower()
            angle = float(m.group(2)) if m.group(2) else 90.0
            dyaw = angle if direction == 'left' else -angle
            return ParsedCommand('locomotion', '',
                                 f'turn {direction} {angle} degrees', dyaw=dyaw)

        # --- Move / turn around (180 deg) ---
        if _AROUND_RE.search(t):
            return ParsedCommand('locomotion', '',
                                 'turn around 180 degrees', dyaw=180.0)

        # --- Move/go/shift/step any direction [distance] ---
        m = _MOVE_DIR_RE.search(t)
        if m:
            direction = m.group(1).lower()
            dist_val = float(m.group(2)) if m.group(2) else 0.5
            unit = (m.group(3) or 'm').lower()
            if unit.startswith('c'):
                dist_val /= 100.0
            dx, dy = 0.0, 0.0
            if direction in ('forward', 'forwards'):
                dx = dist_val
            elif direction in ('backward', 'backwards', 'back'):
                dx = -dist_val
            elif direction == 'left':
                dy = dist_val
            elif direction == 'right':
                dy = -dist_val
            return ParsedCommand('locomotion', '',
                                 f'move {direction}', dx=dx, dy=dy)

        # --- Manipulation (check before navigation so "take the bottle"
        #     doesn't match "take me to") ---
        m = _MANIP_RE.search(t)
        if m:
            action = m.group(1).strip()
            obj = m.group(2).strip()
            # Guard: "bring me to X" / "take me to X" is navigation, not manip
            if re.match(r'me\s+to\b', obj, re.I):
                pass  # fall through to navigation
            else:
                return ParsedCommand('manipulation', obj, f'{action} {obj}')

        # --- Navigation ---
        m = _NAV_RE.search(t)
        if m:
            target = m.group(1).strip()
            first_word = target.split()[0].lower()
            if first_word not in _DIRECTIONS:
                return ParsedCommand('navigation', target, 'go to')

        return None  # no rule matched

    # ------------------------------------------------------------------ #
    #  LLM fallback (slow)
    # ------------------------------------------------------------------ #
    def _parse_llm(self, text: str) -> Optional[ParsedCommand]:
        try:
            response = ollama.chat(
                model=self._model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': text.strip()},
                ],
                format='json',
                options={'temperature': 0},
            )
            content = response['message']['content'].strip()
            log.info('Ollama raw response: %s', content)

            data = json.loads(content)
            category = data.get('category', '').strip().lower()
            obj = data.get('object', '').strip()
            instr = data.get('instruction', '').strip()

            if category not in ('navigation', 'manipulation', 'locomotion'):
                log.warning('Unknown category "%s": %s', category, data)
                return None

            if category != 'locomotion' and (not obj or not instr):
                log.warning('Empty object/instruction for %s: %s', category, data)
                return None

            return ParsedCommand(
                category=category,
                object=obj,
                instruction=instr,
                dx=float(data.get('dx', 0.0)),
                dy=float(data.get('dy', 0.0)),
                dyaw=float(data.get('dyaw', 0.0)),
            )

        except Exception as e:
            log.error('Command parser (LLM) failed: %s', e)
            return None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def parse(self, text: str) -> Optional[ParsedCommand]:
        """Parse a command string. Rules first, LLM fallback."""
        result = self._try_rules(text)
        if result is not None:
            log.info('Rule-based parse: %s', result)
            return result

        log.info('No rule matched, falling back to LLM for: "%s"', text)
        return self._parse_llm(text)
