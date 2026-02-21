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

# Known robot names
_ROBOTS = {'reachy', 'spot'}
_DEFAULT_ROBOT = 'reachy'

# Whisper often mishears "reachy" as these variants
_REACHY_ALIASES = re.compile(
    r'reachy|richie|ritchie|reachie|reechy|richi|reachi|richy|reachy', re.I)
_SPOT_ALIASES = re.compile(r'spot', re.I)

# Robot name prefix: "reachy, go to ..." / "spot go forward" / "tell reachy to ..."
_ROBOT_PREFIX_RE = re.compile(
    r'^(?:tell\s+)?'
    r'(reachy|richie|ritchie|reachie|reechy|richi|reachi|richy|spot)'
    r'[,:]?\s+(?:to\s+)?', re.I)

# Filler words/phrases Whisper commonly adds
# NOTE: "reachy" removed — it is now a meaningful robot identifier
_FILLER_RE = re.compile(
    r'\b(please|can you|could you|would you|i want you to|i need you to'
    r'|go ahead and|just|okay|hey|robot)\b', re.I)

# --- Spot service commands: maps regex → canonical service name ---
# Checked after robot extraction; only fires when robot == "spot".
_SPOT_SERVICE_RULES = [
    (re.compile(r'^(?:stand(?:\s+up)?|get\s+up|rise)$', re.I),         'stand'),
    (re.compile(r'^(?:sit(?:\s+down)?|lie\s+down|lay\s+down)$', re.I), 'sit'),
    (re.compile(r'^(?:stow(?:\s+(?:the\s+)?arm)?|put\s+(?:the\s+)?arm\s+(?:away|back))$', re.I),   'arm_stow'),
    (re.compile(r'^(?:unstow(?:\s+(?:the\s+)?arm)?|deploy\s+(?:the\s+)?arm|bring\s+(?:the\s+)?arm\s+out)$', re.I), 'arm_unstow'),
    (re.compile(r'^open\s+(?:the\s+)?(?:gripper|hand|claw)$', re.I),   'open_gripper'),
    (re.compile(r'^close\s+(?:the\s+)?(?:gripper|hand|claw)$', re.I),  'close_gripper'),
    (re.compile(r'^(?:claim|claim\s+(?:the\s+)?robot)$', re.I),        'claim'),
    (re.compile(r'^(?:power\s+on|power\s+up|boot(?:\s+up)?)$', re.I),  'power_on'),
    (re.compile(r'^(?:power\s+off|shut\s*down)$', re.I),               'power_off'),
    (re.compile(r'^(?:kill|roll\s*over|emergency(?:\s+stop)?)$', re.I), 'kill'),
]

# --- Reachy service commands: arm on/off/up/down ---
_REACHY_SERVICE_RULES = [
    (re.compile(r'^(?:arm\s+on|turn\s+on(?:\s+(?:the\s+)?arm)?|wake\s+up)$', re.I), 'arm_on'),
    (re.compile(r'^(?:arm\s+off|turn\s+off(?:\s+(?:the\s+)?arm)?|shut\s*down)$', re.I), 'arm_off'),
    (re.compile(r'^(?:arm\s+up|arms?\s+up|ready(?:\s+(?:position|pose))?)$', re.I), 'arm_up'),
    (re.compile(r'^(?:arm\s+down|arms?\s+down|rest(?:\s+(?:position|pose))?)$', re.I), 'arm_down'),
]

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

There are two robots: "reachy" and "spot". The user will address one by name.
If no robot is mentioned, default to "reachy".
Note: speech recognition may mishear "reachy" as "richie", "ritchie", "reachie",
"reechy", "richi", or "richy". All of these refer to the robot "reachy".

Return a JSON object with these fields:
- "robot": "reachy" or "spot"
- "category": one of "navigation", "manipulation", "locomotion", or "service"
- "object": the physical thing being acted on (no articles). Empty string for locomotion/service.
- "instruction": the action to perform (verb + modifiers, no filler)
- "service": canonical service name (only for category="service", empty string otherwise)
- "dx": forward displacement in metres (only for locomotion, 0.0 otherwise)
- "dy": leftward displacement in metres (only for locomotion, 0.0 otherwise)
- "dyaw": counter-clockwise rotation in degrees (only for locomotion, 0.0 otherwise)

Category rules:
- "navigation": the robot should move its base to reach a named object or location
- "manipulation": the robot should use its arm to interact with an object
- "locomotion": the robot should move its base by a relative amount without a target object
- "service": a direct body/state command for Spot (stand, sit, stow arm, etc.)

Valid service names for Spot:
  "stand", "sit", "arm_stow", "arm_unstow", "open_gripper", "close_gripper",
  "claim", "power_on", "power_off", "kill"

Valid service names for Reachy:
  "arm_on", "arm_off", "arm_up", "arm_down"

For locomotion, convert distances to metres and directions to the robot body frame:
- forward = +dx, backward = -dx
- left = +dy, right = -dy
- turn left = +dyaw, turn right = -dyaw

Examples:
User: "reachy go to the bottle"
{"robot": "reachy", "category": "navigation", "object": "bottle", "instruction": "go to", "service": "", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "spot, pick up the red cup"
{"robot": "spot", "category": "manipulation", "object": "red cup", "instruction": "pick up", "service": "", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "tell reachy to grab the ball from the table"
{"robot": "reachy", "category": "manipulation", "object": "ball", "instruction": "grab from the table", "service": "", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "spot navigate to the chair"
{"robot": "spot", "category": "navigation", "object": "chair", "instruction": "navigate to", "service": "", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "reachy move forward 50 centimeters"
{"robot": "reachy", "category": "locomotion", "object": "", "instruction": "move forward 50cm", "service": "", "dx": 0.5, "dy": 0.0, "dyaw": 0.0}

User: "spot stand"
{"robot": "spot", "category": "service", "object": "", "instruction": "stand", "service": "stand", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "spot sit down"
{"robot": "spot", "category": "service", "object": "", "instruction": "sit", "service": "sit", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "spot stow the arm"
{"robot": "spot", "category": "service", "object": "", "instruction": "stow arm", "service": "arm_stow", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "spot unstow arm"
{"robot": "spot", "category": "service", "object": "", "instruction": "unstow arm", "service": "arm_unstow", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "spot open gripper"
{"robot": "spot", "category": "service", "object": "", "instruction": "open gripper", "service": "open_gripper", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "spot close gripper"
{"robot": "spot", "category": "service", "object": "", "instruction": "close gripper", "service": "close_gripper", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "spot power on"
{"robot": "spot", "category": "service", "object": "", "instruction": "power on", "service": "power_on", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "spot kill"
{"robot": "spot", "category": "service", "object": "", "instruction": "kill", "service": "kill", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "reachy turn on"
{"robot": "reachy", "category": "service", "object": "", "instruction": "arm on", "service": "arm_on", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "reachy arm up"
{"robot": "reachy", "category": "service", "object": "", "instruction": "arm up", "service": "arm_up", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "arm down"
{"robot": "reachy", "category": "service", "object": "", "instruction": "arm down", "service": "arm_down", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "reachy turn off the arm"
{"robot": "reachy", "category": "service", "object": "", "instruction": "arm off", "service": "arm_off", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "go to the table"
{"robot": "reachy", "category": "navigation", "object": "table", "instruction": "go to", "service": "", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}

User: "put the bottle on the shelf"
{"robot": "reachy", "category": "manipulation", "object": "bottle", "instruction": "put on the shelf", "service": "", "dx": 0.0, "dy": 0.0, "dyaw": 0.0}"""


@dataclass
class ParsedCommand:
    category: str  # "navigation", "manipulation", "locomotion", or "service"
    object: str
    instruction: str
    robot: str = _DEFAULT_ROBOT  # "reachy" or "spot"
    service: str = ''  # canonical service name (only for category="service")
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
    @staticmethod
    def _normalize_robot(name: str) -> str:
        """Map Whisper misheard variants to canonical robot name."""
        if _REACHY_ALIASES.fullmatch(name):
            return 'reachy'
        if _SPOT_ALIASES.fullmatch(name):
            return 'spot'
        return _DEFAULT_ROBOT

    @staticmethod
    def _extract_robot(text: str) -> tuple[str, str]:
        """Extract robot name from the beginning of the command.

        Returns (canonical_robot_name, remaining_text).
        """
        m = _ROBOT_PREFIX_RE.match(text)
        if m:
            robot = CommandParser._normalize_robot(m.group(1))
            return robot, text[m.end():].strip()
        return _DEFAULT_ROBOT, text

    def _try_rules(self, text: str) -> Optional[ParsedCommand]:
        t = text.strip()
        # Strip trailing punctuation (Whisper often adds periods)
        t = t.rstrip('.,!?;:')

        # Extract robot name before stripping filler
        robot, t = self._extract_robot(t)

        # Strip filler words/phrases
        t = _FILLER_RE.sub('', t).strip()
        t = re.sub(r'\s+', ' ', t)

        if not t:
            return None

        # --- Spot service commands (checked first — "stand", "sit", etc.) ---
        if robot == 'spot':
            for pattern, svc_name in _SPOT_SERVICE_RULES:
                if pattern.match(t):
                    return ParsedCommand('service', '', svc_name,
                                         robot=robot, service=svc_name)

        # --- Reachy service commands ("arm on", "arm up", etc.) ---
        if robot == 'reachy':
            for pattern, svc_name in _REACHY_SERVICE_RULES:
                if pattern.match(t):
                    return ParsedCommand('service', '', svc_name,
                                         robot=robot, service=svc_name)

        # --- Stop / halt / freeze ---
        if _STOP_RE.match(t):
            return ParsedCommand('locomotion', '', 'stop', robot=robot)

        # --- Turn / rotate / spin ---
        m = _TURN_RE.search(t)
        if m:
            direction = m.group(1).lower()
            angle = float(m.group(2)) if m.group(2) else 90.0
            dyaw = angle if direction == 'left' else -angle
            return ParsedCommand('locomotion', '',
                                 f'turn {direction} {angle} degrees',
                                 robot=robot, dyaw=dyaw)

        # --- Move / turn around (180 deg) ---
        if _AROUND_RE.search(t):
            return ParsedCommand('locomotion', '',
                                 'turn around 180 degrees',
                                 robot=robot, dyaw=180.0)

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
                                 f'move {direction}',
                                 robot=robot, dx=dx, dy=dy)

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
                return ParsedCommand('manipulation', obj, f'{action} {obj}',
                                     robot=robot)

        # --- Navigation ---
        m = _NAV_RE.search(t)
        if m:
            target = m.group(1).strip()
            first_word = target.split()[0].lower()
            if first_word not in _DIRECTIONS:
                return ParsedCommand('navigation', target, 'go to',
                                     robot=robot)

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
            robot = self._normalize_robot(
                data.get('robot', _DEFAULT_ROBOT).strip()
            )
            category = data.get('category', '').strip().lower()
            obj = data.get('object', '').strip()
            instr = data.get('instruction', '').strip()

            if category not in ('navigation', 'manipulation', 'locomotion',
                                'service'):
                log.warning('Unknown category "%s": %s', category, data)
                return None

            service = data.get('service', '').strip()

            if category == 'service' and not service:
                log.warning('Empty service name for service command: %s', data)
                return None

            if category not in ('locomotion', 'service') and (not obj or not instr):
                log.warning('Empty object/instruction for %s: %s', category, data)
                return None

            return ParsedCommand(
                category=category,
                object=obj,
                instruction=instr,
                robot=robot,
                service=service,
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
