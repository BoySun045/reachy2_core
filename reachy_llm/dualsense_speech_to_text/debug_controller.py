#!/usr/bin/env python3
"""Debug script: prints all DualSense evdev events in real-time."""

import evdev
from evdev import InputDevice, categorize, ecodes

# Find all matching devices
for path in evdev.list_devices():
    dev = InputDevice(path)
    name_lower = dev.name.lower()
    if any(p in name_lower for p in ['dualsense', 'wireless controller']):
        print(f'Found: {dev.path} -> {dev.name}')
        caps = dev.capabilities(verbose=True)
        if (ecodes.EV_KEY, 'EV_KEY') in caps or ecodes.EV_KEY in dev.capabilities(verbose=False):
            print(f'  Has EV_KEY (buttons)')
        if (ecodes.EV_ABS, 'EV_ABS') in caps or ecodes.EV_ABS in dev.capabilities(verbose=False):
            print(f'  Has EV_ABS (axes)')
        print()

# Pick the first gamepad device
device = None
for path in evdev.list_devices():
    dev = InputDevice(path)
    name_lower = dev.name.lower()
    caps = dev.capabilities(verbose=False)
    has_buttons = ecodes.EV_KEY in caps
    if has_buttons and any(p in name_lower for p in ['dualsense', 'wireless controller']):
        device = dev
        break

if device is None:
    print('No DualSense found!')
    exit(1)

print(f'Listening on: {device.path} -> {device.name}')
print(f'BTN_TR (R1) code = {ecodes.BTN_TR}')
print('Press buttons on the controller... (Ctrl+C to stop)\n')

for event in device.read_loop():
    if event.type == ecodes.EV_KEY:
        key = categorize(event)
        print(f'KEY: code={key.scancode} keycode={key.keycode} state={key.keystate} '
              f'({"PRESSED" if key.keystate == 1 else "RELEASED" if key.keystate == 0 else "HOLD"})')
