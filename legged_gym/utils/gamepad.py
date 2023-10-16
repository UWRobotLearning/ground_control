import logging
from enum import Enum
import pygame
from configs.definitions import CommandsConfig
import time

class GamepadMappings(Enum):
    LEFT_JOYSTICK_HORIZONTAL = 0
    LEFT_JOYSTICK_VERTICAL = 1
    RIGHT_JOYSTICK_HORIZONTAL = 2
    RIGHT_JOYSTICK_VERTICAL = 3
    LEFT_BUMPER = 4
    RIGHT_BUMPER = 5

class Gamepad:
    def __init__(self, command_ranges: CommandsConfig.CommandRangesConfig):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.command_ranges = command_ranges
        print(f"Loaded gamepad with {self.gamepad.get_numaxes()} axes")

        self._lin_vel_x = 0.
        self._lin_vel_y = 0.
        self._ang_vel = 0.
        self._right_bump = False

    def get_command(self):
        get_scale = lambda rnge, norm: abs(rnge[1] if norm >= 0 else rnge[0])
        for _ in pygame.event.get():
            left_js_ver_norm = -self.gamepad.get_axis(GamepadMappings.LEFT_JOYSTICK_VERTICAL.value)
            self._lin_vel_x = left_js_ver_norm * get_scale(self.command_ranges.lin_vel_x, left_js_ver_norm)

            right_js_hor_norm = -self.gamepad.get_axis(GamepadMappings.RIGHT_JOYSTICK_HORIZONTAL.value)
            self._lin_vel_y = right_js_hor_norm * get_scale(self.command_ranges.lin_vel_y, right_js_hor_norm)

            left_js_hor_norm = -self.gamepad.get_axis(GamepadMappings.LEFT_JOYSTICK_HORIZONTAL.value)
            self._ang_vel = left_js_hor_norm * get_scale(self.command_ranges.ang_vel_yaw, left_js_hor_norm)

            self._right_bump = bool(self.gamepad.get_button(GamepadMappings.RIGHT_BUMPER.value))

        return self._lin_vel_x, self._lin_vel_y, self._ang_vel, self._right_bump

if __name__ == '__main__':
    log = logging.Logger("logger")
    gamepad = Gamepad(command_ranges=CommandsConfig.CommandRangesConfig)
    log.info("loaded")
    print("hello")
    for i in range(10000):
        print(f"Command: {gamepad.get_command()}")
        time.sleep(0.1)