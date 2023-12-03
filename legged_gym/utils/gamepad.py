import logging
from enum import Enum
import pygame
from configs.definitions import CommandsConfig
import time
import termios, fcntl, sys, os

class GamepadMappings(Enum):
    LEFT_JOYSTICK_HORIZONTAL = 0
    LEFT_JOYSTICK_VERTICAL = 1
    RIGHT_JOYSTICK_HORIZONTAL = 2
    RIGHT_JOYSTICK_VERTICAL = 3
    LEFT_BUMPER = 4
    RIGHT_BUMPER = 5

def get_char_keyboard_nonblock():
    '''
    Taken from: https://stackoverflow.com/a/29187364
    '''
    fd = sys.stdin.fileno()

    oldterm = termios.tcgetattr(fd)
    newattr = termios.tcgetattr(fd)
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, newattr)

    oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

    c = None

    try:
        c = sys.stdin.read(1)
    except IOError: pass

    termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

    return c

class Gamepad:
    def __init__(self, command_ranges: CommandsConfig.CommandRangesConfig):
        pygame.init()
        if pygame.joystick.get_count() > 0:
            self.use_keyboard = False
            self.gamepad = pygame.joystick.Joystick(0)
            self.gamepad.init()
            self.command_ranges = command_ranges
            print(f"Loaded gamepad with {self.gamepad.get_numaxes()} axes")
        else:
            self.use_keyboard = True

        self.reset_values()

    def reset_values(self):
        self._lin_vel_x = 0.
        self._lin_vel_y = 0.
        self._ang_vel = 0.
        self._right_bump = False

    def get_command(self):
        if not self.use_keyboard:
            get_scale = lambda rnge, norm: abs(rnge[1] if norm >= 0 else rnge[0])
            for event in pygame.event.get():
                
                left_js_ver_norm = -self.gamepad.get_axis(GamepadMappings.LEFT_JOYSTICK_VERTICAL.value)
                self._lin_vel_x = left_js_ver_norm * get_scale(self.command_ranges.lin_vel_x, left_js_ver_norm)

                right_js_hor_norm = -self.gamepad.get_axis(GamepadMappings.RIGHT_JOYSTICK_HORIZONTAL.value)
                self._lin_vel_y = right_js_hor_norm * get_scale(self.command_ranges.lin_vel_y, right_js_hor_norm)

                left_js_hor_norm = -self.gamepad.get_axis(GamepadMappings.LEFT_JOYSTICK_HORIZONTAL.value)
                self._ang_vel = left_js_hor_norm * get_scale(self.command_ranges.ang_vel_yaw, left_js_hor_norm)

                self._right_bump = bool(self.gamepad.get_button(GamepadMappings.RIGHT_BUMPER.value))

        else:
            self.reset_values()
            key = get_char_keyboard_nonblock()
            if key == "w":  ## Go up 
                self._lin_vel_x = 1
            if key == "a":  ## Go left
                self._lin_vel_y = 1
            if key == "s":  ## Go down
                self._lin_vel_x = -1
            if key == "d":  ## Go right
                self._lin_vel_y = -1
            if key == "q":  ## Right bump
                self._right_bump = not(self._right_bump)


        return self._lin_vel_x, self._lin_vel_y, self._ang_vel, self._right_bump

if __name__ == '__main__':
    log = logging.Logger("logger")
    gamepad = Gamepad(command_ranges=CommandsConfig.CommandRangesConfig)
    log.info("loaded")
    print("hello")
    for i in range(10000):
        print(f"Command: {gamepad.get_command()}")
        time.sleep(0.1)