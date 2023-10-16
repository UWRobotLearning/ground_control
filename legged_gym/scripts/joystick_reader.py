import time
import pygame

pygame.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Loaded joystick with {joystick.get_numaxes()} axes")

while True:
    for _ in pygame.event.get():
        vx = joystick.get_axis(1)
        vy = joystick.get_axis(2)
        wz = joystick.get_axis(0)

    print(f"Vx = {vx:.4f}, Vy = {vy:.4f}, Wz = {wz:.4f}")
    time.sleep(0.1)
