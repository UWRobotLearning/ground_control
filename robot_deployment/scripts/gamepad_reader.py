import time
from robot_deployment.robots.gamepad_reader import Gamepad

gamepad = Gamepad()
while True:
    lin, rot = gamepad.speed_command

    print(f"Vx = {lin[0]:.4f}, Vy = {lin[1]:.4f}, Wz = {rot:.4f}")
    time.sleep(0.1)
