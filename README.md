Moved to: [https://UWRobotLearning.github.io/LeggedRobots/a1_control_base](https://UWRobotLearning.github.io/LeggedRobots/a1_control_base)

# ground_control


# A1 default policy handling : 

# mode handling
sport/normal mode toggle depends on the remote. Included magic.py that makes toggling esier by pressing L1 + start. magic.py will be running on the target device as it is, while the host code is used for the host device. 

# killing auto startup executables

This piece of code kills the sport mode only when a message is sent to the targe udp port. The target port also sends an ack.

# Cleanup of robot_interfaces

Also the cleanup of robot_interface parent classes are done inside the sdk black box, which is why I faced problems when I tried to clean the udp/safe classes before cleaning the robot_interface class.
As a result for every new robot_interface, I have seperately created a new PID so that I can leave  memory handling aspect of the objects to the sdk whenever a PID is freed up.

# Walk_in_the_park recovery : 

Testing pending