#!/usr/bin/env python3
import socket
import numpy as np
import rospy
import sys
import signal
from ground_control_ros.msg import ground_control



if __name__ == "__main__":
    # ros node iniit
    rospy.init_node('ground_control_pub', anonymous=True)

    pub = rospy.Publisher('ground_control_topic', ground_control, queue_size=10)
    rate = rospy.Rate(10)


    # Create a socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to an address and port
    server_socket.bind(('172.17.0.1', 9090))

    # Listen for incoming connections
    server_socket.listen(1)

    print("Listening for connections...")
    # Accept a connection
    client_socket, client_address = server_socket.accept()
    print("Connected to:", client_address)
    
    def signal_handler(sig, frame):
        rospy.loginfo("Ctrl + C signal received. Shutting down...")
        # Close the connection
        client_socket.close()
        sys.exit(0)

        # Register SIGINT signal handler
    signal.signal(signal.SIGINT, signal_handler)
    while not rospy.is_shutdown():
        # Receive data from the client
        data = client_socket.recv(1024)
        if not data:
            break
        
        msg = ground_control()
        msg.obs = np.frombuffer(data).tolist()
        pub.publish(msg)
        print("Received:", np.frombuffer(data))
    
    rate.sleep()

    # Close the connection
    client_socket.close()

