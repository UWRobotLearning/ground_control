#!usr/bin/env python3

import socket


if __name__ == "__main__":
    # Create a socket
    while True:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to an address and port
        server_socket.bind(('172.17.0.1', 9090))

        # Listen for incoming connections
        server_socket.listen(1)


        print("Listening for connections...")
        # Accept a connection
        client_socket, client_address = server_socket.accept()
        print("Connected to:", client_address)

        while True:
            # Receive data from the client
            data = client_socket.recv(1024)
            if not data:
                break
            print("Received:", data.decode())

        # Close the connection
        client_socket.close()
