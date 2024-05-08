#!/usr/bin/env python3
import socket
import subprocess
import signal
import sys
import time

def process_alive(process_name : str):
     try:
         process_status = subprocess.check_output(["pgrep", process_name])
     except subprocess.CalledProcessError:
         print("process : {} does not exist".format(process_name))
         return False
     return True

def find_process_name(process_name : str):

    #Sanity check
    if not process_alive(process_name):
       return 
    
    #Kill the PIDs associated with that UDP based process
    completed_process = subprocess.run(['pgrep', '-x', process_name], stdout=subprocess.PIPE, check=True)
    print(completed_process)
    PIDs = completed_process.stdout.decode().strip().split()
    print("PIDs : ", PIDs)
    for pid in PIDs:
        try:
            print(subprocess.run(['kill', str(pid)], check = True))
        except subprocess.CalledProcessError as e:
            print("Error : ", e)



def udp_bind_socket(local_port : int,
                     local_ip   : str,
                     target_port: int,
                     target_ip  : str) -> None:
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((local_ip, local_port))
    udp_process = subprocess.Popen(["/home/unitree/Unitree/keep_program_alive/bin/A1_sport_1"])
    print("Listening on {}:{} .. ".format(local_ip, local_port))


    def signal_handler(sig, frame):
        print('Closing socket...')
        sock.close()
        udp_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
        print("Received data from {}:{}".format(addr[0], addr[1]))
        print("Data:", data.decode())
    
    return None

def start():
    
    udp_process = subprocess.Popen(["/home/unitree/Unitree/keep_program_alive/bin/A1_sport_1"])

    def signal_handler(sig, frame):
         print('Closing socket...')
         udp_process.terminate()
         sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
     

if __name__ == "__main__":
      #Kill the processes
      process_names =["A1_sport_1", "A1_sport_2"]
      for process_name in process_names:
          find_process_name(process_name=process_name)
      #initiate process    
      #udp_bind_socket(8010, "192.168.123.161", 8007, "192.168.123.10")
      start()