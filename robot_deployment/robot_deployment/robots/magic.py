#!/usr/bin/env python3
import socket
import subprocess
import signal
import sys
import time



TARGET_RECV_FLAG = False
HOST_SEND_FLAG   = False

def _process_name_alive(process_name : str):
     try:
         process_status = subprocess.check_output(["pgrep", process_name])
     except subprocess.CalledProcessError:
         print("process : {} does not exist".format(process_name))
         return False
     return True

def _find_process_name(process_name : str):

    #Sanity check
    if not _process_name_alive(process_name):
       return 
    
    #Kill the PIDs associated with that UDP based process
    completed_process = subprocess.run(['pgrep', '-x', process_name], stdout=subprocess.PIPE, check=True)
    print(completed_process)
    PIDs = completed_process.stdout.decode().strip().split()[::-1]
    print("PIDs : ", PIDs)
    for pid in PIDs:
        try:
            print(subprocess.run(['kill', str(pid)], check = True))
        except subprocess.CalledProcessError as e:
            print("Error : ", e)


def _process_port_alive(port : int):
    try:
        process_statuses = subprocess.check_output(['netstat', '-up'], universal_newlines=True)
        process_statuses : list = process_statuses.split()
        print(process_statuses)
        for i in process_statuses:
              if '/' in i:
                if i.split()[-1] == 'python3':
                         pass
            # if ':' in i:
            #     port_name = i.split(':')[-1]
            #     if port_name.isnumeric():
            #         if int(port_name) == port:
                        
    except subprocess.CalledProcessError:
        return False
    
    return False


def _find_process_port(port : int):
    if not _process_port_alive(port):
        return False



def _udp_send_recv_host( local_port : int,
             local_ip   : str,
             target_port: int,
             target_ip  : str,
             message    : bytes
                    ) -> bool:
    global HOST_SEND_FLAG
    if not HOST_SEND_FLAG:
        return False
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        recv_sock.bind((local_ip, local_port))
        recv_sock.settimeout(1)
    except socket.error as e: 
        print("Unable to bind! \n")
        return False

    def signal_handler(sig, frame):
        print('Closing socket...')
        recv_sock.close()
        send_sock.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    try:
        while HOST_SEND_FLAG:
            try:
                send_sock.sendto(message, (target_ip, target_port))
                data, addr = recv_sock.recvfrom(64) #receive bytes
                print("data", data)
                if data.decode('utf-8') == "stop":
                    HOST_SEND_FLAG = False
                    print("received stop" )
                    send_sock.close()
                    recv_sock.close()
                    return True
                print("sending {}".format(message))
            except socket.timeout:
                print("timeout")
    finally:
        send_sock.close()
        recv_sock.close()
        return False

    

def _udp_send_recv_target(local_port : int,
                          local_ip   : str,
                          target_port: int,
                          target_ip  : str,
                          host_port  : int, 
                          host_ip    : str, 
                          message    : bytes) -> bool:
    global TARGET_RECV_FLAG
    if not TARGET_RECV_FLAG:
        return False

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        recv_sock.bind((local_ip, local_port))
        recv_sock.settimeout(1)
    except socket.error as e: 
        print("Unable to bind! \n")
        return False

    def signal_handler(sig, frame):
        print('Closing socket...')
        send_sock.close()
        recv_sock.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    process_names =["A1_sport_1", "A1_sport_2"]
    low_flag = False
    high_flag = False
    try :
        while TARGET_RECV_FLAG:
            try:
                data, addr = recv_sock.recvfrom(64)
                if not high_flag and data.decode('utf-8') == "high":
                    #set up flags
                    low_flag = False
                    high_flag = True

                    #Kill the processes
                    print("received high")     
                    kill_processes(process_names, 1)

                    #check if mcb port is available
                    check_port(target_port)

                    #send confirmation to host
                    send_sock.sendto(message, (host_ip, host_port))

                elif not low_flag and data.decode('utf-8') == "low":
                    #set up flags
                    low_flag = True
                    high_flag = False
                    #Kill the processes

                    print("received low")
                    kill_processes(process_names, 1)
                    
                    #send confirmation to host
                    send_sock.sendto(message, (host_ip, host_port))
                print("receiving..")
            except socket.timeout:
                print("timeout..")
    finally:
        recv_sock.close()
        send_sock.close()
        return False

def magic_target(local_port : int,
                 local_ip   : str,
                 
                 target_port: int,
                 target_ip  : str,

                 host_port  : int,
                 host_ip    : str,
                 message    : bytes
                ):
    global TARGET_RECV_FLAG
    if not TARGET_RECV_FLAG:
        TARGET_RECV_FLAG = True
        _udp_send_recv_target(local_port, local_ip, target_port, target_ip, host_port, host_ip, message)
        print("done")

def magic_host( local_port : int,
                local_ip   : str,
                 
                target_port: int,
                target_ip  : str,
                
                message    : bytes
              ):
    # send confirmation
    global HOST_SEND_FLAG
    if not HOST_SEND_FLAG:
        HOST_SEND_FLAG = True
        _udp_send_recv_host(local_port, local_ip, target_port, target_ip, message)

def kill_processes(processes_names : list, delay):
    #Kill em' all
    for process_name in processes_names:
          _find_process_name(process_name=process_name)
          time.sleep(delay) #A delay from this process causes the other process to start


def check_port(local_port : int,
           ) -> bool:
    while True:
        try:
            result = subprocess.run(['lsof', '-i', ':' + str(local_port) ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        except subprocess.CalledProcessError:
            return False
        port_status = result.stdout.split()[-1]
        if (port_status.split(':')[0] != '*'): # *:xxxx; 
            time.sleep(5) #waiting for sport mode initialization to complete on another process
            return True
        time.sleep(1)
        print("listening..")

if __name__ == "__main__":

    #process names
    process_names =["A1_sport_1", "A1_sport_2"]

    #Kill the processes
    kill_processes(process_names)

    # #For sending message to start a fresh A1_sport_1 process
    # _udp_send_recv_host(8001, "192.168.123.24", 8081, "192.168.123.161")

    # #For receiving message to start a fresh A1_sport_1 process
    # _udp_send_recv_target(8081, "192.168.123.161")

    # #check udp port status
    # check_port(8010)

    # # host magic
    # magic_host(local_port=8001, 
    #            local_ip="192.168.123.24", 
    #            target_port=8081, 
    #            target_ip="192.168.123.161", 
    #            )

    # target magic

    magic_target(local_port=8081, 
                 local_ip="192.168.123.161", 
                 target_port=8010, 
                 target_ip="192.168.123.10", 
                 host_port=8001, 
                 host_ip="192.168.123.24", 
                 message= b"stop"
                 ) 


