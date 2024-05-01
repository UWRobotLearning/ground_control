#ifndef UNITREE_WIRELESS_H_
#define UNITREE_WIRELESS_H_ 

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <math.h>
#include <iostream>
#include <unistd.h>


using namespace UNITREE_LEGGED_SDK;

class WIRE_LESS_CONTROL{

    private:

    static WIRE_LESS_CONTROL* UdpSingleton(uint8_t level);

    public:

    WIRE_LESS_CONTROL(uint8_t level) : safe(LeggedType::A1), udp(level){
        udp.InitCmdData(cmd);
    }

    Safety safe;
    UDP udp;
    
    LowCmd cmd = {0};
    LowState state = {0};
    xRockerBtnDataStruct _keyData = {0};
    int motiontime = 0;
    float dt = 0.002;     // 0.001~0.01

    void UDPSend();
    void UDPRecv();
    void UDPCont();
    void UDPLoop();
    void UdpDeleteInstance();
    static WIRE_LESS_CONTROL* wirelesscontrol;
    static WIRE_LESS_CONTROL* GetUdpInstance(uint8_t level);
    
};

#endif