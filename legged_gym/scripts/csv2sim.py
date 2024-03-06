
import pandas as pd
import os
import torch
import math
import sys
sys.path.append("/home/rohan/Desktop/Legged_bags/")
from Filters import butterworth

csv_path = '/'.join(os.getcwd().split('/')[:-3])+'/Legged_bags/Csvs'



def ReadCsv(filename : str, type : str):
    return main(filename, type)



def main(filename:str, type:str):
    file = ''
    csvs = os.listdir(csv_path)

    if filename not in csvs or type not in ("P", "V", "T"):
        print("csv not found! \n")
        return 0
    
    for csv in csvs:
        if csv == filename:
            file = csv
            break

    df = pd.read_csv(csv_path + '/' + file)

    if type == "P":

        Joints      = ["FR0 q", "FR1 q", "FR2 q", "FL0 q", "FL1 q", "FL2 q", 
                      "RR0 q", "RR1 q", "RR2 q", "RL0 q", "RL1 q", "RL2 q"]
    
    elif type == "V":
        Joints      = ["FR0 dq", "FR1 dq", "FR2 dq", "FL0 dq", "FL1 dq", "FL2 dq", 
                      "RR0 dq", "RR1 dq", "RR2 dq", "RL0 dq", "RL1 dq", "RL2 dq"]
       
    else:
        Joints      = ["FR0 tauEst", "FR1 tauEst", "FR2 tauEst", "FL0 tauEst", "FL1 tauEst", "FL2 tauEst", 
                      "RR0 tauEst", "RR1 tauEst", "RR2 tauEst", "RL0 tauEst", "RL1 tauEst", "RL2 tauEst"]
    configs = {"Sampling_frequency":1000,
            "Low_frequency_cutoff":50,
            "High_frequency_cutoff":80,
            "order":2
            }
    
    filter = butterworth.Filter(**configs)

    JointTensor = torch.stack([torch.tensor( filter.butterlow(df[Joint].values)+0.2, dtype = torch.float32) for Joint in Joints], dim = 1)
    
    return JointTensor


if __name__ == "__main__":
    filename = "Circles.csv"
    main(filename, 'P')

