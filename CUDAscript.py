'''
   Tomás Oliveira e Silva, November 2017
   Diogo Daniel Soares Ferreira, 76504
   Luís Davide Jesus Leira, 76514
  
   ACA 2017/2018
'''

import math
import csv
import subprocess
import os

def writeToCSV(text):
    with open('results.csv','wb') as file:
        for line in text:
            file.write(line)

# Initial headers
data = "Grid X, Grid Y, Block X, Block Y, GPU Time, CPU Time, Initialization Time, Transfer from host to GPU, Transfer from GPU to host, Grid Dimensions, Block Dimensions\n"

max_grid = 65535
max_block = 1024
max_power = 2097152

for i1 in range (0,int(math.ceil(math.log(max_grid,2)))):
    grid_x = 1 << i1
    for i2 in range (0,int(math.ceil(math.log(max_grid,2)))):
        grid_y = 1 << i2

        for i3 in range (0,int(math.ceil(math.log(max_block,2))+1)):
            block_x = 1 << i3
            for i4 in range (0,int(math.ceil(math.log(max_block,2))+1)):
                block_y = 1 << i4
                
                # Check if grid does not exceed the maximum size of GPU
                # Compares with 1.0 because of float errors
                if(block_x*block_y-max_block>=1.0):
                    continue
                
                # If the grid and block match the matrix size,
                # Calculate the GPU time
                if(grid_x*grid_y*block_x*block_y==max_power):
                    tempData = str(int(grid_x))+", "+str(int(grid_y))+", "+str(int(block_x))+", "+str(int(block_y))                    
                    execSc = ['./cryptCuda', str(int(grid_x)), str(int(grid_y)), str(int(block_x)), str(int(block_y))]
                    print(execSc)
                    value = 0
                    done = False
                    # If the GPU is busy, try it again
                    while (not done):
                        proc = subprocess.Popen(execSc, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        out, error = proc.communicate()
                        if proc.returncode==0:
                            done = True
                            for idx, line in enumerate(out.split(os.linesep)):
                                print(line)
                                # Get the times printed by the program
                                if idx==0:
                                    init_time = float(line)
                                if idx==1:
                                    copToGpu = float(line)
                                if idx==2:
                                    gpu_value = float(line)
                                elif idx==3:
                                    copToCpu = float(line)
                                elif idx==4:
                                    cpu_value = float(line)
                            # Store the times on the right CSV format to be saved
                            tempData += ", "+str(gpu_value)+", "+str(cpu_value)+", "+str(init_time)+", "+str(copToGpu)+", "+str(copToCpu)
                            print(tempData)
                            data += tempData+"\n"
                        else:
                            print error
                            print "execError"
                    

writeToCSV(data)
