
import math
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.transforms as transforms

def simulate_failures(fl_failures, fl_no_fault_run, val_loss = 2.9, checkpoint_freq = 20, label = ""):
    global maximum_size
    failures = []
    actual_run = []
    validation_loss = []
    start = False
    with open(fl_failures,"r") as fd:
        for ln in fd.readlines():
            
            if "Iteration failure probability" in ln:
                start = True
                continue
            if not start:
                continue
            if "SAVED" in ln:
                continue
            if "failure" in ln:
                
                to_add = int(ln.split(" ")[1]) 
                if len(failures) > 0 and to_add == failures[-1]:
                    continue
                failures.append(to_add)
                continue
    holder = failures
    failures = iter(failures)
    to_fail = next(failures)
    actual_iteration = 0
    start = False
    checkpoints = []
    with open(fl_no_fault_run,"r") as fd:
        for ln in fd.readlines():
            
            
            if "Iteration failure probability" in ln:
                start = True
                continue
            if not start:
                continue
            if "SAVED" in ln:
                continue
            if "failure" in ln:

                continue

            if "time" in ln:
                continue
            if "SAVING" in ln:
                # print(ln)
                prev_checkpoint = actual_iteration
                
                # print(label,prev_checkpoint)
                continue
            if "NORMAL" in ln:

                validation_loss.append(float(ln.strip().split(" ")[3].strip()))
                continue


            # print(ln)

            try:
                # print(ln)
                actual_run.append(actual_iteration)
                actual_iteration += 1
                
                
                
            except ValueError:
                continue
            except IndexError:
                continue
    tmp = []
    actual_run = []
    actual_iteration = 0
    for i in range(200_000):
        if i == to_fail:
            try:
                to_fail = next(failures)
            except StopIteration: 
                to_fail = -1
            # print(i)
            if len(actual_run) < checkpoint_freq:
                actual_iteration = 0
            elif i % checkpoint_freq == 0:
                actual_iteration = actual_iteration - checkpoint_freq
            else:
                t = i // checkpoint_freq
                t = t * checkpoint_freq
                
                print("returning from",actual_iteration,"to",actual_run[t],t,i)
                actual_iteration = actual_run[t]
        actual_run.append(actual_iteration)
        actual_iteration += 1


    # print(len(actual_run))
    # print(actual_run[0])
    # print(actual_run[500])
    # print(actual_run[1000])

    # print(actual_run[1500])
    # print(actual_run[2000])
    # print(actual_run[2500])
    for i in range(0,len(actual_run)//500):
        k = i
        i = actual_run[i*500] / 500
        print(i,k)
        fl = math.floor(i)
        cl = math.ceil(i)
        alpha = i - fl
        tmp.append((1-alpha)*validation_loss[fl] + alpha * validation_loss[cl])
        if tmp[-1] < val_loss:
            print("ACTUAL",i*500, actual_run[k*500],k*500,validation_loss[fl])

            # print(tmp)
            break
    # print(len(tmp))
    # print(tmp)
    # print(validation_loss)
    plt.plot(tmp,label=label,linewidth=3.0)
    return len(tmp)
    

    

# simulate_failures("results/medium_gradavg_16/out0.txt", "results/medium_baseline_16/out0.txt")
# plt.show()