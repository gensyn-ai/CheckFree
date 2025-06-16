import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.transforms as transforms
from checkpoint_simulation import simulate_failures
plt.figure(figsize=(12,8))
plt.rcParams.update({'font.size': 24})
plt.locator_params(axis='x', nbins=5)
maximum_size = 0
def smooth_func(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
def plot_fl(fl,label, validation = False, pad = [], flag = False, max_el = -1, show_failures = False, smooth = 1, val_loss = 2.9):
    global maximum_size
    start = False
    validation_loss = [] + pad
    training_loss = []
    tmp = 0
    prev_checkpoint = 0
    actual_iteration = 0
    actual_run = []
    failures = []
    with open(fl,"r") as fd:
        for ln in fd.readlines():
            
            if "Iteration failure probability" in ln:
                start = True
                continue
            if not start:
                continue
            if "SAVED" in ln:
                continue
            if "failure" in ln:
                if flag:
                    # print(actual_iteration, "returinging to ", prev_checkpoint)
                    
                    actual_iteration = prev_checkpoint
                to_add = int(ln.split(" ")[1])
                node = int(ln.split(" ")[-1].strip())
                # print(node)
                # if node != 0 and node != 5:
                #     continue
                if to_add > max_el and max_el != -1:
                    break
                failures.append(to_add)
                

                continue
            if "time" in ln:
                continue
            if "SAVING" in ln:
                # print(ln)
                prev_checkpoint = actual_iteration
                # print(label,prev_checkpoint)
                continue
            if "NORMAL" in ln and validation:
                
                
                validation_loss.append(float(ln.strip().split(" ")[3].strip()))
                if not flag and validation_loss[-1] < val_loss and max_el == -1:
                    print("BREAK")
                    break
                continue
            # print(ln)
            if "VALIDATION" in ln or "NORMAL" in ln:
                continue

            # print(ln)

            try:
                # print(ln)
                actual_run.append(actual_iteration)
                actual_iteration += 1
                training_loss.append(float(ln.split(" ")[1].strip()))
                
            except ValueError:
                continue
            except IndexError:
                continue
    # print(arr1)
    adjust = False
    if validation and max_el != -1:
        adjust = True
        max_el = max_el // 500
    ret = training_loss
    if validation:
        ret = validation_loss
    
    
    if flag:
        if validation:
            tmp = []
            for i in range(0,len(actual_run)//500):
                # print(i)
                i = actual_run[i*500] / 500
                fl = math.floor(i)
                cl = math.ceil(i)
                alpha = i - fl
                tmp.append((1-alpha)*validation_loss[fl] + alpha * validation_loss[cl])
                if tmp[-1] < val_loss:
                    break
            
        else:
            tmp = []
            for i in range(0,max_el):
                tmp.append(training_loss[actual_run[i]])

        ret = tmp
    print(label,len(ret))
    if (max_el != -1 and not adjust) or adjust:
        
        ret = ret[:max_el]
    ret = smooth_func(ret,smooth)
    maximum_size = max(maximum_size,len(ret))
    plt.plot(ret,label=label,linewidth=3.0)
    
    if validation:
        # 
        if show_failures:
            plt.vlines(list(map(lambda el: el/500,failures)),ymin=0,ymax=10,color=(1,0,0,0.1))
    else:
        if show_failures:
            plt.vlines(failures,ymin=0,ymax=10,color=(1,0,0,0.1))

MAX_EL = 20000
validate = True
show_failures = False
smooth = 1
val_loss = 2.85
tmp = 0
# tmp = simulate_failures("results/send_0/out0.txt","results/results_new/medium_10/out0.txt",val_loss=val_loss,checkpoint_freq=100, label="Checkpointing 100")
# tmp = simulate_failures("results/send_0/out0.txt","results/results_new/medium_10/out0.txt",val_loss=val_loss,checkpoint_freq=50, label="Checkpointing 50")
# tmp = simulate_failures("results/send_0/out0.txt","results/results_new/medium_10/out0.txt",val_loss=val_loss,checkpoint_freq=10, label="Checkpointing 10")
# # tmp = max(tmp,simulate_failures("results/send_0/out0.txt","results/results_new/medium_16/out0.txt",val_loss=val_loss,checkpoint_freq=100, label="Checkpointing 50"))
# tmp = max(tmp,simulate_failures("results/send_0/out0.txt","results/checkfreep_5/out0.txt",val_loss=val_loss,checkpoint_freq=100, label="Checkpointing 50"))
# 24000
# 24500
# 22000
# tmp = max(tmp,simulate_failures("results/small_baseline_10/out0.txt","results/small_ours_10/out0.txt",val_loss=val_loss,checkpoint_freq=50, label="Checkpointing"))
# plot_fl("results/medium_naive_16/out0.txt", "Naive copy 16%",max_el=MAX_EL,validation=validate, show_failures=show_failures, smooth = smooth)
# plot_fl("results/medium_baseline_16/out0.txt", "Checkpointing 16%",flag=True,validation=validate, show_failures=show_failures, smooth = smooth)
# plot_fl("results/medium_gradavg_33/out0.txt", "Ours 33%",validation=validate, show_failures=show_failures, smooth = smooth)
# plot_fl("results/small_gradavg_16/out0.txt", "Ours 16%",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
# plot_fl("results/medium_gradavg_33/out0.txt", "Ours 33%",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
# plot_fl("results/small_fix_10/out0.txt", "CheckFree",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
# plot_fl("results/small_ours_10/out0.txt", "CheckFree+",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
# plot_fl("results/small_baseline_10/out0.txt", "Redundant computation",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
# plot_fl("results/medium_naive_16/out0.txt", "Copy",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
# plot_fl("results/medium_gradavg_16/out0.txt", "Weighted Average",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)

# plot_fl("results/medium_gradavg_10/out0.txt", "Ours 10%",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
# plot_fl("results/checkfree_10/out0.txt", "CheckFree",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)

# plot_fl("results/results_large_ours_2/out0.txt", "CheckFree",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
# plot_fl("results/results_large_baseline/out0.txt", "Redundant Computation",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
plot_fl("results/checkfreep_0/out0.txt", "CheckFree+",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
plot_fl("results/send_0/out0.txt", "No swaps",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
# plot_fl("results/small_baseline_16/out0.txt", "Redundant 16%",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
# plot_fl("results/to_send_small_no_fault/out0.txt", "Baseline")
# plot_fl("results/to_send_grad_avg_16_small/out0.txt", "ours", pad=[10.04,10.04,10.04,10.04,10.04,10.04])
# plot_fl("results/small_baseline_16/out0.txt", "Checkpointing",flag=True,validation=validate, show_failures=show_failures, smooth = smooth, val_loss=1.3)
# plot_fl("results/small_/out0.txt", "Ours",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=1.3)
# plot_fl("results/small_baseline_16/out0.txt", "Redundant",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=1.3)


# plot_fl("results/medium_random_16/out0.txt", "Random",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
# plot_fl("results/results_new/medium_16_copy/out0.txt", "Copy",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
# plot_fl("results/results_new/medium_16/out0.txt", "Gradient Average",validation=validate, show_failures=show_failures, smooth = smooth, val_loss=val_loss)
maximum_size = max(maximum_size,tmp)
if validate:
    nbins = min(maximum_size,5)
    print(maximum_size)
    bottom = list(range(0,maximum_size + maximum_size//nbins,maximum_size//nbins))
    remap = map(lambda el: el*500, bottom)
    plt.xticks(bottom,remap)
title = "Effects of swapping"
plt.legend()
plt.title(title)
plt.ylabel("Validation Loss")
plt.xlabel("Iteration")
# plt.savefig("small_results.pdf")
plt.savefig(f"{title.replace(" ","_").replace("%","")}.pdf")
plt.show()