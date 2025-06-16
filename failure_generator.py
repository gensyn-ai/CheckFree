import random
import json
r1 = random.Random()
r2 = random.Random()
r1.seed(33)
hourly_failure_rate = 16
r2.seed(hourly_failure_rate*5)
p_fail = 0

"""
Generates the failure pattenr of different models and fault settings
"""

# Convert hourly failure to iteration failure
if hourly_failure_rate == 10:
    p_fail = 0.002
elif hourly_failure_rate == 16:
    p_fail = 0.004
elif hourly_failure_rate == 33:
    p_fail = 0.01


stages = [
    [0,1,2],
    [3,4,5],
    [6,7,8],
    [9,10,11],
    [12,13,14],
    [15,16,17],
    [18,19,20]
]
failures = {}

for itr in range(500):
    # print("------")
    if itr % 100 != 1:
        stage_failure = r1.random() < p_fail
        if stage_failure:
            print(stage_failure,itr)
        failed_stage = r2.randint(1,len(stages)-1) if stage_failure else -1
    else:
        failed_stage = -1
    for i,s in enumerate(stages):
        faults = 0
           
        for nd in s:
            if nd not in failures:
                failures[nd] = []
            if failed_stage == i:
                failures[nd].append(itr)
           

for v in failures.values():
    v.append(501)
with open(f"failure_p_configs/{hourly_failure_rate}.json", 'w') as fd:
    json.dump(failures, fd)


            
                
        
       
    
# print(failures)

# print(failures)
