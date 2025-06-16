import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.transforms as transforms
from checkpoint_simulation import simulate_failures
plt.figure(figsize=(12,8))
plt.rcParams.update({'font.size': 24})
plt.locator_params(axis='x', nbins=5)

barWidth = 0.2
br1 = np.arange(3) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 

plt.bar([-0], [22.60155671926645], color ='r', width = barWidth, 
        edgecolor ='grey', label ='IT') 
plt.bar([0.3], [29.244000053405763], color ='g', width = barWidth, 
        edgecolor ='grey', label ='IT') 
plt.bar([0.6], [121.03475341796874], color ='b', width = barWidth, 
        edgecolor ='grey', label ='IT') 
plt.xticks([r * 0.3 for r in range(3)], 
        ['Gradient Average', 'Copy','Random'])
plt.yscale("log")
plt.title("Perplexity after 10000 iterations")
# plt.ylim((0,1e2))
# plt.bar(br2, ECE, color ='g', width = barWidth, 
#         edgecolor ='grey', label ='ECE') 
# plt.bar(br3, CSE, color ='b', width = barWidth, 
#         edgecolor ='grey', label ='CSE') 

# plt.xlabel('Branch', fontweight ='bold', fontsize = 15) 
# plt.ylabel('Students passed', fontweight ='bold', fontsize = 15) 


# plt.legend()
# plt.show() 
# plt.bar(("Gradient Average", "Copy", "Random"), [,
# ,
# ])
plt.savefig("perplexity.pdf")

plt.show()

