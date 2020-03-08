# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
dataset_ = np.matrix(dataset)

# Implementing UCB
import math
N = len(dataset_)
d = len(dataset_.transpose())
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    UCB_max = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            avg_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            UCB = avg_reward + delta_i
        else:
            UCB = 1e400
        if UCB > UCB_max:
            UCB_max = UCB
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
    
# Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of Ad Selections')
plt.xlabel('Ad')
plt.ylabel('# of Times Ad was Selected')
plt.show()