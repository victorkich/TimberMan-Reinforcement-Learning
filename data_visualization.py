from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import lfilter
from tqdm import tqdm
import pandas as pd
import numpy as np
import time

def animate(i):
    global count
    global first
    if first:
        time.sleep(0)
        first = False
    if count < len(rewards):
        count = count + 50
    elif count > len(rewards):
        count = len(rewards)

    ax.clear()
    ax.plot(num_epochs[:count], rewards[:count], color='b', linestyle='-',
            linewidth=1, label='Real Rewards')
    ax.plot(num_epochs[:count], mean[:count], color='r', linestyle='-',
            linewidth=1, label='Filtered Rewards')
    ax.set_title('Reward per Epoch', size=20)
    ax.legend(loc=2, prop={'size':20})
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.set_xlim([0, len(rewards)+50])
    ax.set_ylim([0, max(rewards)+5])

df = pd.read_csv('data.csv')[1:-4]
filter = df["evaluation_state"]==0
df.where(filter, inplace = True)
df.dropna(inplace=True)
epochs = pd.unique(df.epoch)
print('Computing Real Rewards: ')
rewards = [len(df[df.epoch == epochs[i]])-1 for i in tqdm(range(epochs.size))]
num_epochs = np.arange(len(rewards))

n = 15  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
mean = lfilter(b,a,rewards)

print('Plotting the graphs: ')
first = True
count = 0
fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate)
plt.show()
