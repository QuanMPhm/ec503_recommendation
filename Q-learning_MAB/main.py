import numpy as np
from numpy import random
from MABenv import MAB
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt


e_greedy_phase1 = 0.5
e_greedy_phase2 = 0.98

iter_size = 20000
sample_size = 500
explore_phase_size = 1000

if __name__ == '__main__':
    mab = MAB(numOfMAB=10)
    ccr = np.zeros(int(iter_size/sample_size))
    for i in range(iter_size):
        if i == 5000:
            pass
            xx=1


        actionC = random.rand()
        if i<=explore_phase_size:
            e_greedy = e_greedy_phase1
        else:
            e_greedy = e_greedy_phase2

        # Decide whether or not to explore or pick current optimal value
        if actionC <= e_greedy:
            action = np.argmax(mab.Q)
        else:
            action = np.random.randint(10,size = 1)
        mab.step(int(action))
        if i%sample_size==0:
            print('Possibility of winning: ')
            print(mab.Q)
            print('times of pulling of each machine: ')
            print(mab.pull)
            print()
            x=int(i/500)
            ccr[x] = mab.pull[9]/i

            
            
    plt.figure()
    ccr_x = range(int(iter_size/sample_size))
    plt.plot(ccr_x,ccr)
    plt.xlabel('per 500 iterations')
    plt.ylabel('ccr')
    plt.title('testing ccr')
    plt.show()

    plt.figure()
    plt.suptitle('pulling times with corresponding winning probabilities')
    y = mab.pull
    x=np.array(range(11))
    x = x[1:11]
    plt.subplot(2,1,1)
    plt.bar(x,y)
    plt.grid(visible=True)
    plt.xlabel('bandit machine')
    plt.ylabel('pulling times')
    plt.title('pulling times for ten machines')
    plt.subplot(2,1,2)
    y = mab.win
    plt.bar(x,y)
    plt.grid(visible=True)
    plt.xlabel('bandit machine')
    plt.ylabel('possibility of winning')
    #plt.title('possibility of winning for ten machines')
    plt.show()
