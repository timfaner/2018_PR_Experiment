import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sklearn.datasets as dataset
import time


#生产测试数据集
#


#sgn函数
def sgn(input):
    return 1 if input > 0 else -1



def main():
    
    #plt.ion()
    
    #generate dataset
    DATA_SET,LABEL = dataset.make_blobs(n_samples=100,centers=2)
    plt.scatter(DATA_SET[:,0],DATA_SET[:,1],c=LABEL)

    #
    W = np.ones(3)
    base = np.array((1))

    nita = 0.5
    count = 0
    n=0
    
    while True:
        n += 1
        print('count is {}'.format(n))
        #time.sleep(0.01)
        old_W = W.copy()
        for position,x in enumerate(DATA_SET):
            X = np.hstack((x,base))
            #print('input x is {}'.format(X))
            chg = 1 if LABEL[position] else -1
            temp = np.dot(W,X) * chg
            if temp > 0:
                pass
            elif temp <= 0:
                #update
                W = W + nita*X*chg
                
                break


        #plt.clf()
        #plt.scatter(DATA_SET[:,0],DATA_SET[:,1],c=LABEL)


        if (old_W==W).all():
            a = W[0];b=W[1];c=W[2]
            
            bian = plt.axis()
            temp = [0,0,0,0]
            temp[0] = (-c - a*bian[0])/b
            temp[1] = (-c - a*bian[1])/b
            temp[2] = (-c - b*bian[2])/a
            temp[3] = (-c - b*bian[3])/a
            plot_W = [(bian[0],temp[0]),(bian[1],temp[1]),(temp[2],bian[2]),(temp[3],bian[3])]
            new = []
            for p in plot_W:
                if in_rect(p,bian):
                    new.append(p)
            
            line = plt.plot((new[0][0],new[1][0]),(new[0][1],new[1][1]))
            break
        
    print('Resualt is {}'.format(W))
    print('Count is {}'.format(count))

    plt.show()
def in_rect(plot,rect):
    if plot[0] <= rect[1] and plot[0] >= rect[0] \
    and plot[1] <= rect[3] and plot[1] >= rect[2]:
        return True
    else:return False
if __name__ == '__main__':
    main()
    while True:
        pass
