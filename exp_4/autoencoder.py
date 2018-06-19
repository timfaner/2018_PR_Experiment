import nn_base

import random,time,multiprocessing

def main():
    dataset = [[random.randint(0,1) for r in range(3)] for i in range(20)]
    labels = dataset.copy()
    print(dataset)
    autoencoder = nn_base.BPNeuralNetwork()
    autoencoder.setup(3,2,3)
    autoencoder.train(dataset,labels,learn = 0.05,correct=0.1,limit=50000)
    for data in dataset:
        print(autoencoder.predict(data))

    def newTrain(mtp,node_count):
        print('NN_{} begin, hidden node count = {}'.\
        format(mtp,node_count))
        t = time.time()
        nn = nn_base.BPNeuralNetwork()
        nn.setup(3,node_count,3)
        error = nn.train(dataset,labels,limit=50000,learn=0.05,correct=0.1)
        dt = round(time.time() - t,2)
        print('NN_{} done , spent {}s, with error {}'\
        .format(mtp,dt,round(error,4)))

    for i in range(1,7):
        a = multiprocessing.Process(target=newTrain,args=(i,i))
        a.start()

if __name__ == '__main__':
    main()