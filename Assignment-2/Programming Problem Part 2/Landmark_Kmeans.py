import numpy as np
import matplotlib.pyplot as plt



def Compute_mean(data, c):
    new_mean = np.zeros((2, data.shape[1]))
    new_mean[0,:] = np.mean(data[c==0], axis=0)
    new_mean[1,:] = np.mean(data[c==1], axis=0)
    return new_mean

def Prediction(data, mean):
    d = np.zeros((data.shape[0], mean.shape[0]))
    for i in range(mean.shape[0]):
        diff = data - mean[i,:].reshape((1,-1))
        d[:,i] = np.power(np.sum(np.square(diff), axis=1),0.5)
    c = np.argmin(d, axis=1)
    return c.reshape(-1,1)


#taking data from the files
#just specify the correct path where file present
training_data=np.genfromtxt('kmeans_data.txt',delimiter='  ')
num_rows = training_data.shape[0]

for i in range(10):
    #generate randomly index to select landmark
    L = np.random.randint(training_data.shape[0]-1)
    print(L)
    Tranformed= np.exp(-0.1*np.sum(np.square(training_data - training_data[L,:].reshape((1,-1))), axis=1)).reshape(-1,1)
    #cluster mean (Intialize with first 2)
    mean_points = Tranformed[:2,:]
    cluster_id = Prediction(Tranformed, mean_points)

    mean_points = Compute_mean(Tranformed, cluster_id)
    cluster_id = Prediction(Tranformed, mean_points)
    positive = (cluster_id==0).reshape(cluster_id.shape[0])
    negative = (cluster_id==1).reshape(cluster_id.shape[0])

    plt.figure(i)
    plt.scatter(training_data[positive,0], training_data[positive,1], c='g')
    plt.scatter(training_data[negative,0], training_data[negative,1], c='r')
    plt.scatter(training_data[L, 0], training_data[L, 1], color='blue', marker='s', label='Highlighted Point')

plt.show()

    

