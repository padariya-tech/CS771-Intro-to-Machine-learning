import numpy as np
import matplotlib.pyplot as plt


def distance(x, mean):
    d = np.zeros((x.shape[0], mean.shape[0]))
    for i in range(mean.shape[0]):
        diff = x - mean[i,:].reshape((1,-1))
        d[:,i] = np.sum(np.square(diff), axis=1)
    return d

def prediction(x, mean):
    d = distance(x, mean)
    c = np.argmin(d, axis=1)
    return c.reshape(-1,1)

#taking data from the files
#just specify the correct path where file present
X = np.genfromtxt('kmeans_data.txt', delimiter='  ')
fy=np.zeros((X.shape[0], 3))
#Apply 2D to 3D transformation
for i in range(0,X.shape[0]):
    fy[i][0]=X[i][0]*X[i][0]
    fy[i][1]=np.sqrt(2*np.abs(X[i][0]*X[i][1]))
    fy[i][2]=X[i][1]*X[i][1]

fx=fy
x_cord= X[:, 0]
y_cord =X[:, 1]
#plot original data 
plt.scatter(x_cord, y_cord, c='black')
plt.title('Original Data')
plt.show()
#intially first two rows are initial mean.
mean = fx[:2,:]
c = prediction(fx, mean)

for iter in range(10):
    c=c.ravel()
    mean = np.zeros((2, fx.shape[1]))
    mean[0,:] = np.mean(fx[c==0], axis=0)
    mean[1,:] = np.mean(fx[c==1], axis=0)
    c = prediction(fx, mean)
    c=c.ravel()
    positive = (c==1).reshape(c.shape[0])
    negative = (c==0).reshape(c.shape[0])

plt.scatter(X[positive,0], X[positive,1], c='r')
plt.scatter(X[negative,0], X[negative,1], c='g')
plt.title('After applying K-means')
plt.show()

    
