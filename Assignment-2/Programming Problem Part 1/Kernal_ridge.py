import numpy as np
import matplotlib.pyplot as plt

#defining RBF kernal
def RBF_kernel(xn, xm):
    # gamma value is 0.1
    return np.exp(-0.1*np.square(xn.reshape((-1,1)) - xm.reshape((1,-1))))


#taking data from the files
#just specify the correct path where file present
training_data=np.genfromtxt('ridgetrain.txt',delimiter='  ')
test_data=np.genfromtxt('ridgetest.txt',delimiter='  ')

#first column in input and second is output
training_input=training_data[:,0]
training_output=training_data[:,1]
test_input=test_data[:,0]
test_output=test_data[:,1]

K = RBF_kernel(training_input, training_input)
Identity_Matrix = np.eye(training_input.shape[0])
lamda_array = [0.1, 1, 10, 100]

for lamda in lamda_array:
    temp=np.linalg.inv(K + lamda*Identity_Matrix) #(K(xT.X)+lambda*I)^-1
    w_partial=np.dot(temp,training_output.reshape((-1,1))) #(K(xT.X)+lambda*I)^-1*y
    #alpha = np.dot(np.linalg.inv(K + lam*In), y_train.reshape((-1,1)))
    #prediction y_pred=w.T*x=(K(xT.X)+lambda*I)^-1*y.T*K(x_training,x_test)
    #Kernel_temp = RBF_kernel(training_input, test_input)
    y_pred = (np.dot(w_partial.T, RBF_kernel(training_input, test_input))).reshape((-1,1))

    RMSE = np.sqrt(np.mean(np.square(test_output.reshape((-1,1)) - y_pred)))
    print('RMSE for lambda = ' + str(lamda) + ' is ='+ str(RMSE))

    plt.figure(lamda)
    plt.title('lambda = ' + str(lamda) + ',RMSE ='+ str(RMSE))
    #prediction in red colour
    plt.plot(test_input, y_pred, 'r*')
    #Actual output in blue colour
    plt.plot(test_input, test_output, 'b*')

plt.show()



