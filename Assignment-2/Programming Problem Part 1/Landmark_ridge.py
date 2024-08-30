import numpy as np
import matplotlib.pyplot as plt

def RBF_kernel(xn, xm):
    # gamma value is 0.1
    return np.exp(-0.1*np.square(xn.reshape((-1,1)) - xm.reshape((1,-1))))


#taking data from the files
#just specify the correct path where file present
training_data=np.genfromtxt('ridgetrain.txt',delimiter='  ')
test_data=np.genfromtxt('ridgetest.txt',delimiter='  ')


training_input=training_data[:,0]
training_output=training_data[:,1]
test_input=test_data[:,0]
test_output=test_data[:,1]
#landmark array
L_Values= [2, 5, 20, 50, 100]
lamda=0.1

for L in L_Values:
    landmarks=np.random.choice(training_input,L)
    #For each input 𝒙𝑛, using a RBF kernel , define an 𝐿-dimensional feature vector so we create a feature matrix which contain all inputs
    L_dimension_training_input = RBF_kernel(training_input, landmarks)
    Identity_Matrix=np.eye(L)



    temp1=np.linalg.inv(np.dot(L_dimension_training_input.T,L_dimension_training_input) + lamda*Identity_Matrix)
    temp2=np.dot(L_dimension_training_input.T, training_output.reshape((-1,1)))
    W=np.dot(temp1,temp2)#weight vector
    #prediction y_pred=w.T*x
    L_dimension_test_input=RBF_kernel(test_input,landmarks)

    y_pred = np.dot(L_dimension_test_input, W)

    RMSE = np.sqrt(np.mean(np.square(test_output.reshape((-1,1)) - y_pred)))
    print('Lambda value is :',lamda)
    print('RMSE for L = ' + str(L) + ' is ='+ str(RMSE))

    plt.figure(L)
    plt.title('L = ' + str(L) + ',RMSE ='+ str(RMSE))
    #prediction in red colour
    plt.plot(test_input, y_pred, 'r*')
    #Actual output in blue colour
    plt.plot(test_input, test_output, 'b*')

plt.show()