import numpy as np

unseen_class_attributes = np.load('class_attributes_unseen.npy', encoding='bytes', allow_pickle=True)
seen_class_attributes = np.load('class_attributes_seen.npy', encoding='bytes', allow_pickle=True)



seen_classes_count=40
unseen_classes_count=10
dimensions_in_class_vector=85
seen_classes = np.load('X_seen.npy', encoding='bytes', allow_pickle=True)

# create an array to store of mean of all seen classes
means_of_seen_classes= np.empty((40, 4096)) # we have given 40 classes and each has 4096 features

# Loop through each seen class and compute mean of each class
class_means_list = [np.mean(class_data, axis=0) for class_data in seen_classes]

# Convert the list of mean feature vectors to a NumPy array
means_of_seen_classes = np.array(class_means_list)

#Now start calculate W as given in method2
w=np.dot(seen_class_attributes.T,seen_class_attributes)

#create an indentity matrix of dimension 85
I=np.eye(dimensions_in_class_vector)

#create an array for lamda values
l=np.array([0.01,0.1,1,10,20,50,100])
# create a list store accuracy for each value lamda
Accuracy=[]

#iterate for each lamda
for k in range(len(l)):
 M_I=l[k]*I

 temp=w+M_I
 Y= np.linalg.inv(temp)

 X=np.dot(seen_class_attributes.T,means_of_seen_classes)
 #final value of weight vector
 W=np.dot(Y,X)

 unseen_class_means_list=[]

 for i in range(unseen_classes_count):
    mean=np.dot(W.T,unseen_class_attributes[i])
    unseen_class_means_list.append(mean)

 unseen_class_means=np.array(unseen_class_means_list)
 #print(unseen_class_means)

#load the test data
 test_data_X= np.load('Xtest.npy', encoding='bytes', allow_pickle=True)
 test_data_Y = np.load('Ytest.npy', encoding='bytes', allow_pickle=True)

#it is given that test data is only from the unseen class so we need to compute distance from the mean from unseen class.
 correct_prediction=0
 for i in range(len(test_data_X)):
  first=0
  indices=0
  for j in range(unseen_classes_count):
    distance = np.linalg.norm(test_data_X[i]-unseen_class_means[j])
    if(first==0):
       first=1
       d=distance
    else:
       if(distance<d):
          d=distance
          indices=j

  predict_value=indices+1
  if predict_value == test_data_Y[i]:
    correct_prediction+=1
 accuracy=correct_prediction/len(test_data_X)
 print('Accuracy in percentage is for lamda ',l[k],'is',accuracy*100)
 Accuracy.append(accuracy*100)


best_l=np.argmax(Accuracy)
print("best Accuracy at lamda =",l[best_l])




