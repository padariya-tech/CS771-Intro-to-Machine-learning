import numpy as np


seen_classes_count=40
unseen_classes_count=10
dimensions_in_class_vector=85
# Load the seen classes training data to compute
seen_classes = np.load('X_seen.npy', encoding='bytes', allow_pickle=True)

# create an array to store of mean of all seen classes
means_of_seen_classes= np.empty((40, 4096)) # we have given 40 seen classes and each has 4096 features

# compute mean of each class
class_means_list = [np.mean(class_data, axis=0) for class_data in seen_classes]

# Convert the list of mean feature vectors to a NumPy array
means_of_seen_classes = np.array(class_means_list)

# load the 85 Dimension class atrritube for both seen(40 seen classes) and unseen classes(10 unseen classes).
unseen_class_attributes = np.load('class_attributes_unseen.npy', encoding='bytes', allow_pickle=True)
seen_class_attributes = np.load('class_attributes_seen.npy', encoding='bytes', allow_pickle=True)

#now we need to compute Sc,1,Sc,2..Sc,40 for every c from 41 to 50 and store in Sck_list;

Sck_list=[]

for i in range(unseen_classes_count):
    Sck = np.empty(seen_classes_count) 
    
    for j in range(seen_classes_count):
        innerproduct=0
        #each class attribute vector is of 85 dimension so we have to take sum of all 85 pair wise dot product
        for k in range(dimensions_in_class_vector):
            innerproduct += unseen_class_attributes[i][k] * seen_class_attributes[j][k]
        Sck[j] = innerproduct
    
    sum=0
    for l in range(seen_classes_count):
        sum=sum+Sck[l]

    # Normalize the vector
    Sck=Sck/sum
    Sck_list.append(Sck)

Sck_unseen = np.array(Sck_list)



#now compute the mean of unseen classes
# create a list to store unseen classes means
unseen_class_means_list=[]

for i in range(unseen_classes_count):
  sum=0
  for j in range(seen_classes_count):
      sum = sum + Sck_unseen[i][j] * means_of_seen_classes[j] # ith unseen class mean=sum of product  ith unseen class*each jth seen class mean

  unseen_class_means_list.append(sum)

unseen_class_means=np.array(unseen_class_means_list)

#load the test data
test_data_X= np.load('Xtest.npy', encoding='bytes', allow_pickle=True)
#load the correct label for test data
test_data_Y = np.load('Ytest.npy', encoding='bytes', allow_pickle=True)

#it is given that test data is only from the unseen class so we need to compute distance from the mean from unseen class.
correct_prediction=0
for i in range(len(test_data_X)):
  first=0
  indices=0
  for j in range(unseen_classes_count):
    distance = np.linalg.norm(test_data_X[i]-unseen_class_means[j]) # calculate the euclidian distance between test input and jth unseen class mean
    if(first==0):
       first=1
       d=distance
    else:         # when distance is lesser then update the distance
       if(distance<d):
          d=distance
          indices=j

  prediction=indices+1
  if prediction == test_data_Y[i]:
    correct_prediction+=1         
Accuracy=correct_prediction/len(test_data_X)
print('Accuracy in percentage is ',Accuracy*100)



