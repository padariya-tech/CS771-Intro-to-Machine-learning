1.In Landmark_ridge.py we are taking landmarks from data randomly so every time we may get different output.
2.In Landmark_kmeans.py we are taking landmark from data randomly so every time we may get different output.

To run the code of any program  we have to specify the path name of datafile(I have written a comment in each program file where path must
be mention).
In Kernal_ridge.py
training_data=np.genfromtxt('ridgetrain.txt',delimiter='  ')//here first parameter is path
test_data=np.genfromtxt('ridgetest.txt',delimiter='  ')//here first parameter is path
In Landmark_ridge.py
training_data=np.genfromtxt('ridgetrain.txt',delimiter='  ')//here first parameter is path
test_data=np.genfromtxt('ridgetest.txt',delimiter='  ')//here first parameter is path

In Handcraft_features.py
X = np.genfromtxt('kmeans_data.txt', delimiter='  ')//here first parameter is path

In Landmark_Kmeans.py
training_data=np.genfromtxt('kmeans_data.txt',delimiter='  ')//here first parameter is path

In PCA_tSNE.py
data = np.load('mnist_small.pkl', encoding='bytes', allow_pickle=True)//here first parameter is path

