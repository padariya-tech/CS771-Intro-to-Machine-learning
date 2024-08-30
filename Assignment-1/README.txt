/*In all data loading statements just specify your path to load the data*
seen_classes = np.load('X_seen.npy', encoding='bytes', allow_pickle=True) //here first parameter is path
unseen_class_attributes = np.load('class_attributes_unseen.npy', encoding='bytes', allow_pickle=True)//here first parameter is path
seen_class_attributes = np.load('class_attributes_seen.npy', encoding='bytes', allow_pickle=True)//here first parameter is path
 test_data_X= np.load('Xtest.npy', encoding='bytes', allow_pickle=True)//here first parameter is path
 test_data_Y = np.load('Ytest.npy', encoding='bytes', allow_pickle=True)//here first parameter is path


/*change these values according to the data_set*/
seen_classes_count=40
unseen_classes_count=10
dimensions_in_class_vector=85

/*intialize this according to the seen_class data_set
means_of_seen_classes= np.empty((40, 4096))

