import csv
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


np.set_printoptions(threshold=np.nan)

X_data = np.zeros((300000, 144))

with open('train_set.csv', "r") as f:
    reader = csv.reader(f)
    j=-2
    for row in reader:
        j=j+1
        if(j==-1):
            continue
        vals = row[1]
        vals = [int(i) for i in vals.split()]
        
        for k in range(len(vals)):
            X_data[j, math.ceil(int(vals[k]/7))]= vals[k]%7+1



Y_data = np.array([])
with open('test_set.csv', "r") as f:
    reader = csv.reader(f)
    j=-2
    for row in reader:
        j=j+1
        if(j==-1):
            continue
        vals2 = row[1]
        Y_data = np.append(Y_data, int(vals2))

all_data = np.column_stack((X_data, Y_data))
np.random.shuffle(all_data)


#in case needed to save the pre-processed data in another csv file.
#with open('data.csv', 'w') as ff:
    #writer = csv.writer(ff)
    #writer.writerows(all_data)



#divide the dataset into 
Y_data = all_data[:,-1]
X_train = all_data[0:250000,0:-1]
Y_train = Y_data[0:250000]
X_test = all_data[250001:,0:-1]
Y_test = Y_data[250001:]




#the code for the cross validation; however, it is commented to save the time needed to run it

'''
neighbors = list(range(1,20))

# subsetting just the odd ones

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, metric="hamming")
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(str(k)+" "+str(scores.mean()))
    
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

'''


neigh = KNeighborsClassifier(n_neighbors=1, metric="hamming" )
neigh.fit(X_train, Y_train)
pred = neigh.predict(X_test)
accuracy=neigh.score(X_test, Y_test)
print('\nThe accuracy of our classifier is %d%%' % accuracy*100)


