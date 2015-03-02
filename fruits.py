import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

dataset_filename = 'fruit.csv'

height1 = []
width1 = []

height2 = []
width2 = []

height3 = []
width3 = []

fruits_data = []
fruits_classes = []

with open(dataset_filename, 'r') as fruitfile:

    # Parse it as a CSV file.
    fruits_csv = csv.reader(fruitfile, delimiter=',', quotechar='\'')

    # Skip the header row.
    next(fruits_csv, None)
    
    # Load the data.
    for row in fruits_csv:
        fruits_data.append([float(row[1]),float(row[2])])
        fruits_classes.append(int(row[0]))
        
        if row[0] == '1':
            width1.append(float(row[1]))
            height1.append(float(row[2]))
            
        elif row[0] == '2':
            width2.append(float(row[1]))
            height2.append(float(row[2]))
            
        else:
            width3.append(float(row[1]))
            height3.append(float(row[2]))
    
    means1 = [sum(width1)/len(width1),sum(height1)/len(height1)]
    means2 = [sum(width2)/len(width2),sum(height2)/len(height2)]
    means3 = [sum(width3)/len(width3),sum(height3)/len(height3)]

    vars1 = [width1,height1]
    vars2 = [width2,height2]
    vars3 = [width3,height3]

    cov1 = np.cov(vars1)
    cov2 = np.cov(vars2)
    cov3 = np.cov(vars3)
    
    MV1 = multivariate_normal(mean=means1, cov=cov1)
    MV2 = multivariate_normal(mean=means2, cov=cov2)
    MV3 = multivariate_normal(mean=means3, cov=cov3)
    
    def model(Z):
        result = []
        for coord in Z:
            xy = [coord[0],coord[1]]
            
            values = [MV1.pdf(xy),MV2.pdf(xy),MV3.pdf(xy)]
            #print values
            
            result.append(values.index(max(values))+1)
            
        return np.array(result)

        # X - some data in 2dimensional np.array
X = np.array(fruits_data)
Y = np.array(fruits_classes)

h=.03

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# here "model" is your model's prediction (classification) function
Z = model(np.c_[xx.ravel(), yy.ravel()]) 

# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Set2)
#plt.axis('off')

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Accent)
plt.xlabel('Width (cm)')
plt.ylabel('Height (cm)')


plt.show()