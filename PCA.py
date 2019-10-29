# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 19:45:30 2018

@author: abhij
"""
import numpy as np
import matplotlib.pyplot as py
from sklearn import linear_model

"""
1. (Principal Component Analysis, 50 points)
We explore how to use PCA to generate \good" fea-
tures for Handwritten Digit Recognition using the
USPS data set. A reduced and pre-processed ver-
sion of this data set is available on eLearning with

les usps.train, usps.valid and usps.test. The

les are in CSV format, with the 
rst column be
ing the digit label and the next 256 columns rep-
resenting gray scale intensities of the 16  16 digit.
The features are all in the range [0; 1]. Perform the
following experiment and write a brief report with
visualizations, plots and answers. Denote the orig-
inal data set X100, which corresponds to using all
the features to capture 100% of the variance, and
let k100 = 256 denote the total dimension of the
original data set.
"""
#Loading the dataset
def TrainCumulativeSum(k,TrainEigenPairs,s):
    cs=0;
    for i in range(k):
        cs=cs+TrainEigenPairs[i][0]
    print("Percentage:",(cs/s)*100,"\b%")
    return cs

def TrainNumberOfEigenVal(Per,eigenPer):
    cs=0
    i=0
    while cs<Per:
        cs=cs+eigenPer[i]
        i+=1
    return i



x_test=np.loadtxt(fname="usps.test",dtype="int",delimiter=',',usecols = (range(1,257)))
x_train=np.loadtxt(fname="usps.train",dtype="int",delimiter=',',usecols = (range(1,257)))
x_valid=np.loadtxt(fname="usps.valid",dtype="int",delimiter=',',usecols = (range(1,257)))

y_test=np.loadtxt(fname="usps.test",dtype="int",delimiter=',',usecols = (0))
y_train=np.loadtxt(fname="usps.train",dtype="int",delimiter=',',usecols = (0))
y_valid=np.loadtxt(fname="usps.valid",dtype="int",delimiter=',',usecols = (0))
"""
print("\nX test",x_test)
print("\nX train",x_train)
print("\nX valid",x_valid)

print("\nY test",y_test)
print("\nY train",y_train)
print("\nY valid",y_valid)
"""
"""
a. Perform PCA on the training data and extract a full set of eigenvalues and eigenvectors.
Ensure that the eigenvalues and corresponding eigenvectors are sorted in decreasing order.
Visualize the top 16 eigendigits.
"""
#Finding the co-variance materix by with the help of transpose of X_test as an input to functionnp.cov
covariance_materix=np.cov(x_train.T)
print("\nCov\n",covariance_materix)
print("\nCov Size",covariance_materix.size)

eigenValues, eigenVectors=np.linalg.eig(covariance_materix)

print("\nEigen Values\n",eigenValues)
print("\nEigen Vectors\n",eigenVectors)

#Making a pair of eigen values and vectors
TrainEigenPairs=[(np.abs(eigenValues[i]),eigenVectors[:,i],i) for i in range(len(eigenValues))]
#print("PAir\n",TrainEigenPairs)
TrainEigenPairs.sort(key=lambda TrainEigenPairs: TrainEigenPairs[0])
TrainEigenPairs.reverse()

#print("Eigen Values in decreasing order")
#print("Eigen Values | ith feature | Percentage\n")
s=0
for i in range(len(eigenValues)):
    s=s+TrainEigenPairs[i][0]

eigenPer=[]
for i in range(len(eigenValues)):
    eigenPer.append((TrainEigenPairs[i][0]/s)*100)    
    print(TrainEigenPairs[i][0],TrainEigenPairs[i][2],eigenPer[i],"\b%")

#TrainEigenPairs[i][0] will denote eigenValue
#TrainEigenPairs[i][1] will denote the eigen vector
Mat16ev=[]
for i in range(16):
    Mat16ev.append(TrainEigenPairs[i][1])
eigenMat16ev=np.array(Mat16ev).T
#eigenMat16ev is d * l
print(x_train.shape)

print(eigenMat16ev.shape)
projectionMat16ev=np.matmul(x_train,eigenMat16ev)
print(projectionMat16ev.shape)
print(projectionMat16ev)
#projectionMat16ev is the Reduced data set with only 16 attributes/features
#Plotting the dataSet based on every attribute...
for i in range(16):
    py.plot(projectionMat16ev.T[i], len(projectionMat16ev.T[i]) * [1], "o")
    py.title("Attribute "+str(i+1))
    py.show() 
#print("Sum:",s)
#print("CS:",TrainCumulativeSum(TrainNumberOfEigenVal(100,eigenPer),TrainEigenPairs,s),"\b%")

"""
b. Plot the cumulative variance (cumulative sum of eigenvalues) vs. number of components.
What dimensionality does it take to achieve 70%, 80% and 90% of the total variance? Denote
these lower dimensionalities k70, k80 and k90.
"""
k70=TrainNumberOfEigenVal(70,eigenPer)
k80=TrainNumberOfEigenVal(80,eigenPer)
k90=TrainNumberOfEigenVal(90,eigenPer)
numberOfComponents=[k70,k80,k90]
print("Dimensionality to achieve 70% i.e k70:",k70)
print("Dimensionality to achieve 80% i.e k80:",k80)
print("Dimensionality to achieve 90% i.e k90:",k90)

cummularitveVariance=[TrainCumulativeSum(k70,TrainEigenPairs,s),
                      TrainCumulativeSum(k80,TrainEigenPairs,s),
                      TrainCumulativeSum(k90,TrainEigenPairs,s)]
    
py.plot(numberOfComponents,cummularitveVariance)
py.ylabel("Cummulative Variance")
py.xlabel("Number of Components")
print(numberOfComponents)
print(cummularitveVariance)
py.show()


"""
c. Use sklearn.linear model.SGDClassifier with settings loss='hinge' and penalty='l2'
to realize a linear support vector machine classi
er optimized via stochastic gradient descent.
This is a 10-class classi
cation problem, and SGDClassifier supports multi-class classi
-
cation by combining binary classi
ers in a one-vs-all (OVA) scheme1. For each of the three
projections and the original data set (i.e., k = k70; k80; k90; k100), perform the following steps: 
"""

"""
(a) Compute the projection Xf , by projecting the data on to the top kf eigenvectors, where
f = 70; 80; 90. For f = 100, simply use the original training set.
"""
Mat70=[]
for i in range(k70):
    Mat70.append(TrainEigenPairs[i][1])
eigenMat70=np.array(Mat70).T
#eigenMat70 is d * lprint()
projectionMat70=np.matmul(x_train,eigenMat70)
print(x_train.shape,"*",eigenMat70.shape,"=",projectionMat70.shape)

Mat80=[]
for i in range(k80):
    Mat80.append(TrainEigenPairs[i][1])
eigenMat80=np.array(Mat80).T
#eigenMat80 is d * lprint()
projectionMat80=np.matmul(x_train,eigenMat80)
print(x_train.shape,"*",eigenMat80.shape,"=",projectionMat80.shape)

Mat90=[]
for i in range(k90):
    Mat90.append(TrainEigenPairs[i][1])
eigenMat90=np.array(Mat90).T
#eigenMat90 is d * lprint()
projectionMat90=np.matmul(x_train,eigenMat90)
print(x_train.shape,"*",eigenMat90.shape,"=",projectionMat90.shape)

projectionMat100=x_train
print(projectionMat100.shape)


print("\n\n\n\n Img",eigenMat16ev[:][0].shape)

#py.imshow(np.reshape(np.matmul(projectionMat16ev,(eigenMat16ev.T))[0],((16,16))))
#py.show()
#x=np.array[(eigenMat16ev.T)[0]]
#print(x.shape)
for i in range(16):
    py.imshow(np.reshape((eigenMat16ev.T)[i],((16,16))))
    py.show()


"""
(b) Learn di
erent multi-class SVM classi
ers for each alpha 2 f0:0001; 0:001; 0:01; 0:1g. 
 cor-
responds to the weight on the regularization term and can be passed to SGDClassifier
via alpha=0.001.
"""

"""
Evaluate the learned SVM model on the validation set.
"""




#projectionMatk are the projections of Training data set similarly we need to generate projections on ValidationSet

validProjectionMat70=np.matmul(x_valid,eigenMat70)
validProjectionMat80=np.matmul(x_valid,eigenMat80)
validProjectionMat90=np.matmul(x_valid,eigenMat90)
validProjectionMat100=x_valid

testProjectionMat70=np.matmul(x_test,eigenMat70)
testProjectionMat80=np.matmul(x_test,eigenMat80)
testProjectionMat90=np.matmul(x_test,eigenMat90)
testProjectionMat100=x_test


print(x_train.shape,"*",eigenMat70.shape,"=",validProjectionMat70.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001)
classf.fit(projectionMat70,y_train)
#print(classf.predict(projectionMat70))
print("error",1-classf.score(validProjectionMat70,y_valid))
print("70test error",1-classf.score(testProjectionMat70,y_test))


print(x_train.shape,"*",eigenMat70.shape,"=",validProjectionMat70.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.001)
classf.fit(projectionMat70,y_train)
#print(classf.predict(projectionMat70))
print("error",1-classf.score(validProjectionMat70,y_valid))
print("70test error",1-classf.score(testProjectionMat70,y_test))


print(x_train.shape,"*",eigenMat70.shape,"=",validProjectionMat70.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.01)
classf.fit(projectionMat70,y_train)
#print(classf.predict(projectionMat70))
print("error",1-classf.score(validProjectionMat70,y_valid))
print("70test error",1-classf.score(testProjectionMat70,y_test))


print(x_train.shape,"*",eigenMat70.shape,"=",validProjectionMat70.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.1)
classf.fit(projectionMat70,y_train)
#print(classf.predict(projectionMat70))
print("error",1-classf.score(validProjectionMat70,y_valid))
print("70test error",1-classf.score(testProjectionMat70,y_test))


print(x_train.shape,"*",eigenMat80.shape,"=",validProjectionMat80.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001)
classf.fit(projectionMat80,y_train)
#print(classf.predict(projectionMat80))
print("error",1-classf.score(validProjectionMat80,y_valid))
print("80test error",1-classf.score(testProjectionMat80,y_test))


print(x_train.shape,"*",eigenMat80.shape,"=",validProjectionMat80.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.001)
classf.fit(projectionMat80,y_train)
#print(classf.predict(projectionMat80))
print("error",1-classf.score(validProjectionMat80,y_valid))
print("80test error",1-classf.score(testProjectionMat80,y_test))

print(x_train.shape,"*",eigenMat80.shape,"=",validProjectionMat80.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.01)
classf.fit(projectionMat80,y_train)
#print(classf.predict(projectionMat80))
print("error",1-classf.score(validProjectionMat80,y_valid))
print("80test error",1-classf.score(testProjectionMat80,y_test))

print(x_train.shape,"*",eigenMat80.shape,"=",validProjectionMat80.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.1)
classf.fit(projectionMat80,y_train)
#print(classf.predict(projectionMat80))
print("error",1-classf.score(validProjectionMat80,y_valid))
print("80test error",1-classf.score(testProjectionMat80,y_test))

print(x_train.shape,"*",eigenMat90.shape,"=",validProjectionMat90.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001)
classf.fit(projectionMat90,y_train)
#print(classf.predict(projectionMat90))
print("error",1-classf.score(validProjectionMat90,y_valid))
print("90test error",1-classf.score(testProjectionMat90,y_test))

print(x_train.shape,"*",eigenMat90.shape,"=",validProjectionMat90.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.001)
classf.fit(projectionMat90,y_train)

#print(classf.predict(projectionMat90))
print("error",1-classf.score(validProjectionMat90,y_valid))
print("90test error",1-classf.score(testProjectionMat90,y_test))

print(x_train.shape,"*",eigenMat90.shape,"=",validProjectionMat90.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.01)
classf.fit(projectionMat90,y_train)
#print(classf.predict(projectionMat90))
print("error",1-classf.score(validProjectionMat90,y_valid))
print("90test error",1-classf.score(testProjectionMat90,y_test))

print(x_train.shape,"*",eigenMat90.shape,"=",validProjectionMat90.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.1)
classf.fit(projectionMat90,y_train)
#print(classf.predict(projectionMat90))
print("error",1-classf.score(validProjectionMat90,y_valid))
print("90test error",1-classf.score(testProjectionMat90,y_test))

print(projectionMat100.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001)
classf.fit(projectionMat100,y_train)
#print(classf.predict(projectionMat100))
print("error",1-classf.score(validProjectionMat100,y_valid))
print("100test error",1-classf.score(testProjectionMat100,y_test))


print(projectionMat100.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.001)
classf.fit(projectionMat100,y_train)
#print(classf.predict(projectionMat100))
print("error",1-classf.score(validProjectionMat100,y_valid))
print("100test error",1-classf.score(testProjectionMat100,y_test))

print(projectionMat100.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.01)
classf.fit(projectionMat100,y_train)
#print(classf.predict(projectionMat100))
print("error",1-classf.score(validProjectionMat100,y_valid))
print("100test error",1-classf.score(testProjectionMat100,y_test))

print(projectionMat100.shape)
classf= linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=0.1)
classf.fit(projectionMat100,y_train)
#print(classf.predict(projectionMat100))
print("error",1-classf.score(validProjectionMat100,y_valid))
print("100test error",1-classf.score(testProjectionMat100,y_test))

"""
Report the validation error of each (kf ; 
) as a table. Note that k100 corresponds to the
original data set.
"""


"""
d. Report the error of the best (k; 
) pair on the test data? Explain how it compares to the
performance of the SVM without feature selection?
"""


