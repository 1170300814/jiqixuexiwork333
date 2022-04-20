
import pandas as pd
from numpy.linalg import inv
from numpy import dot


iris = pd.read_csv('001.csv')
iris=iris.sample(frac=1.0)
iris=iris.reset_index()
temp = iris.iloc[:130, 0:4]

temp['x0'] = 1
X = temp.iloc[:, [0,1,2,3,4]]
Y = iris.iloc[:130, 4]
Y = Y.values.reshape(130, 1)

theta_n = dot(dot(inv(dot(X.T, X)), X.T), Y)

test=iris.iloc[130:,1:5]
testlabel=iris.iloc[130:,5]
print(testlabel)
rightnum=0
for i in range(130,150):
    score=-0.07631934*test['sepal length'][i]-0.0729171*test['sepal width'][i]+ 0.17031638*test['petal length'][i]+0.67070806*test['petal width'][i]+0.19540535
    print(round(score))
    if round(score)==testlabel[i]:
        rightnum+=1
print(theta_n)
print("rightratio is : ",int((rightnum/20)*100),'%')
