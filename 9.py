
# coding: utf-8

# In[47]:


import numpy as np
from sklearn import datasets


# In[48]:


iris = load_iris()
targets = iris.target_names


# In[ ]:


print("Class : Number\nsetosa : 0\nversicolor : 1\nvirginica : 2")


# In[54]:


x_train, x_test, y_train, y_test = train_test_split(iris["data"],iris["target"])
kn = KNeighborsClassifier(2.5).fit(x_train, y_train)


# In[ ]:


for i in range(len(x_test)):
    x_new = np.array([x_test[i]])
    pred = kn.predict(x_new)
    print("Actual:[{0}] [{1}],Predicted:{2} {3}".format(y_test[i], targets[y_test[i]], pred, targets[pred]))
print("\nAccuracy: ", kn.score(x_test, y_test))

