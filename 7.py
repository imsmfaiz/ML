
# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd
import csv
from  pgmpy import *


# In[55]:


disease = pd.read_csv('heart.csv')
disease = disease.replace('?', np.nan)


# In[56]:


model=BayesianModel([
                                                ('age','trestbps'),
                                                ('age','fbs'),
                                                ('sex','trestbps'),
                                                ('trestbps','heartdisease'),
                                                ('fbs','heartdisease'),
                                                ('chol','heartdisease')
                                            ])


# In[ ]:


print('\nLearning CPDs using maximum likelihood estimators...')
model.fit(disease,estimator=MaximumLikelihoodEstimator)


# In[ ]:


print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)


# In[ ]:


print('\n1.Probability of HeartDisease given Age=20')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 20})
print(q['heartdisease'])


# In[ ]:


print('\n2. Probability of HeartDisease given chol (Cholesterol) =600')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 151})
print(q['heartdisease'])

