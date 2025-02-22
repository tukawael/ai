#!/usr/bin/env python
# coding: utf-8

# In[36]:


import random

def tanh(x):
    return (2 / (1 + 2.718281828459045 ** (-2 * x))) - 1

def neural_network(inputs, weight1, weight2, bias1, bias2):
   
    weighted_sum1=sum([inputs[i] * weight1[i] for i in range(len(inputs))]) + b1
    output1=tanh(weighted_sum1)
    weighted_sum2=output1 * weight2[0] + b2
    output2=tanh(weighted_sum2)
    
    return output2

weight1=[random.uniform(-0.5, 0.5) for _ in range(3)]
weight2=[random.uniform(-0.5, 0.5)]                   
bias1=0.5  
bias2=0.7 

inputs=[0.5, 0.3, 0.2]
output=neural_network(inputs, weight1, weight2, bias1, bias2)
print("Output of the network:", output)


# In[ ]:




