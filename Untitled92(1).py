#!/usr/bin/env python
# coding: utf-8

# In[5]:


import random
def tanh(x):
    return (2 / (1 + 2.718281828459045 ** (-2 * x))) - 1

weights1 = [random.uniform(-0.5, 0.5) for _ in range(2)] 
weights2 = [random.uniform(-0.5, 0.5)]                   
weights3 = [random.uniform(-0.5, 0.5)]                  
b1 = 0.5 
b2 = 0.7 

inputs = [0.05, 0.1]

def neural_network(inputs, weights1, weights2, weights3, b1, b2, b3):
    weighted_sum1 = sum([inputs[i] * weights1[i] for i in range(len(inputs))]) + b1
    hidden_output = tanh(weighted_sum1)
    weighted_sum2 = hidden_output * weights2[0] + b2
    output1 = tanh(weighted_sum2)
    weighted_sum3 = hidden_output * weights3[0] + b2
    output2 = tanh(weighted_sum3)
    
    return output1, output2


output1, output2 = neural_network(inputs, weights1, weights2, weights3, b1, b2, b3)

print("output of the first output layer:", output1)
print("output of the second output layer:", output2)


# In[ ]:





# In[ ]:




