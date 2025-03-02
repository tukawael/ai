#!/usr/bin/env python
# coding: utf-8

# In[11]:


def sigmoid(x):
    return 1/(1+(2.71828**-x)) 
def sigmoid_derivative(x):
    return x*(1-x)
weights_input_hidden = [[0.15, 0.20], [0.25, 0.30]]  
weights_hidden_output = [[0.40, 0.45], [0.50, 0.55]]
bias_hidden = [0.35, 0.35]
bias_output = [0.60, 0.60]
inputs = [0.05, 0.10]
expected_output = [0.99, 0.01]  
learning_rate = 0.5

hidden_inputs = [inputs[0] * weights_input_hidden[0][0] + inputs[1] * weights_input_hidden[1][0] + bias_hidden[0],
                 inputs[0] * weights_input_hidden[0][1] + inputs[1] * weights_input_hidden[1][1] + bias_hidden[1]]

hidden_outputs = [sigmoid(hidden_inputs[0]), sigmoid(hidden_inputs[1])]

final_inputs = [hidden_outputs[0] * weights_hidden_output[0][0] + hidden_outputs[1] * weights_hidden_output[1][0] + bias_output[0],
    hidden_outputs[0] * weights_hidden_output[0][1] + hidden_outputs[1] * weights_hidden_output[1][1] + bias_output[1]]

final_outputs = [sigmoid(final_inputs[0]), sigmoid(final_inputs[1])]

total_error = 0.5 * ((expected_output[0]-final_outputs[0])**2 +(expected_output[1]-final_outputs[1])**2)

output_errors = [(expected_output[0] - final_outputs[0]) * sigmoid_derivative(final_outputs[0]),
    (expected_output[1] - final_outputs[1]) * sigmoid_derivative(final_outputs[1])]

for i in range(2):
    weights_hidden_output[0][i] += learning_rate * output_errors[i] * hidden_outputs[0]
    weights_hidden_output[1][i] += learning_rate * output_errors[i] * hidden_outputs[1]
    
hidden_errors = [ output_errors[0] * weights_hidden_output[0][0] + output_errors[1] * weights_hidden_output[0][1],
    output_errors[0] * weights_hidden_output[1][0] + output_errors[1] * weights_hidden_output[1][1]]

hidden_errors = [hidden_errors[0] * sigmoid_derivative(hidden_outputs[0]), hidden_errors[1] * sigmoid_derivative(hidden_outputs[1])]
for i in range(2):
    weights_input_hidden[0][i] += learning_rate * hidden_errors[i] * inputs[0]
    weights_input_hidden[1][i] += learning_rate * hidden_errors[i] * inputs[1]
    bias_hidden[i] += learning_rate * hidden_errors[i]
print("Updated weights (input to hidden):", weights_input_hidden)
print("Updated weights (hidden to output):", weights_hidden_output)


# In[ ]:




