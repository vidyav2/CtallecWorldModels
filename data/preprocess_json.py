import json
import numpy as np
import torch

# Load the JSON file
with open('carracing.cma.16.64.best.json', 'r') as f:
    pre_trained_data = json.load(f)

# Extract parameters and reshape
parameters = np.array(pre_trained_data[0])
reward = pre_trained_data[1]

# Assuming the parameters are a flat array, reshape according to the model's layers
# This example assumes a single fully connected layer with input size 288 and output size 3
# Modify according to your actual model architecture

fc_weight_shape = (3, 288)  # Example shape for the fully connected layer weight
fc_bias_shape = (3,)        # Example shape for the fully connected layer bias

# Split parameters for fc_weight and fc_bias
fc_weight_params = parameters[:np.prod(fc_weight_shape)].reshape(fc_weight_shape)
fc_bias_params = parameters[np.prod(fc_weight_shape):np.prod(fc_weight_shape) + np.prod(fc_bias_shape)].reshape(fc_bias_shape)

# Create a dictionary to hold the reshaped parameters
reshaped_params = {
    'fc.weight': torch.tensor(fc_weight_params, dtype=torch.float32).tolist(),
    'fc.bias': torch.tensor(fc_bias_params, dtype=torch.float32).tolist()
}

# Save the reshaped parameters to a new JSON file
with open('reshaped_parameters.json', 'w') as f:
    json.dump(reshaped_params, f)

print("Preprocessing complete. Reshaped parameters saved to 'reshaped_parameters.json'")
