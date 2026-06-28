"""
Test script for Module 3: CNN Base Model Module
"""

import numpy as np
from cnn_model import build_base_cnn, get_model_summary_dict

# Generate random dummy input: batch=8, timesteps=10, features=7
X_dummy = np.random.rand(8, 10, 7).astype(np.float32)

# Build model
model = build_base_cnn((10, 7))

# Retrieve summary info
summary = get_model_summary_dict(model)

print("Model Built Successfully")
print(f"Input Shape: {summary['input_shape']}")
print(f"Output Shape: {summary['output_shape']}")
print(f"Total Params: {summary['total_params']}")
print(f"Layer Names: {summary['layer_names']}")

# Forward pass
y_pred = model.predict(X_dummy)
print(f"Prediction Shape: {y_pred.shape}")
