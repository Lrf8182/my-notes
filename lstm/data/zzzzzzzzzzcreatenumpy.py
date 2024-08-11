import numpy as np

# Define the shape of the arrays
data_shape = (1024, 32, 128)
labels_shape = (1024,)

# Create 'complex-data.npy' with random floats between 0 and 1
complex_data = np.random.rand(*data_shape)

# Create 'complex-labels.npy' with random integers between 0 and 5
complex_labels = np.random.randint(0, 6, size=labels_shape)

# Save the arrays to .npy files
np.save('complex-data.npy', complex_data)
np.save('complex-labels.npy', complex_labels)

print("Files 'complex-data.npy' and 'complex-labels.npy' have been created successfully.")
