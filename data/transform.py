"""import json
import numpy as np

# Load the original JSON file
input_file = 'data/carracing.cma.16.64.best.json'
output_file = 'data/transformed_carracing.json'

with open(input_file, 'r') as f:
    data = json.load(f)

# Inspect and transform the data if necessary
parameters = np.array(data[0])
reward = data[1]

# Verify the structure
print("Parameters Length:", len(parameters))
print("Sample Parameters:", parameters[:10])
print("Reward:", reward)

# If parameters need reshaping or other transformations, do it here
# For now, assuming the data is correctly structured and directly usable

# Save the transformed data
transformed_data = [parameters.tolist(), reward]
with open(output_file, 'w') as f:
    json.dump(transformed_data, f)

print(f"Transformed data saved to {output_file}")"""


import gymnasium as gym
import numpy as np

def test_environment():
    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    obs, _ = env.reset()
    env.render()

    seq_len = 1000
    for t in range(seq_len):
        action = env.action_space.sample()  # Use random action for simplicity
        action = np.array(action, dtype=np.float64)  # Ensure action is float64

        # Print the action and its type
        print(f"Action: {action}, Type: {type(action)}, Dtype: {action.dtype}")

        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        if terminated:
            break

if __name__ == "__main__":
    test_environment()


