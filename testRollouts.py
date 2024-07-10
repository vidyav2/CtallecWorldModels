import os
import argparse
import numpy as np
import imageio
import gym

def verify_rollout(file_path):
    data = np.load(file_path)
    observations = data['observations']
    rewards = data['rewards']
    actions = data['actions']
    terminals = data['terminals']
    
    print(f"Contents of {file_path}:")
    #print(f"Observations shape: {observations.shape}")
    #print(f"Rewards shape: {rewards.shape}")
    print(f"Actions shape: {actions}")
    #print(f"Terminals shape: {terminals.shape}")
    print(f"Rewards: {rewards}")
    #print(f"Terminals: {terminals}")

    #assert observations.shape == (1000, 96, 96, 3), "Incorrect observations shape"
    #assert rewards.shape == (1000,), "Incorrect rewards shape"
    #assert actions.shape == (1001, 3), "Incorrect actions shape"
    #assert terminals.shape == (1000,), "Incorrect terminals shape"

def create_gif(file_path, output_gif):
    data = np.load(file_path)
    observations = data['observations']
    
    with imageio.get_writer(output_gif, mode='I', duration=0.1) as writer:
        for frame in observations:
            writer.append_data(frame)
    print(f"GIF saved to {output_gif}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="Directory containing rollouts")
    parser.add_argument('--output_gif', type=str, help="Path to save the output GIF")
    args = parser.parse_args()
    
    for file_name in os.listdir(args.data_dir):
        if file_name.endswith('.npz'):
            file_path = os.path.join(args.data_dir, file_name)
            verify_rollout(file_path)
            create_gif(file_path, args.output_gif)
