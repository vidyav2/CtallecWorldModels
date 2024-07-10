import argparse
import json
import numpy as np
from os.path import join, exists
import gymnasium as gym
import torch
from torchvision import transforms
from models import VAE, MDRNNCell, Controller

# Define the transform to resize the image to 64x64
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def load_parameters_from_json(json_file, controller):
    with open(json_file, 'r') as f:
        data = json.load(f)
    params = np.array(data[0])  # Assuming the parameters are stored in the first element of the JSON array
    
    # Flatten the parameters to match the controller's state dict
    flat_params = params.flatten()
    state_dict = controller.state_dict()
    index = 0

    for name, param in state_dict.items():
        param_length = param.numel()
        param_data = flat_params[index:index + param_length]
        param_tensor = torch.tensor(param_data, dtype=torch.float32).view(param.shape)
        state_dict[name].copy_(param_tensor)
        index += param_length

    controller.load_state_dict(state_dict)

def generate_data(rollouts, data_dir, pre_trained_file):
    """Generates data"""
    assert exists(data_dir), "The data directory does not exist..."

    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    print("Action space dtype:", env.action_space.dtype)  # Check the action space data type
    seq_len = 1000

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(3, 32).to(device)
    mdrnn = MDRNNCell(32, 3, 256, 5).to(device)
    controller = Controller(32, 256, 3).to(device)

    # Load parameters into the PyTorch controller model from reshaped JSON
    load_parameters_from_json(pre_trained_file, controller)

    for i in range(rollouts):
        obs, _ = env.reset()
        env.render()

        if isinstance(obs, dict):
            obs = obs.get('image', obs.get('observation', None))
            if obs is None:
                raise ValueError("Could not find the image data in the observation dictionary")

        s_rollout = []
        r_rollout = []
        a_rollout = []
        d_rollout = []

        hidden = [torch.zeros(1, 256).to(device) for _ in range(2)]
        t = 0
        while True:
            if isinstance(obs, dict):
                obs = obs.get('image', obs.get('observation', None))
                if obs is None:
                    raise ValueError("Could not find the image data in the observation dictionary")

            if obs is None:
                raise ValueError("Could not find the image data in the observation dictionary")

            obs = transform(obs).unsqueeze(0).float().to(device)

            _, latent_mu, _ = vae(obs)
            action = controller(latent_mu, hidden[0])
            action = action.squeeze().detach().cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float64)  # Ensure float64

            # Print the action and its type
            #print(f"Action: {action}, Type: {type(action)}, Dtype: {action.dtype}")

            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()

            if isinstance(obs, dict):
                obs = obs.get('image', obs.get('observation', None))
                if obs is None:
                    raise ValueError("Could not find the image data in the observation dictionary")

            if obs is None:
                raise ValueError("Could not find the image data in the observation dictionary")

            s_rollout.append(obs)
            r_rollout.append(reward)
            d_rollout.append(terminated)
            a_rollout.append(action)

            if terminated or t >= seq_len:
                print(f"> End of rollout {i}, {len(s_rollout)} frames...")
                np.savez(join(data_dir, f'rollout_{i}'),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

            t += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--pretrained', type=str, help="Path to the pre-trained model parameters")
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.pretrained)
