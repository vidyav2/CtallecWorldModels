import argparse
from os.path import join, exists
import gymnasium as gym
import numpy as np
from utils.misc import sample_continuous_policy
from PIL import Image

def generate_data(rollouts, data_dir, noise_type, seq_len=1000, gif_file='rollout.gif'):
    """ Generates data and saves rollouts as a GIF """
    assert exists(data_dir), "The data directory does not exist..."

    env = gym.make("CarRacing-v2", render_mode='rgb_array', lap_complete_percent=1.0)
    frames = []

    for i in range(rollouts):
        obs = env.reset()
        frame = env.render()
        frames.append(Image.fromarray(frame))

        if noise_type == 'white':
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == 'brown':
            a_rollout = sample_continuous_policy(env.action_space, seq_len, env)

        s_rollout = []
        r_rollout = []
        d_rollout = []

        t = 0
        while True:
            action = a_rollout[t]
            t += 1

            s, r, done, truncated, info = env.step(action)
            s_rollout.append(s)
            r_rollout.append(r)
            d_rollout.append(done or truncated)

            # Capture the frame and add it to the frames list
            frame = env.render()
            frames.append(Image.fromarray(frame))

            if done or truncated or t >= seq_len:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

    # Save frames as a GIF
    frames[0].save(gif_file, save_all=True, append_images=frames[1:], loop=0, duration=40)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    parser.add_argument('--gif', type=str, help="Output GIF file", default='rollout.gif')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy, gif_file=args.gif)
