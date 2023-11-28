import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import cv2
import gym


class PPO_Network(nn.Module):
    # nature paper architecture

    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.num_actions = num_actions

        network = [
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions + 1)
        ]

        self.network = nn.Sequential(*network)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        policy, value = torch.split(self.network(x), (self.num_actions, 1), dim=1)
        policy = self.softmax(policy)
        return policy, value
class PPO_Agent(nn.Module):

    def __init__(self, in_channels, num_actions):
        super().__init__()

        self.in_channels = in_channels
        self.num_actions = num_actions
        self.network = PPO_Network(in_channels, num_actions)

    def forward(self, x):
        policy, value = self.network(x)
        return policy, value

    def select_action(self, policy):
        return np.random.choice(range(self.num_actions), 1, p=policy)[0]


class Batch_DataSet(torch.utils.data.Dataset):

    def __init__(self, obs, actions, adv, v_t, old_action_prob):
        super().__init__()
        self.obs = obs
        self.actions = actions
        self.adv = adv
        self.v_t = v_t
        self.old_action_prob = old_action_prob

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, i):
        return self.obs[i], self.actions[i], self.adv[i], self.v_t[i], self.old_action_prob[i]
class Atari_Wrapper(gym.Wrapper):
    # env wrapper to resize images, grey scale and frame stacking and other misc.

    def __init__(self, env, env_name, k, dsize=(84, 84), use_add_done=False):
        super(Atari_Wrapper, self).__init__(env)
        self.dsize = dsize
        self.k = k
        self.use_add_done = use_add_done

        # set image cutout depending on game
        if "Breakout" in env_name:
            self.frame_cutout_h = (31, -16)
            self.frame_cutout_w = (7, -7)
        else:
            # no cutout
            self.frame_cutout_h = (0, -1)
            self.frame_cutout_w = (0, -1)

    def reset(self):

        self.Return = 0
        self.last_life_count = 0

        ob = self.env.reset()
        ob = self.preprocess_observation(ob)

        # stack k times the reset ob
        self.frame_stack = np.stack([ob for i in range(self.k)])

        return self.frame_stack

    def step(self, action):
        # do k frameskips, same action for every intermediate frame
        # stacking k frames

        reward = 0
        done = False
        additional_done = False

        # k frame skips or end of episode
        frames = []
        for i in range(self.k):

            ob, r, d, _ , add_info = self.env.step(action)
            # insert a (additional) done, when agent loses a life (Games with lives)
            if self.use_add_done:

                if add_info['lives'] < self.last_life_count:
                    additional_done = True
                self.last_life_count = add_info['lives']

            ob = self.preprocess_observation(ob)
            frames.append(ob)

            # add reward
            reward += r

            if d:  # env done
                done = True
                break

        # build the observation
        self.step_frame_stack(frames)

        # add info, get return of the completed episode
        self.Return += reward
        if done:
            add_info["return"] = self.Return

        # clip reward
        if reward > 0:
            reward = 1
        elif reward == 0:
            reward = 0
        else:
            reward = -1

        return self.frame_stack, reward, done, additional_done, add_info

    def step_frame_stack(self, frames):

        num_frames = len(frames)

        if num_frames == self.k:
            self.frame_stack = np.stack(frames)
        elif num_frames > self.k:
            self.frame_stack = np.array(frames[-k::])
        else:  # mostly used when episode ends

            # shift the existing frames in the framestack to the front=0 (0->k, index is time)
            self.frame_stack[0: self.k - num_frames] = self.frame_stack[num_frames::]
            # insert the new frames into the stack
            self.frame_stack[self.k - num_frames::] = np.array(frames)

    def preprocess_observation(self, ob):
        image = ob  # 'ob' is the image

        frame_length = len(image)
        end_index = frame_length + self.frame_cutout_h[1] if self.frame_cutout_h[1] < 0 else self.frame_cutout_h[1]
        # Check if the slicing indices are valid
        if self.frame_cutout_h[0] < end_index and self.frame_cutout_w[0] < self.frame_cutout_w[1]:
            sliced_image = image[self.frame_cutout_h[0]:end_index, self.frame_cutout_w[0]:self.frame_cutout_w[1]]
            # Proceed only if sliced image is not empty
            if sliced_image.size != 0:
                grayscale_image = cv2.cvtColor(sliced_image, cv2.COLOR_BGR2GRAY)
                resized_image = cv2.resize(grayscale_image, dsize=self.dsize)
                return resized_image
            else:
                print("Sliced image is empty.")
        else:
            return np.zeros(self.dsize)  # Return a black image if preprocessing fails


        return None  # Return None if preprocessing fails



