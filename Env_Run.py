import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cpu")
dtype = torch.float


class Logger:

    def __init__(self, filename):
        self.filename = filename

        f = open(f"{self.filename}.csv", "w")
        f.close()

    def log(self, msg):
        f = open(f"{self.filename}.csv", "a+")
        f.write(f"{msg}\n")
        f.close()


cur_step = 0


class Env_Runner:

    def __init__(self, env, agent, logger_folder):
        super().__init__()

        self.env = env
        self.agent = agent

        self.logger = Logger(f'{logger_folder}/training_info')
        self.logger.log("training_step, return")

        self.ob = self.env.reset()

    def run(self, steps):

        global cur_step

        obs = []
        actions = []
        rewards = []
        dones = []
        values = []
        action_prob = []

        for step in range(steps):
            # Convert the observation to a tensor and reshape it
            if self.ob is not None:
                self.ob = torch.tensor(self.ob, dtype=dtype, device=device)

                # Ensure the tensor is 4D: [N, C, H, W] - N is batch size (1 here)
                self.ob = self.ob.unsqueeze(0)  # Add batch dimension

                policy, value = self.agent(self.ob)
                action = self.agent.select_action(policy.detach().cpu().numpy()[0])


                obs.append(self.ob)
                actions.append(action)
                values.append(value.detach())
                action_prob.append(policy[0, action].detach())

                self.ob, r, done, additional_done, add_info= self.env.step(action)

                if done:
                    self.ob = self.env.reset()
                    # Handle case where reset returns None
                    if self.ob is None:
                        continue

                    if "return" in add_info:
                        self.logger.log(f'{cur_step + step},{add_info["return"]}')

            rewards.append(r)
            dones.append(done or additional_done)

        cur_step += steps

        return [obs, actions, rewards, dones, values, action_prob]