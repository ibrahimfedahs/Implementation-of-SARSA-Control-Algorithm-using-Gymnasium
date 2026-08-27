# Implementation of SARSA Control Algorithm using Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that enables the agent to select better actions for reaching the goal state while avoiding holes.

## Problem Statement

Implement the SARSA reinforcement learning control algorithm for the `FrozenLake-v1` environment in Gymnasium. The agent should learn an optimal action-selection policy through repeated interaction with the environment using an epsilon-greedy exploration strategy.

## Software Requirements

* Python 3.x
* Gymnasium
* NumPy
* Jupyter Notebook / Google Colab / Python IDE

## Environment Description

The `FrozenLake-v1` environment consists of a 4 × 4 grid. The agent starts from the initial state and must reach the goal while avoiding holes.

The environment contains:

* **S** – Starting position
* **F** – Frozen/safe surface
* **H** – Hole
* **G** – Goal

There are four possible actions:

* `0` – Left
* `1` – Down
* `2` – Right
* `3` – Up

The agent receives a reward of **1** for reaching the goal and **0** for other transitions.

## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

SARSA is an **on-policy temporal-difference control algorithm**. It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]
$$

Where:

| Symbol    | Meaning                                       |
| --------- | --------------------------------------------- |
| $S_t$     | Current state                                 |
| $A_t$     | Current action                                |
| $R_{t+1}$ | Reward received after taking action $A_t$     |
| $S_{t+1}$ | Next state                                    |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$  | Learning rate                                 |
| $\gamma$  | Discount factor                               |
| $Q(s,a)$  | Action-value function                         |

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

Initially, epsilon is kept relatively high so that the agent explores different actions. It is gradually decreased during training so that the agent increasingly exploits the learned Q-values.

## Algorithm

1. Create the `FrozenLake-v1` environment.
2. Initialize the Q-table with zeros.
3. Set the learning rate $\alpha$, discount factor $\gamma$, and exploration rate $\epsilon$.
4. For each training episode:

   * Reset the environment.
   * Select the initial action using the epsilon-greedy policy.
5. Repeat until the episode terminates:

   * Execute the selected action.
   * Observe the next state and reward.
   * Select the next action using the epsilon-greedy policy.
   * Update the Q-value using the SARSA update equation.
   * Move to the next state and action.
6. Decrease epsilon after every episode.
7. Repeat the process for the required number of episodes.
8. Calculate the state-value function from the learned Q-table.
9. Extract the learned policy by selecting the action with the maximum Q-value for each state.
10. Evaluate the learned policy and calculate the average reward.

## Python Program

```python
import gymnasium as gym
import numpy as np

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Environment parameters
state_size = env.observation_space.n
action_size = env.action_space.n

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 10000

# Initialize Q-table
Q = np.zeros((state_size, action_size))


# Epsilon-greedy action selection
def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# Training
rewards = []

for episode in range(episodes):

    state, info = env.reset()

    # Select initial action
    action = choose_action(state, epsilon)

    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Select next action
        if not (terminated or truncated):
            next_action = choose_action(next_state, epsilon)

            # SARSA update
            Q[state, action] = Q[state, action] + alpha * (
                reward
                + gamma * Q[next_state, next_action]
                - Q[state, action]
            )

            # Move to next state and action
            state = next_state
            action = next_action

        else:
            # Terminal-state SARSA update
            Q[state, action] = Q[state, action] + alpha * (
                reward - Q[state, action]
            )

        total_reward += reward

    rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -------------------------------------------------
# Calculate State-Value Function
# -------------------------------------------------

V = np.max(Q, axis=1)


# -------------------------------------------------
# Extract Learned Policy
# -------------------------------------------------

policy = np.argmax(Q, axis=1)

action_symbols = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}

print("Final Q-table:")
print(np.round(Q, 3))

print("\nEstimated State-Value Function:")
print(np.round(V, 3))

print("\nLearned Policy:")

for state in range(state_size):
    print(
        f"State {state:2d} : "
        f"{action_symbols[policy[state]]}"
    )


# -------------------------------------------------
# Average Reward
# -------------------------------------------------

average_reward = np.mean(rewards[-1000:])

print(
    "\nAverage reward over last 1000 episodes:",
    round(average_reward, 3)
)

env.close()
```

## Output
Final Q-table:

<img width="389" height="326" alt="image" src="https://github.com/user-attachments/assets/506fd4cc-0134-42f3-9bc2-43d3a1d0de52" />

Estimated State-Value Function:

<img width="628" height="80" alt="image" src="https://github.com/user-attachments/assets/dc014527-5798-4260-b254-9c71f60d8366" />

Learned Policy:
 
  <img width="514" height="329" alt="image" src="https://github.com/user-attachments/assets/4c6a3ebe-7b91-4b05-b2c9-936c5f23116c" />

Average reward over last 1000 episodes: 1.0



```

**Note:** The exact Q-table values and learned actions can vary because SARSA uses random exploration. The output above is an example of a successful training run.

## Result

```text
Thus, the SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned an action-value function and an epsilon-greedy policy that enables it to reach the goal while avoiding the holes.
```

## Inference

```text
The experiment demonstrates that SARSA can learn an effective policy through repeated interaction with the environment. As training progresses, the Q-values become more accurate and the agent increasingly selects actions that lead toward the goal. The epsilon-greedy strategy allows the agent to balance exploration and exploitation during learning.
```
