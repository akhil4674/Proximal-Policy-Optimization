# Proximal Policy Optimization (PPO) Algorithm Implementation 🚀

This code implements the **Proximal Policy Optimization (PPO)** algorithm using **TensorFlow** and **NumPy**. PPO is a reinforcement learning algorithm designed to train agents to make decisions in an environment through trial and error, ensuring that updates to the policy are stable and efficient.

## 📚 Important Classes

### 1. **PolicyNetwork**
The `PolicyNetwork` class defines the neural network architecture used by the agent to decide which action to take based on the current state. The architecture consists of:
- Two **dense layers** with **ReLU activation**.
- An **output layer** with **softmax activation** to produce action probabilities.

#### Methods:
- **`__init__`:** Initializes the policy network with specified layers.
- **`call`:** Defines the forward pass of the network, taking a state as input and returning action probabilities.

### 2. **PPO (Proximal Policy Optimization)**
The `PPO` class implements the core logic of the **PPO algorithm**. It handles action selection, training the policy network, and updating the agent's behavior based on collected experiences.

#### Methods:
- **`__init__`:** Initializes the PPO algorithm with hyperparameters like learning rate, clip ratio, and coefficients for value and entropy terms.
- **`get_action`:** Uses the policy network to select an action based on the current state, ensuring the agent chooses actions in a manner consistent with its learned policy.
- **`train`:** Updates the policy network’s weights based on the collected experience (states, actions, advantages, returns, and old action probabilities) using gradient descent and the PPO objective function.

---

## 🎯 Important Functions

### 1. **PolicyNetwork.__init__**
The constructor for the `PolicyNetwork` class initializes the network's architecture. It includes:
- Two hidden layers with ReLU activation for learning non-linear relationships in the state space.
- An output layer with a softmax activation that outputs action probabilities for each possible action.

### 2. **PolicyNetwork.call**
This method defines the **forward pass** of the network:
- It receives a state `s` as input.
- The state is passed through the hidden layers and the output layer.
- The output layer returns a probability distribution over the possible actions, from which the agent selects an action.

### 3. **PPO.__init__**
The constructor for the `PPO` class initializes several key hyperparameters:
- **Learning Rate:** Controls the magnitude of updates to the policy network.
- **Clip Ratio:** Ensures stable updates by clipping the probability ratio between old and new policies.
- **Value and Entropy Coefficients:** Used to balance the value function loss and the entropy term in the total objective function.

### 4. **PPO.get_action**
This method uses the trained `PolicyNetwork` to select an action from the current state. The action is chosen based on the highest probability output by the policy network, ensuring that the agent behaves according to the learned policy.

### 5. **PPO.train**
The `train` method is the core of the PPO training process. It:
- Collects **experience** (states, actions, advantages, returns, and old action probabilities) during interactions with the environment.
- Computes the **advantages** (estimating how much better taking an action at a given state is compared to the average action).
- Updates the policy network using the **PPO objective function**, which involves a clipped surrogate objective to ensure stable updates. The update is done using **gradient descent**.

---

## 🧠 Key Formulas

### PPO Objective Function

The core idea of PPO is to ensure that updates to the policy are **proximal**, i.e., not too large. The objective function used to update the policy is the **clipped surrogate objective**, which helps to avoid large policy changes. 

Given an old policy \( \pi_{\text{old}} \) and a new policy \( \pi_{\theta} \), the clipped objective is defined as:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

Where:
- \( r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\text{old}}(a_t | s_t)} \) is the probability ratio between the new and old policies.
- \( \hat{A}_t \) is the estimated advantage at time step \( t \).
- \( \epsilon \) is a hyperparameter that defines the clipping range (typically small, e.g., 0.2).

This objective function encourages the agent to avoid large updates while still improving its policy.

### Total Objective Function

The total objective function that combines the policy objective and the value function objective, including an entropy term for exploration, is:

$$
L^{\text{PPO}}(\theta) = \hat{\mathbb{E}}_t \left[ L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 H(\pi_{\theta}) \right]
$$

Where:
- \( L^{\text{VF}}(\theta) \) is the value function loss.
- \( H(\pi_{\theta}) \) is the entropy of the policy, encouraging exploration.
- \( c_1 \) and \( c_2 \) are coefficients that balance the different terms.

### Value Function Loss

The value function loss is typically defined as the mean squared error between the predicted value and the target value \( V_{\text{target}} \):

$$
L^{\text{VF}}(\theta) = \mathbb{E}_t \left[ \left( V_{\theta}(s_t) - V_{\text{target}}(s_t) \right)^2 \right]
$$

Where \( V_{\text{target}}(s_t) \) is the target value (often the return or discounted reward at time step \( t \)).

---

## 🧑‍💻 Requirements

- Python 3.x
- TensorFlow (for deep learning models)
- NumPy (for numerical operations)
- Matplotlib (optional, for visualizations)

## 📚 References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. [Paper](https://arxiv.org/abs/1707.06347)
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.
