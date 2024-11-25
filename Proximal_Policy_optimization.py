import tensorflow as tf
import numpy as np

# Define the neural network architecture
class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# Define the PPO algorithm
class PPO:
    def __init__(self, num_actions, learning_rate=0.001, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        self.policy_network = PolicyNetwork(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def get_action(self, state):
        action_probs = self.policy_network(state)
        action = np.random.choice(range(len(action_probs.numpy()[0])), p=action_probs.numpy()[0])
        return action, action_probs

    def train(self, states, actions, advantages, returns, old_action_probs):
        with tf.GradientTape() as tape:
            action_probs = self.policy_network(states, training=True)
            action_masks = tf.one_hot(actions, len(action_probs[0]))
            selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            ratio = tf.exp(tf.math.log(selected_action_probs) - tf.math.log(old_action_probs))

            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            value_preds = self.value_network(states)
            value_loss = tf.reduce_mean(tf.square(returns - value_preds))

            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs))
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        gradients = tape.gradient(total_loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
