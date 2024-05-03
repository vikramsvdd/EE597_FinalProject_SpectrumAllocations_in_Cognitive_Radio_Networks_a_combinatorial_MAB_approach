
import numpy as np
import matplotlib.pyplot as plt

class BanditProblem:
    def __init__(self, true_action_values, total_steps, strategy='epsilon_greedy', epsilon=0.1, confidence_level=2):
        self.arm_number = len(true_action_values)
        self.total_steps = total_steps
        self.true_action_values = true_action_values
        self.strategy = strategy
        self.epsilon = epsilon
        self.confidence_level = confidence_level
        self.reset()

    def reset(self):
        self.current_step = 0
        self.arm_selection_counts = np.zeros(self.arm_number)
        self.estimated_values = np.zeros(self.arm_number)
        self.mean_rewards = np.zeros(self.total_steps + 1)
        self.optimal_action_count = np.zeros(self.total_steps + 1)

    def select_action(self):
        if self.strategy == 'epsilon_greedy':
            if np.random.rand() < self.epsilon or self.current_step == 0:
                return np.random.randint(self.arm_number)
            return np.argmax(self.estimated_values)
        elif self.strategy == 'UCB':
            if self.current_step < self.arm_number:
                return self.current_step
            ucb_values = self.estimated_values + np.sqrt(self.confidence_level * np.log(self.current_step + 1) / self.arm_selection_counts)
            return np.argmax(ucb_values)
        elif self.strategy == 'thompson_sampling':
            samples = np.random.normal(self.estimated_values, np.sqrt(1 / (self.arm_selection_counts + 1)))
            return np.argmax(samples)

    def simulate(self):
        optimal_arm = np.argmax(self.true_action_values)
        for i in range(self.total_steps):
            arm = self.select_action()
            reward = np.random.normal(self.true_action_values[arm], 1)
            self.arm_selection_counts[arm] += 1
            alpha = 1 / self.arm_selection_counts[arm]
            self.estimated_values[arm] += alpha * (reward - self.estimated_values[arm])

            self.mean_rewards[i + 1] = self.mean_rewards[i] + (reward - self.mean_rewards[i]) / (self.current_step + 1)
            self.optimal_action_count[i + 1] = self.optimal_action_count[i] + (1 if arm == optimal_arm else 0)
            self.current_step += 1
        self.optimal_action_count = self.optimal_action_count / np.arange(1, self.total_steps + 2)

def run_simulations(action_values, total_steps, strategies):
    results = {}
    for strategy, params in strategies.items():
        bandit = BanditProblem(action_values, total_steps, strategy, **params)
        bandit.simulate()
        results[strategy] = (bandit.mean_rewards, bandit.optimal_action_count)
    return results

def plot_results(results, total_steps):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    for strategy, (mean_rewards, optimal_actions) in results.items():
        ax[0].plot(range(total_steps + 1), mean_rewards, label=f'{strategy}')
        ax[1].plot(range(total_steps + 1), optimal_actions, label=f'{strategy} - Optimal Action %')
    ax[0].set_xscale('log')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Average Reward')
    ax[0].legend()
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('Optimal Action %')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('multi_strategy_results.png', dpi=300)
    plt.show()

action_values = np.array([1, 4, 2, 0, 7, 1, -1])
total_steps = 100000
strategies = {
    'epsilon_greedy': {'epsilon': 0.1},
    'UCB': {'confidence_level': 2},  # Adjust confidence_level as needed
    'thompson_sampling': {}
}
results = run_simulations(action_values, total_steps, strategies)
plot_results(results, total_steps)
