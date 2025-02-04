# 🚖 Reinforcement Learning: Taxi-V3 Simulation

## 📌 Overview
This is a **Reinforcement Learning (RL) simulation** using the **Q-Learning Algorithm** and **Gymnasium Library**. The agent (a taxi) is trained in a Gym environment to **pick up a passenger** and **drive to the destination** efficiently.

## 🧠 Learning Strategy
- **Q-Learning Algorithm (Bellman Equation)** updates the Q-table based on rewards.
- **Epsilon-Greedy Strategy** is used for exploration vs. exploitation.
- **Epsilon Decay** ensures the agent learns efficiently over time.
- **Hyperparameters:**
  - Learning Rate: `0.7`
  - Discount Factor: `0.95`
  - Episodes: `5000`
  - Max Steps per Episode: `100`

## 🏗️ Training Process
1. The agent starts randomly exploring actions.
2. Over 5000 episodes, it refines its Q-table.
3. After training, the agent exploits the learned policy to optimize navigation.

## 📜 Execution
```bash
python taxi_qlearning.py
```

## 🏆 Trained Agent Performance

```
Total Reward: -1
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
  (North)

Step 2
Total Reward: -2
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
  (North)

Step 3
Total Reward: -3
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
  (Pickup)

...

Step 12
Total Reward: 9
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
  (Dropoff)
```


## 🏁 Conclusion
The trained agent efficiently **navigates** the grid, picks up passengers, and drops them off while maximizing the reward. 🚖✨

