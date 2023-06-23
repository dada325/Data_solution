# Data Solution Code
Find question to practise with.
Find practical solutions to your coding problems in data science.


## About this Page




### What will I build in this page?





## Description
When I first confront a problem of data science, I want to try to solve it with the knowledge from the book. But most of time it fails, the problem may be too complex and not easy to abstract to a computer problem. 

We understand the challenges and complexities that data scientists face while working on various projects. Our mission is to provide a comprehensive platform where data science enthusiasts and professionals can find practical solutions to their coding problems.

We want to cover a wide range of topics, including data preprocessing, data visualization, machine learning algorithms, statistical analysis, deep learning, natural language processing, Reinforcement learning and more. We believe in hands-on learning, so you'll find plenty of interactive coding exercises and projects that allow you to apply your newfound knowledge. Now the Contents is still Work in progress.

## Status:
 now WIP is the section of Reinforcement learning, so please feel free to fork of other section. 

# Machine Learning 

## Supervised Learning

## Unsupervised Learning

## Deep Learning

## Natural Language Processing (NLP)

## Computer Vision

## Recommender Systems

## Time Series Forecasting

## Transfer Learning

## Adversarial Machine Learning

## AutoML (Automated Machine Learning)

## Reinforcement Learning.


 0. The Basic

 ### Some of the library
 

 | Library | SOTA RL Algorithms | Documentation & Tutorials | Code Customization | Supported Environments | Logging & Tracking Tools | VE Feature | Regular Updates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KerasRL | ✔️ | Partial | ✔️ | Limited | Partial | ✔️ | ❌ |
| Pyqlearning | Limited | Partial | Partial | ✔️ | ❌ | ❌ | ✔️ |
| Tensorforce | Almost all | ✔️ | ✔️ | Multiple | ✔️ | ✔️ | ✔️ |
| RL_Coach | Most complete | ✔️ | Partial | Multiple | ✔️ | ✔️ | ✔️ |
| TFAgents | Great set | Partial | ✔️ | Agnostic | ✔️ | ✔️ | ✔️ |
| Stable Baselines | Great set | ✔️ | Partial | OpenAI Gym | ✔️ | ✔️ | ✔️ |
| MushroomRL | Good set | Partial | Partial | Multiple | ✔️ | ✔️ | ✔️ |
| RLlib | All | ✔️ | Partial | Multiple | ✔️ | ✔️ | ✔️ |
| Dopamine | DQN based | ✔️ | ✔️ | OpenAI Gym | ✔️ | ❌ | ✔️ |
| SpinningUp | VPG, PPO, TRPO, DDPG, TD3, SAC | ✔️ | ✔️ | OpenAI Gym | Partial | ❌ | ✔️ |


For a detailed overview of each library, please refer to the [original article](https://neptune.ai/blog/the-best-tools-for-reinforcement-learning-in-python).

Conclusion
The choice of library depends on your specific needs and requirements. Some libraries like RL_Coach and RLlib offer a wide range of algorithms and are regularly updated. Others like KerasRL and Pyqlearning have limited features but are easy to understand and use. TFAgents and Stable Baselines are promising libraries with a great set of algorithms and active development. Dopamine and SpinningUp are designed for fast prototyping and are highly customizable.


 1. The Intersection of Reinforcement Learning and other discipline
      * Economics
      * Psychology
      * Neuroscience
      * Computer Science
      * Engineering
      * Mathematic
 
 2. Some of the Paper reviews


| Algorithm                                       | Authors           | Year |
|-------------------------------------------------|-------------------|------|
| [A2C / A3C (Asynchronous Advantage Actor-Critic)](https://arxiv.org/abs/1602.01783) | Mnih et al        | 2016 |
| [PPO (Proximal Policy Optimization)](https://arxiv.org/abs/1707.06347)              | Schulman et al    | 2017 |
| [TRPO (Trust Region Policy Optimization)](https://arxiv.org/abs/1502.05477)         | Schulman et al    | 2015 |
| [DDPG (Deep Deterministic Policy Gradient)](https://arxiv.org/abs/1509.02971)       | Lillicrap et al   | 2015 |
| [TD3 (Twin Delayed DDPG)](https://arxiv.org/abs/1802.09477)                         | Fujimoto et al    | 2018 |
| [SAC (Soft Actor-Critic)](https://arxiv.org/abs/1801.01290)                         | Haarnoja et al    | 2018 |
| [DQN (Deep Q-Networks)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)                           | Mnih et al        | 2013 |
| [C51 (Categorical 51-Atom DQN)](https://arxiv.org/abs/1707.06887)                   | Bellemare et al   | 2017 |
| [QR-DQN (Quantile Regression DQN)](https://arxiv.org/abs/1710.10044)               | Dabney et al      | 2017 |
| [HER (Hindsight Experience Replay)](https://arxiv.org/abs/1707.01495)               | Andrychowicz et al| 2017 |
| [World Models](https://worldmodels.github.io/)                                  | Ha and Schmidhuber| 2018 |
| [I2A (Imagination-Augmented Agents)](https://arxiv.org/abs/1707.06203)              | Weber et al       | 2017 |
| [MBMF (Model-Based RL with Model-Free Fine-Tuning)](https://sites.google.com/view/mbmf)| Nagabandi et al  | 2017 |
| [MBVE (Model-Based Value Expansion)](https://arxiv.org/abs/1803.00101)              | Feinberg et al    | 2018 |
| [AlphaZero](https://arxiv.org/abs/1712.01815)                                      | Silver et al      | 2017 |
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)|LeCun et al| 2022 | 
| [DreamerV3](https://arxiv.org/pdf/2301.04104.pdf)                                  | Hafner et al      | 2023|

  3. Model & Algorithm
      First I will interpret the algorithm in the book "Reinforcement learning: an Introduction"

| Chapter | Algorithms |
| ------- | ---------- |
| 2: Multi-armed Bandits | - Epsilon-Greedy Algorithm <br> - Upper-Confidence-Bound Action Selection |
| 3: Finite Markov Decision Processes | - Policy Iteration <br> - Value Iteration |
| 4: Dynamic Programming | - Iterative Policy Evaluation <br> - Policy Improvement <br> - Policy Iteration <br> - Value Iteration |
| 5: Monte Carlo Methods | - First-Visit Monte Carlo Policy Evaluation <br> - Monte Carlo Exploring Starts <br> - On-Policy First-Visit MC Control (for epsilon-soft policies) |
| 6: Temporal-Difference Learning | - TD(0) <br> - Sarsa (on-policy TD control) <br> - Q-learning (off-policy TD control) |
| 7: n-step Bootstrapping | - n-step TD <br> - n-step Sarsa <br> - n-step Off-policy Learning by Importance Sampling |
| 8: Planning and Learning with Tabular Methods | - Dyna-Q <br> - Dyna-Q+ |
| 9: On-policy Prediction with Approximation | - Gradient Monte Carlo Algorithm <br> - Semi-gradient TD(0) |
| 10: On-policy Control with Approximation | - Episodic Semi-gradient Sarsa <br> - Semi-gradient n-step Sarsa |
| 11: Off-policy Methods with Approximation | - Semi-gradient DP <br> - Semi-gradient Q(σ) <br> - Differential Semi-gradient Sarsa |
| 12: Eligibility Traces | - TD(λ) <br> - Sarsa(λ) <br> - True Online TD(λ) <br> - True Online Sarsa(λ) |
| 13: Policy Gradient Methods | - REINFORCE: Monte Carlo Policy Gradient <br> - Actor-Critic <br> - REINFORCE with Baseline <br> - Actor-Critic with Baseline |

      
  5. Task
      * Navigation
      * Continous Control
      * 
  6. Case Study






## Main task

- Mathmatic
- Data Processing
- Machine Learning 
- Deep Learning
- Reinforcement Learning


## Learning JAX 

| Name | Maintainer | Summary | Link |
| --- | --- | --- | --- |
| Flax | Google Brain | A high-level neural network library designed for flexibility. | [Flax](https://github.com/google/flax) |
| Haiku | Deepmind | A JAX-based neural network library, aka Sonnet for JAX. | [Haiku](https://github.com/deepmind/dm-haiku) |
| Trax | Google Brain | An end-to-end library for deep learning that focuses on clear code and speed. | [Trax](https://github.com/google/trax) |
| Objax | Google | A minimalist object-oriented framework with a PyTorch-like interface. | [Objax](https://github.com/google/objax) |
| Stax | JAX | A small but flexible neural net specification library from scratch. | [Stax](https://jax.readthedocs.io/en/latest/jax.experimental.stax.html) |
| Elegy | PoetsAI | A Neural Networks framework based on Jax and inspired by Keras. | [Elegy](https://github.com/poets-ai/elegy) |
| Mesh Transformer JAX | kingoflolz | The framework which was used to build the recent GPT-J-6B language model. | [Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax) |
| Swarm JAX | kingoflolz | Swarm training framework using Haiku + JAX + Ray for layer parallel transformer language models on unreliable, heterogeneous nodes. | [Swarm JAX](https://github.com/kingoflolz/swarm-jax) |
| RLax | Deepmind | Building blocks for implementing reinforcement learning agents. | [RLax](https://github.com/deepmind/rlax) |
| Coax | Microsoft | A modular Reinforcement Learning (RL) python package for solving OpenAI Gym environments with JAX-based function approximations. | [Coax](https://github.com/microsoft/coax) |
| Optax | Deepmind | A gradient processing and optimization library for JAX. | [Optax](https://github.com/deepmind/optax) |
| Chex | Deepmind | A library of utilities for helping to write reliable JAX code. | [Chex](https://github.com/deepmind/chex) |
| Jraph | Deepmind | A lightweight library for working with graph neural networks in JAX. | [Jraph](https://github.com/deepmind/jraph) |
| JAX, M.D. | Google | A Framework for Differentiable Physics. | [JAX, M.D.](https://github.com/google/jax-md) |
| Oryx | Google? | A library for probabilistic programming and deep learning built on top of Jax. | [Oryx](https://github.com/tensorflow/probability/tree/master/spinoffs/oryx) |




## Contributing

Guidelines for contributing to the project, including instructions for setting up a development environment and submitting pull requests.

## License

Information about the project's license and any relevant disclaimers.
