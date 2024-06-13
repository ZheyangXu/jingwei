# jingwei

> 又北二百里，曰发鸠之山，其上多柘木，有鸟焉，其状如乌，文首，白喙，赤足，名曰：“精卫”，其鸣自詨。是炎帝之少女，名曰女娃。女娃游于东海，溺而不返，故为精卫，常衔西山之木石，以堙于东海。漳水出焉，东流注于河。
> 《山海经·北山经》

## 代码结构

```mermaid
classDiagram
    Agent <|.. QLearningAgent
    Agent <|.. ActorCriticAgent

    Transition o-- Transitions

    Actor <.. QLearningAgent
    Distribution <.. QLearningAgent

    Actor <.. ActorCriticAgent
    Distribution <.. ActorCriticAgent
    Critic <.. ActorCriticAgent

    Agent <.. Trainer
    Rollout <.. Trainer

    DataWrapper <.. Rollout
    ReplayBuffer <.. Rollout

    Transitions <.. ReplayBuffer

    QLearningAgent <|-- DQN
    QLearningAgent <|-- DoubleDQN
    QLearningAgent <|-- DuelingDQN
    
    ActorCriticAgent <|-- PolicyGradientAgent
    ActorCriticAgent <|-- ReinforceAgent
    ActorCriticAgent <|-- PpoAgent
    ActorCriticAgent <|-- A2cAgent
    ActorCriticAgent <|-- A3cAgent


class Trainer {
    + agent: Agent
    + env: Env
    + rollout: Rollout
    + train(): Any
}

class Actor {
    + model: nn.Module
    + optimizer: optim.Optimizer
    + get_action(observation: Observation): Action
    + get_probs(transitions: Transitions): Tensor
    + get_log_probs(transitions: Transitions): Tensor
    + update_step(loss: Tensor): None
}

class Critic {
    + model: nn.Module
    + optimizer: optim.Optimizer
    + estimate_return(transitions: Transitions): Value
    + update_step(loss: Tensor): None
}

class Agent {
    <<interface>>
    + get_action(observation: Observation): Action
    + estimate_return(transitions: Transitions): Value
    + update_step(transitions: Transitions): None
}

class QLearningAgent {
    + actor: Actor
    + distribution: Distribution
    + actor_loss: Callable
    + get_action(observation: Observation): Action 
    + update_step(transitions: Transitions): None
}

class ActorCriticAgent {
    + actor: Actor
    + critic: Critic
    + distribution: Distribution
    + actor_loss_fn: Callable
    + critic_loss_fn: Callable
    + get_action(observation: Observation): Action
    + estimate_return(transitions: Transitions): Value
}

class Rollout {
    + agent: Agent
    + env: Env
    + replay_buffer: ReplayBuffer
}

class ReplayBuffer {
    <<interface>>
    + add(transitions: Transitions): int
    + sample(batch_size: int): Transitions
    + clear(): int
}

class Transition {
    + state: np.ndarray
    + reward: float
    + action: np.ndarray
    + next_state: np.ndarray
    + terminated: bool
    + truncated: bool
    + done: bool
}

class Transitions {
    + state: Tensor
    + reward: Tensor
    + action: Tensor
    + next_state: Tensor
    + terminated: Tensor
    + truncated: Tensor
    + done: Tensor
}

class DataWrapper {
    + wrap(transitions: Transitions): Transitions
}

class PPOAgent

class DQN
```

## 算法

1. [DQN](https://github.com/ZheyangXu/playground/blob/main/src/playground/dqn.py)
2. [REINFORCE](https://github.com/ZheyangXu/playground/blob/main/src/playground/reinforce.py)
3. [A2C](https://github.com/ZheyangXu/playground/blob/main/src/playground/a2c.py)
4. [TRPO](https://github.com/ZheyangXu/playground/blob/main/src/playground/trpo.py)
5. [PPO](https://github.com/ZheyangXu/playground/blob/main/src/playground/ppo.py)
6. [SAC](https://github.com/ZheyangXu/playground/blob/main/src/playground/sac.py)
7. coming soon ...

## 强化学习组成

1. generate samples
2. policy evaluation
3. policy iteration

|算法|General Samples|Fit a model to estimate return|Improve the Policy|
|---|---|---|---|
|Policy Gradient|run the policy|$\hat{Q}^\pi(x_t, u_t) = \sum\limits_{t^\prime=t}^Tr(x_{t^\prime}, u_{t^\prime})$|$\theta \leftarrow \theta + \alpha\nabla_\theta j(\theta)$|
|Actor-Critic|run the policy|fit $\hat{V}_\phi^\pi$|$\theta \leftarrow \theta + \alpha\nabla_\theta j(\theta)$|
|Q-Learning|run the policy|$Q_\phi(s, a) \leftarrow r(s, a)+ \gamma\max_{a^\prime}Q_\phi(s^\prime, a^\prime)$|$a = argmax_a Q_\phi(s, a)$|

|Component|Policy Gradient|Actor-Critic|DQN|
|---|---|---|---|
|General Sample|
|estimate return|
|improve policy|
|advantage|
|on-policy|
|off-policy|
|target network|
|replay buffer|

## LICENSE

MIT
