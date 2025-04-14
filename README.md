# jingwei

> 又北二百里，曰发鸠之山，其上多柘木，有鸟焉，其状如乌，文首，白喙，赤足，名曰：“精卫”，其鸣自詨。是炎帝之少女，名曰女娃。女娃游于东海，溺而不返，故为精卫，常衔西山之木石，以堙于东海。漳水出焉，东流注于河。
> 《山海经·北山经》

## 代码结构

![UML](./docs/public/jingwei-uml.png)

## 算法

1. Model free RL:
   1. DQN
   2. Double DQN
   3. A2C
   4. TRPO
   5. PPO
   6. Rainbow
   7. SAC
   8. TD3
2. Model based RL:
   1. MBPO
   2. DDPPO
3. Multi-Agent RL:
   1. MAPPO
   2. HAPPO
   3. QMIX
   4. ACE
4. Offline RL:
   1. BCQ
   2. CQL
5. Imitation Learning:
   1. GAIL
   2. BC
6. Exploration:
   1. HER
   2. RND
   3. ICM
   4. NGU

## 强化学习组成

1. generate samples
2. policy evaluation
3. policy iteration

|Algo|General Samples|Fit a model to estimate return|Improve the Policy|
|---|---|---|---|
|Policy Gradient|run the policy|$\hat{Q}^\pi(x_t, u_t) = \sum\limits_{t^\prime=t}^Tr(x_{t^\prime}, u_{t^\prime})$|$\theta \leftarrow \theta + \alpha\nabla_\theta j(\theta)$|
|Actor-Critic|run the policy|fit $\hat{V}_\phi^\pi$|$\theta \leftarrow \theta + \alpha\nabla_\theta j(\theta)$|
|Q-Learning|run the policy|$Q_\phi(s, a) \leftarrow r(s, a)+ \gamma\max_{a^\prime}Q_\phi(s^\prime, a^\prime)$|$a = argmax_a Q_\phi(s, a)$|

## LICENSE

MIT
