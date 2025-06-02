# 策略迭代

## 简介

解决马尔可夫决策过程(MDP)的另一种常见方法是使用策略迭代——这种方法与价值迭代类似。价值迭代是对价值函数进行迭代，而策略迭代则是对策略本身进行迭代，在每次迭代中创建一个严格改进的策略（除非当前迭代的策略已经是最优的）。

策略迭代首先从某个（非最优）策略开始，例如随机策略，然后计算给定该策略下MDP中每个状态的价值——这一步称为策略评估。随后，它通过计算从每个状态可执行的各个动作的期望奖励来更新该状态的策略。

这里的基本思想是，策略评估比价值迭代更容易计算，因为要考虑的动作集是由我们目前拥有的策略固定的。

## 策略评估

策略评估是策略迭代中的一个重要概念，它用于评估策略的期望回报。

策略 $\pi$ 从状态 $s$ 出发的期望回报 $V^\pi(s)$ 是由该策略定义的所有可能状态序列的回报与其在策略 $\pi$ 下的概率的加权平均。

策略评估可以通过以下方程来定义 $V^{\pi}(s)$：

$$V^\pi(s) = \sum_{s' \in S} P_{\pi(s)} (s' \mid s)\ [r(s, a, s') + \gamma\ V^\pi(s') ]$$

其中，对于终止状态，$V^\pi(s)=0$。

需要注意的是，这个方程与贝尔曼方程非常相似，但区别在于 $V^\pi(s)$ 并非最佳动作的价值，而仅仅是策略 $\pi$ 在状态 $s$ 中选择的动作 $\pi(s)$ 的价值。还要注意表达式 $P_{\pi(s)}(s' \mid s)$ 而非 $P_a(s' \mid s)$，这意味着我们只评估策略所定义的那个动作。

一旦理解了策略评估的定义，其实现就相对简单了。它与价值迭代的实现方式相似，只是使用策略评估方程而非贝尔曼方程。

$$
\begin{array}{l} Input:\ \pi\ \text{the policy for evaluation}, V^\pi\ \text{value function, and}\\ \quad\quad\quad\quad \text{MDP}\ M = \langle S, s_0, A, P_a(s' \mid s), r(s, a, s')\rangle\\ Output:\ \text{Value function}\ V^\pi\\[2mm] Repeat\\ \quad\quad \Delta \leftarrow 0\\ \quad\quad foreach\ s \in S\\ \quad\quad\quad\quad \underbrace{V'^{\pi}(s) \leftarrow \sum_{s' \in S} P_{\pi(s)}(s' \mid s)\ [r(s, a, s') + \gamma\ V^\pi(s') ]}_{\text{Policy evaluation equation}}\\ \quad\quad\quad\quad \Delta \leftarrow \max(\Delta, |V'^\pi(s) - V^\pi(s)|)\\ \quad\quad V^\pi \leftarrow V'^\pi\\ until\ \Delta \leq \theta \end{array} 
$$

最优期望奖励是 $V^*(s)$ 是 $\max_{\pi} V^\pi(s)$ 且最优策略是 $\textrm{argmax}_{\pi} V^\pi(s)$.

## 策略提升

如果我们已有一个策略并希望改进它，可以使用策略提升来更新策略（即改变针对各状态推荐的动作）。这一过程基于我们从策略评估中获得的 $V(s)$ 值来更新策略所推荐的动作。

令 $Q^{\pi}(s, a)$ 表示从状态 $s$ 开始，首先执行动作 $a$ 然后遵循策略 $\pi$ 的期望回报。回顾马尔可夫决策过程一章，我们将其定义为：

$$Q^{\pi}(s, a) = \sum_{s' \in S} P_a(s' \mid s)\ [r(s, a, s') \, + \, \gamma\ V^{\pi}(s')]$$

在这里，$V^{\pi}(s')$ 是通过策略评估得到的价值函数。

如果存在某个动作 $a$ 使得 $Q^{\pi}(s, a) > Q^{\pi}(s, \pi(s))$，那么策略 $\pi$ 可以通过设置 $\pi(s) \leftarrow a$ 得到严格改进。这将提升整体策略的效果。

## 策略迭代

结合策略评估和策略提升，我们可以定义策略迭代，它通过执行一系列交替的策略评估和改进来计算最优策略 $\pi$：

$$
\begin{array}{l} Input:\ \text{MDP}\ M = \langle S, s_0, A, P_a(s' \mid s), r(s, a, s')\rangle\\ Output:\ Policy\ \pi\\[2mm] \text{Set}\ V^\pi\ \text{to arbitrary value function; e.g., }\ V^\pi(s)=0\ \text{for all}\ s\\ \text{Set}\ \pi\ \text{to arbitrary policy; e.g.}\ \pi(s) = a\ \text{for all}\ s, \ \text{where}\ a \in A\ \text{is an arbitrary action}\\[2mm] Repeat\\ \quad\quad \text{Compute}\ V^\pi(s)\ \text{for all}\ s\ \text{using policy evaluation}\\ \quad\quad foreach\ s \in S\\ \quad\quad\quad\quad \pi(s) \leftarrow \textrm{argmax}_{a \in A(s)}Q^{\pi}(s, a)\\ until\ \pi\ \text{does not change} \end{array}
$$

策略迭代算法在有限次迭代后会得到最优策略 $\pi$，这是因为策略的数量是有限的，上限为 $O(|A|^{|S|})$，这与价值迭代不同，后者理论上可能需要无限次迭代。

然而，每次迭代的计算复杂度为 $O(|S|^2 |A| + |S|^3)$。经验证据表明，最高效的方法取决于所解决的特定MDP模型，但令人惊讶的是，策略迭代通常只需要很少的迭代次数。

## 总结

1. 策略迭代是一种动态规划技术，它直接计算策略，而不是计算最优 $V(s)$ 然后提取策略；但它也使用了价值的概念。
2. 它能在有限步骤内产生最优策略。
3. 与价值迭代类似，对于中等规模的问题，它表现良好，但随着状态空间的增长，它的扩展性不佳。
