# 值迭代

## 简介

值迭代是一种动态规划方法，通过迭代求解贝尔曼方程来寻找最优值函数 $V^{\star}$。它利用动态规划的概念维护一个值函数 $V$，该函数近似最优值函数 $V^{\star}$，并通过迭代不断改进 $V$，直至其收敛到 $V^{\star}$（或接近 $V^{\star}$）。

## 算法

理解贝尔曼方程后，值迭代算法其实相当直观：我们只需使用贝尔曼方程反复计算 $V$ 值，直到结果收敛或达到预设的迭代次数即可。

$$\begin{array}{l} Input:\ \text{MDP}\ M = \langle S, s_0, A, P_a(s' \mid s), r(s, a, s')\rangle\\ Output:\ \text{Value function}\ V\\[2mm] \text{Set}\ V\ \text{to arbitrary value function; e.g., }\ V(s) = 0\ \text{for all}\ s\\[2mm] Repeat\ \\ \quad\quad \Delta \leftarrow 0 \\ \quad\quad foreach\ s \in S \\ \quad\quad\quad\quad \underbrace{V'(s) \leftarrow \max_{a \in A(s)} \sum_{s' \in S} P_a(s' \mid s)\ [r(s, a, s') + \gamma\ V(s') ]}_{\text{Bellman equation}} \\ \quad\quad\quad\quad \Delta \leftarrow \max(\Delta, |V'(s) - V(s)|) \\ \quad\quad V \leftarrow V' \\ Until\ \Delta \leq \theta \end{array}$$

值迭代实际上就是反复应用贝尔曼方程，直到值函数 $$V$$ 不再变化，或者变化量小于某个很小的阈值（$$\theta$$）。

我们也可以使用 Q 值的概念来重写算法，这种形式更接近代码实现。循环部分可表示为：

$$\begin{array}{l} \quad\quad \Delta \leftarrow 0 \\ \quad\quad \text{对每个} s \in S \\ \quad\quad\quad\quad \text{对每个} a \in A(s) \\ \quad\quad\quad\quad\quad\quad Q(s, a) \leftarrow \sum_{s' \in S} P_a(s' \mid s)\ [r(s, a, s') + \gamma\ V(s') ] \\ \quad\quad\quad\quad \Delta \leftarrow \max(\Delta, |\max_{a \in A(s)} Q(s, a) - V(s)|) \\ \quad\quad\quad\quad V(s) \leftarrow \max_{a \in A(s)} Q(s, a) \end{array}$$

随着迭代次数增加，值迭代会收敛到最优策略：当迭代次数 $i \mapsto \infty$ 时，$V \mapsto V^*$。理论上，给定无限次迭代，算法必然能得到最优解。

在实际应用中，值迭代渐近收敛到最优值函数 $V^*$，但算法通常在残差 $\Delta$ 达到预设阈值 $\theta$ 时终止——即当迭代间值函数的最大变化"足够小"时停止。

有了值函数 $V$，我们就可以轻松定义策略：在状态 $s$，选择具有最高期望奖励的动作，这就是策略提取过程。

经过 $k$ 次迭代后终止得到的贪婪策略的损失上界为 $\frac{2 \gamma \delta_{max}}{1-\gamma}$，其中 $\delta_{max}= \max_{s}|V^{*}(s) - V_k(s)|$。

值得注意的是，我们不需要完全精确的最优值函数 $$V$$ 就能获得最优策略。一个"足够接近"最优的值函数仍然可以产生最优策略，因为微小的值函数差异不会改变最终策略选择。当然，除非我们确定值函数是最优的，否则无法确定得到的策略是否最优。

## 策略评估

