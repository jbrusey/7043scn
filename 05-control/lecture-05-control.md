---
marp: true
theme: prediction
paginate: true
math: mathjax
title: "Lecture 05: Model-free control"
author: James Brusey
---

<!-- _class: title -->
<!-- _paginate: false -->

# Model-free control

## SARSA and Q-learning

7043SCN Reinforcement Learning

---

# Today’s question

> How can an agent improve its decisions from experience, when it does not know the environment’s model?

By the end, you should be able to:

- explain how action values support model-free control
- distinguish SARSA’s target from Q-learning’s target
- trace one update by hand and use it to check an implementation

---

# From prediction to control

Last lecture: estimate how good states are under a policy.

This lecture: learn **action values** and use them to improve the policy.

$$
Q(s,a)\approx q_\pi(s,a)
$$

The value $Q(s,a)$ estimates the return after taking action $a$ in state $s$, then following policy $\pi$.

---

# Why action values?

With state values alone, choosing the best action needs a model:

$$
\pi(s)=\arg\max_a\;\mathbb{E}[R_{t+1}+\gamma V(S_{t+1})\mid s,a]
$$

With action values, choose directly:

$$
\pi(s)=\arg\max_a Q(s,a)
$$

No transition probabilities are needed.

---

# The control loop

<div class="callout">

**Evaluate:** update $Q$ from observed transitions  
**Improve:** favour actions with larger estimated $Q(s,a)$

</div>

Repeat while interacting with the environment. This is generalised policy iteration using sampled experience.

---

# Exploration and choice

A greedy policy always chooses the current best estimate. Early estimates can be wrong, so keep exploring.

An $\varepsilon$-greedy policy:

- with probability $1-\varepsilon$, choose a highest-valued action
- with probability $\varepsilon$, choose a random action

The policy improves as estimates improve, while exploration gives uncertain actions a chance to be tried.

---

<!-- _class: section -->
<!-- _paginate: false -->

# SARSA

Learn the value of the actions the current policy actually takes

---

# SARSA: the next action matters

From $(S_t,A_t)$, observe reward $R_{t+1}$ and next state $S_{t+1}$. Then choose $A_{t+1}$ using the current policy.

$$
\delta_t=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)
$$

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\delta_t
$$

The sequence **State, Action, Reward, State, Action** gives SARSA its name.

---

# A SARSA update check

Suppose $R_{t+1}=1$, $\gamma=0.9$, $Q(S_t,A_t)=2$, and the next action selected is worth $Q(S_{t+1},A_{t+1})=3$.

With $\alpha=0.1$:

$$
\text{target}=1+0.9(3)=3.7,\qquad \delta_t=3.7-2=1.7
$$

$$
Q(S_t,A_t)\leftarrow2+0.1(1.7)=\boxed{2.17}
$$

The estimate rises because the target exceeds its old value.

---

# SARSA at a terminal state

There is no next action after termination. The terminal bootstrap value is zero:

$$
\text{target}=R_{t+1}
$$

If the terminal reward is $-1$ and $Q(S_t,A_t)=0.4$, with $\alpha=0.2$:

$$
Q(S_t,A_t)\leftarrow0.4+0.2(-1-0.4)=\boxed{0.12}
$$

Do not bootstrap from a stale value stored for a terminal state.

---

<!-- _class: section -->
<!-- _paginate: false -->

# Q-learning

Learn towards the best estimated next action, even while exploring

---

# Q-learning: the greedy target

Q-learning uses the best next action value in its target:

$$
\delta_t=R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)
$$

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\delta_t
$$

The action actually taken at $S_{t+1}$ does not appear in the target.

---

# A Q-learning update check

Use the same transition values as before, but suppose the next action selected during exploration is worth $1$. The best estimated next action is worth $3$.

Q-learning still uses the maximum:

$$
\text{target}=1+0.9(3)=3.7
$$

$$
Q(S_t,A_t)\leftarrow2+0.1(3.7-2)=\boxed{2.17}
$$

If this were SARSA, the chosen next action would give target $1+0.9(1)=1.9$ and update $Q$ to $1.99$.

---

# On-policy and off-policy

<div class="callout">

**SARSA — on-policy**  
Learns about the policy that selects the next action, including its exploration.

**Q-learning — off-policy**  
Learns towards a greedy target while behaviour may still explore.

</div>

Both can use an $\varepsilon$-greedy behaviour policy. Their difference is the target used for learning.

---

# SARSA or Q-learning?

| | SARSA | Q-learning |
|---|---|---|
| Next value in target | $Q(S',A')$ | $\max_{a'}Q(S',a')$ |
| Target follows | Chosen next action | Greedy next action |
| Learns about | Behaviour policy | Greedy policy |
| Exploration affects target? | Yes | No |

The update structure is the same; the target encodes a different control objective.

---

# A compact algorithm sketch

Initialise $Q(s,a)$; repeat for each episode:

1. Start in state $S$ and choose $A$ using an exploratory policy.
2. Take $A$; observe reward $R$ and next state $S'$.
3. If $S'$ is not terminal, choose $A'$ for SARSA and form the target. Q-learning uses $\max_{a'}Q(S',a')$ instead.
4. Update $Q(S,A)$; advance to $(S',A')$ for SARSA, or choose the next behaviour action for Q-learning.
5. Continue until terminal; keep action choices exploratory while improving their preferences from $Q$.

At termination, both methods use reward only as the target. Q-learning does not need $A'$ in its target.

---

# Checks worth automating

**Target**  
Does SARSA use the chosen next action, while Q-learning uses the maximum?

**Direction**  
If the target is above $Q(S,A)$, does the update increase it?

**Step size**  
Is the change exactly $\alpha(\text{target}-Q)$?

**Terminal**  
Does the target use reward only when the episode ends?

**Exploration**  
Can a non-greedy action still be selected by the behaviour policy?

---

# The key idea

- Action values let us improve a policy without an explicit model.
- SARSA learns from the action the behaviour policy takes next.
- Q-learning learns towards the best estimated next action.
- A hand-worked update is a small, useful test of the algorithm’s target and arithmetic.

Next: how can we represent and learn values when the state space is too large for a table?

---

# Reading

Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed.

- Chapter 6: Temporal-Difference Learning
- Sections 6.4–6.5: SARSA and Q-learning

