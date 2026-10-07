---
marp: true
theme: prediction
paginate: true
math: mathjax
title: "Lecture 04: Prediction from experience"
author: James Brusey
---

<!-- _class: title -->
<!-- _paginate: false -->

# Prediction from experience

## Monte Carlo, TD and TD(λ)

7043SCN Reinforcement Learning

---

# Today’s question

> If we can observe experience but do not know the model, how can we estimate what a state is worth?

By the end, you should be able to:

- explain the targets used by Monte Carlo and TD methods
- check an update using a short trajectory
- describe how exploring starts provides coverage
- use λ to connect one-step TD with Monte Carlo

---

# Prediction setup

We follow a fixed policy $\pi$ and observe:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots
$$

We want to estimate:

$$
v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]
$$

The transition probabilities and expected rewards are unknown.

---

# Running example

<div class="walk">
  <span class="terminal">0</span><span>A</span><span>B</span><span class="current">C</span><span>D</span><span>E</span><span class="terminal">1</span>
</div>

- Start in **C**
- Move left or right with equal probability
- Reward is zero except on entering the right terminal
- $\gamma=1$

Question: what is the probability of reaching the right terminal from each state?

---

# Return

The return collects future rewards:

$$
G_t = R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
$$

For an episode that terminates at time $T$:

$$
G_t = \sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}
$$

This gives us a measurable target for value.

---

<!-- _class: section -->
<!-- _paginate: false -->

# Monte Carlo prediction

Wait for the outcome, then learn from the observed return

---

# Monte Carlo target

For each visit to $S_t$, move the estimate towards the return:

$$
V(S_t) \leftarrow V(S_t)+\alpha\big[G_t-V(S_t)\big]
$$

<div class="callout">

**Target:** $G_t$  
**Error:** observed return minus current estimate

</div>

No model and no estimate of a later state are required.

---

# A Monte Carlo check

Suppose:

$$
C\rightarrow D\rightarrow E\rightarrow \text{right terminal}
$$

with rewards $0,0,1$, $\gamma=1$, $V(C)=0.40$, and $\alpha=0.10$.

$$
G_0=1
$$

$$
V(C)\leftarrow 0.40+0.10(1-0.40)=\boxed{0.46}
$$

The successful episode must increase the estimate.

---

# Sample averages

If state $s$ has produced returns $G_1,\ldots,G_n$:

$$
V_n(s)=\frac{1}{n}\sum_{i=1}^{n}G_i
$$

The same mean can be updated without storing every return:

$$
V_n(s)=V_{n-1}(s)+\frac{1}{n}\big[G_n-V_{n-1}(s)\big]
$$

Here the step size is $\alpha_n=1/n$.

---

# First visit or every visit?

**First-visit MC**

Uses the return following the first occurrence of a state in each episode.

**Every-visit MC**

Uses the return following every occurrence of the state.

Both converge under standard assumptions. Every-visit MC often makes simpler code.

---

# Monte Carlo exploring starts

Start each episode from a randomly selected state-action pair:

$$
\Pr(S_0=s,A_0=a)>0 \quad \text{for every }(s,a)
$$

This ensures that all actions can be evaluated, including actions the current policy would not choose.

<div class="callout">

Exploring starts is a **coverage assumption**. It is especially useful when moving from state prediction to action-value estimation and control.

</div>

---

# Exploring starts in an episode

1. Sample a starting pair $(S_0,A_0)$
2. Take $A_0$, then follow the current policy
3. Observe the complete episode
4. Update $Q(S_t,A_t)$ from each return
5. For control, make the policy greedy with respect to $Q$

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[G_t-Q(S_t,A_t)]
$$

---

# What exploring starts assumes

Exploring starts works neatly in simulators, games, and resettable experiments.

It may be impossible or unsafe in a physical system:

- we may not control the initial state
- forcing an arbitrary first action may be unacceptable
- some state-action pairs may be unreachable

Later, stochastic policies and off-policy learning provide more practical coverage mechanisms.

---

<!-- _class: section -->
<!-- _paginate: false -->

# Temporal-difference prediction

Learn from the next step instead of waiting for the episode to end

---

# TD(0) target

TD(0) uses one reward and the next value estimate:

$$
V(S_t)\leftarrow V(S_t)+\alpha\delta_t
$$

$$
\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)
$$

<div class="callout">

**Target:** $R_{t+1}+\gamma V(S_{t+1})$  
The update bootstraps from an existing estimate.

</div>

---

# A TD(0) check

Suppose a transition gives $R_{t+1}=0$, with:

$$
V(C)=0.40,\quad V(D)=0.70,\quad \gamma=1,\quad \alpha=0.10
$$

Then:

$$
\delta_t=0+0.70-0.40=0.30
$$

$$
V(C)\leftarrow0.40+0.10(0.30)=\boxed{0.43}
$$

Moving to a more valuable state must increase $V(C)$.

---

# Monte Carlo and TD(0)

| | Monte Carlo | TD(0) |
|---|---|---|
| Target | Complete return | Reward plus next estimate |
| Update time | End of episode | Every step |
| Bootstrap | No | Yes |
| Tasks | Episodic | Episodic or continuing |
| Typical target | Higher variance | Lower variance, some bias |

Both learn directly from sampled experience without a model.

---

# The n-step bridge

One-step TD looks ahead once. Monte Carlo looks to the end.

An n-step return lies between them:

$$
G_t^{(n)}=R_{t+1}+\gamma R_{t+2}+\cdots+
\gamma^{n-1}R_{t+n}+\gamma^nV(S_{t+n})
$$

- small $n$: more bootstrapping
- large $n$: more sampled reward

---

<!-- _class: section -->
<!-- _paginate: false -->

# TD(λ)

Blend short and long backups, then assign credit backwards

---

# The λ-return

The forward view forms a weighted average of n-step returns:

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}
\lambda^{n-1}G_t^{(n)}
$$

The weights decay geometrically:

$$
(1-\lambda),\quad (1-\lambda)\lambda,\quad
(1-\lambda)\lambda^2,\ldots
$$

The weights sum to one.

---

# A λ weighting check

Let $\lambda=0.5$. The first four weights are:

$$
0.5,\quad 0.25,\quad 0.125,\quad 0.0625
$$

Their sum is $0.9375$. The remaining tail has weight $0.0625$.

$$
0.9375+0.0625=\boxed{1}
$$

If the weights do not sum to one, the implementation has lost or duplicated credit.

---

# Eligibility traces

The backward view keeps a short-term memory for every state:

$$
E_t(s)=\gamma\lambda E_{t-1}(s)+\mathbb{1}(S_t=s)
$$

Every TD error updates recently visited states:

$$
V(s)\leftarrow V(s)+\alpha\delta_tE_t(s)
$$

Recent states receive more credit. Older traces decay by $\gamma\lambda$ each step.

---

# A trace check

Let $\gamma=1$, $\lambda=0.8$, and visit $A$, then $B$, then $C$.

Immediately after visiting $C$:

$$
E(C)=1,\qquad E(B)=0.8,\qquad E(A)=0.8^2=0.64
$$

If the current TD error is $\delta=0.5$ and $\alpha=0.1$:

$$
\Delta V(A)=0.032,\quad \Delta V(B)=0.040,\quad \Delta V(C)=0.050
$$

Credit should weaken with temporal distance.

---

# TD(λ) loop

At the start of each episode, set all traces to zero.

For each transition $S_t,R_{t+1},S_{t+1}$:

$$
\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)
$$

$$
E(s)\leftarrow\gamma\lambda E(s)
$$

$$
E(S_t)\leftarrow E(S_t)+1
$$

$$
V(s)\leftarrow V(s)+\alpha\delta_tE(s)\quad\text{for all }s
$$

---

# What λ changes

<div class="lambda-line">
  <span><strong>λ = 0</strong><br>TD(0)<br>one-step target</span>
  <span><strong>0 &lt; λ &lt; 1</strong><br>mixed horizons<br>decaying traces</span>
  <span><strong>λ near 1</strong><br>longer credit<br>MC-like target</span>
</div>

Larger λ can reduce bootstrapping bias but usually increases variance and keeps credit alive for longer.

---

# Checks worth automating

**Direction**  
Does a better-than-expected outcome increase the estimate?

**Terminal state**  
Does the bootstrap value become zero after termination?

**Bounds**  
For the random walk, do estimates remain near $[0,1]$?

**Trace decay**  
Without a revisit, does each trace shrink by exactly $\gamma\lambda$?

**Endpoints**  
Does λ = 0 reproduce TD(0), and does large λ behave more like MC?

---

# One family of prediction methods

<div class="spectrum">
  <span>TD(0)</span><span>n-step TD</span><span>TD(λ)</span><span>Monte Carlo</span>
</div>

- Monte Carlo learns from complete observed returns
- TD(0) learns sooner by bootstrapping
- TD(λ) spreads each TD error across recent states
- Exploring starts provides broad state-action coverage when resets allow it

The equations give us useful invariants for checking the code.

---

# Reading and next steps

Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed.

- Chapter 5: Monte Carlo methods
- Chapter 6: Temporal-difference learning
- Chapter 7: n-step bootstrapping
- Chapter 12: eligibility traces

Next lecture: prediction becomes control by learning action values and improving the policy.

