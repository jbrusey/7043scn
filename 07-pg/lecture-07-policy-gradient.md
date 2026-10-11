---
marp: true
theme: policy-gradient
paginate: true
math: mathjax
title: "Lecture 07: Policy gradient"
author: James Brusey
---

<!-- _class: title -->
<!-- _paginate: false -->

# Policy gradient

## Learning the policy directly

7043SCN Reinforcement Learning

---

# Today’s question

> How can experience tell us which way to change a policy?

By the end, you should be able to:

- explain the REINFORCE update and check its direction
- describe how a baseline and a critic reduce variance
- explain why large policy updates can be risky
- compare TRPO’s trust region with PPO’s clipped objective

---

# From values to a policy

Value-based methods learn values, then derive a policy from them.

Policy gradient methods represent the policy directly:

$$
\pi_\theta(a\mid s)=\Pr(A_t=a\mid S_t=s;\theta)
$$

The parameters $\theta$ control action probabilities.

<div class="callout">

The policy can be stochastic by design — useful when randomness is part of good behaviour.

</div>

---

# Why learn a stochastic policy?

In rock–paper–scissors, a predictable player can be exploited.

The uniform policy assigns probability $1/3$ to each action. Randomness is the strategy.

Policy-based methods can represent this directly. A greedy value-based policy tends to become nearly deterministic.

---

# Policy search

Choose parameters that maximise expected performance:

$$
\theta^*=\arg\max_\theta J(\theta)
$$

For episodic tasks, $J(\theta)$ can be the expected return from the start state.

Gradient ascent moves parameters uphill:

$$
\theta\leftarrow\theta+\alpha\nabla_\theta J(\theta)
$$

We do not need to compute the whole objective exactly; sampled experience can estimate its gradient.

---

# The score of an action

For a differentiable policy, define its score:

$$
\nabla_\theta\log\pi_\theta(a\mid s)
$$

It points toward parameter changes that make this action more likely in this state.

For a softmax policy over action preferences $z_\theta(s,a)$:

$$
\pi_\theta(a\mid s)=\frac{e^{z_\theta(s,a)}}{\sum_b e^{z_\theta(s,b)}}
$$

The probabilities remain positive and sum to one.

---

# REINFORCE: learn from the outcome

After an episode, use the return following each chosen action:

$$
\theta\leftarrow\theta+\alpha G_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

- A better-than-expected outcome should reinforce the action.
- A worse outcome should make it less likely.
- The update is applied to the policy that generated the data.

This is Monte Carlo learning: it waits for the return.

---

# Check the direction before coding

Suppose the sampled action has $G_t=2$.

$$
\theta_{\rm new}=\theta+\alpha\,2\,\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

The multiplier is positive, so the update increases the sampled action’s log probability (locally).

If instead $G_t=-1$, the update goes in the opposite direction.

<div class="callout">

Sanity check: positive learning signal reinforces the sampled action; negative signal discourages it.

</div>

---

# Why the raw return is noisy

The return includes reward variation that may have little to do with the action at this step.

Two episodes can take the same action in the same state but have very different returns.

High variance makes the gradient estimate jump around, so learning can be slow.

Can we measure whether the outcome was good relative to what this state usually produces?

---

# A baseline changes the signal, not its mean

Subtract a state-dependent baseline $b(s)$:

$$
\theta\leftarrow\theta+\alpha\big(G_t-b(S_t)\big)\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

Because the baseline does not depend on the sampled action, it does not change the expected policy gradient.

It can reduce variance by centring the learning signal.

---

# Advantage: better or worse than expected?

A natural baseline is the state value:

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

- $A^\pi(s,a)>0$: this action did better than the state’s usual outcome.
- $A^\pi(s,a)<0$: this action did worse than usual.

The actor update becomes:

$$
\Delta\theta=\alpha A_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

---

# A critic supplies the baseline

The **actor** chooses actions; the **critic** estimates how good they are.

<div class="ac-loop">
  <span class="state">State sₜ</span><b class="to-actor">→</b><span class="actor">Actor<br>πθ</span><b class="to-env">→</b><span class="experience">Action aₜ<br>Reward, next state</span>
  <span class="critic">Critic<br>Vw(s)</span><b class="up">↑</b><span class="signal">TD error / advantage</span>
</div>

The critic learns about the current policy. Its estimate gives the actor a lower-variance learning signal.

---

# One-step actor–critic

Use the critic’s TD error as an advantage estimate:

$$
\delta_t=R_{t+1}+\gamma V_w(S_{t+1})-V_w(S_t)
$$

Update the actor in the direction of that signal:

$$
\theta\leftarrow\theta+\alpha\delta_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

The critic updates $w$ to improve its value predictions. Actor and critic learn together, often at different step sizes.

---

# A practical actor–critic loop

1. Sample $A_t\sim\pi_\theta(\cdot\mid S_t)$
2. Observe $R_{t+1},S_{t+1}$
3. Compute $\delta_t=R_{t+1}+\gamma V_w(S_{t+1})-V_w(S_t)$
4. Update critic from $\delta_t$
5. Update actor from $\delta_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)$

The TD error is a useful check: if reward plus next-state value exceeds the current estimate, it is positive.

---

# The step-size problem

A policy gradient points uphill **near the current policy**.

A large parameter step can change action probabilities sharply, so the data gathered under the old policy may no longer describe the new one.

Too small: progress is slow. Too large: performance can collapse.

<div class="callout">

Can we improve the policy while limiting how far it moves?

</div>

---

# TRPO: stay in a trust region

TRPO maximises a surrogate improvement while constraining the policy change:

$$
\max_\theta\; \mathbb E\!\left[\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\rm old}(A_t\mid S_t)}\,\hat A_t\right]
\quad\text{subject to}\quad
\mathbb E\!\left[D_{\rm KL}(\pi_{\rm old}\,\|\,\pi_\theta)\right]\leq\delta
$$

The ratio measures how the new policy reweights an old sampled action. KL divergence limits the average change.

Think: take a useful step, but keep it inside a locally trusted region.

---

# PPO: a simpler safeguard

PPO uses the same probability ratio:

$$
r_t(\theta)=\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\rm old}(A_t\mid S_t)}
$$

Its clipped objective is:

$$
L^{\rm CLIP}(\theta)=\mathbb E\!\left[\min\!\left(r_t\hat A_t,\;\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right]
$$

Clipping limits the incentive to move the probability ratio too far in a direction that improves the sampled estimate.

---

# What clipping does

If $\hat A_t>0$, increasing the action probability helps — until the ratio reaches $1+\epsilon$.

If $\hat A_t<0$, decreasing the action probability helps — until the ratio reaches $1-\epsilon$.

<div class="ratio-band">
  <span>discourage too much<br>1 − ε</span><span class="centre">old policy<br>1</span><span>reinforce too much<br>1 + ε</span>
</div>

PPO clipping is a practical guardrail, not a hard guarantee that every policy change stays within a fixed KL distance.

---

# TRPO and PPO

| | TRPO | PPO |
|---|---|---|
| Policy change | Explicit KL constraint | Clipped probability ratio |
| Optimisation | Constrained, more involved | Standard minibatch gradient updates |
| Main idea | Trust a local improvement | Limit the incentive for a large update |

Both reuse data collected by the old policy for a limited number of updates, and both depend on good advantage estimates.

---

# What to check in an implementation

- **Direction:** positive advantage raises the sampled action’s probability; negative advantage lowers it.
- **Normalisation:** action probabilities sum to one.
- **Old policy:** PPO ratios compare the updated policy with the policy that collected the sample.
- **Bounded change:** monitor KL or clip fraction; a small clipping parameter does not guarantee a small KL.
- **Critic:** value targets and TD errors have the expected sign and scale.

Maths gives us more than an update rule: it gives checks for whether the algorithm is behaving as intended.

---

# Takeaways

- REINFORCE reinforces actions using sampled returns.
- A baseline or critic turns return into a relative learning signal.
- Actor–critic trades some bias for lower variance and faster feedback.
- TRPO constrains policy change; PPO clips the surrogate objective.

The core question remains: **did the update make a good action more likely, without changing the policy too abruptly?**

---

# Further reading

- Sutton and Barto, *Reinforcement Learning: An Introduction*, policy-gradient methods
- Schulman et al., “Trust Region Policy Optimization” (2015)
- Schulman et al., “Proximal Policy Optimization Algorithms” (2017)

