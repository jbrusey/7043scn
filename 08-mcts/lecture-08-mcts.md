---
marp: true
theme: prediction
paginate: true
math: mathjax
title: "Lecture 08: Monte Carlo tree search"
author: James Brusey
---

<!-- _class: title -->
<!-- _paginate: false -->

# Monte Carlo tree search

## Search guided by experience and value estimates

7043SCN Reinforcement Learning

---

# Today’s question

> How can an agent use a limited number of simulations to choose a strong action?

By the end, you should be able to:

- describe the MCTS loop and its action-value estimates
- explain how a value network guides search
- check a tree backup and understand what it measures
- explain why returning to saved states matters for exploration

---

# From model to planning

In model-based RL, we learn or use a model of what actions do:

$$
(s,a)\longrightarrow (r,s')
$$

Planning asks: **which action looks best if we simulate what might follow?**

- A full state-value table may be too large.
- Forward search focuses computation on states reachable from here.
- A simulator can be enough; we need not write down every transition probability.

---

# Forward search starts here

Start a search tree at the current state $s_t$.

```text
                         s_t
                    /     |     \
                  a₁      a₂      a₃
                 /        |        \
               ...       ...       ...
```

Each simulation follows possible actions and outcomes from the root. The tree grows where search spends effort.

---

# Why a tree?

Suppose there are $b$ actions at each step and we search $d$ steps.

$$
\text{all action sequences} \approx b^d
$$

Even a modest branching factor grows quickly. MCTS samples promising branches instead of expanding every sequence.

It is **selective** and **anytime**: more simulations can improve the decision, and we can stop when the budget runs out.

---

<!-- _class: section -->
<!-- _paginate: false -->

# Monte Carlo tree search

Grow a useful tree with repeated simulations

---

# One simulation: four steps

1. **Select** actions through the existing tree.
2. **Expand** a new child when search reaches the frontier.
3. **Evaluate** the new position, by rollout or value estimate.
4. **Back up** the result along the visited path.

Repeat from the current root until the simulation budget is used.

---

# Selection balances value and visits

An action value is the mean backed-up return:

$$
Q(s,a)=\frac{W(s,a)}{N(s,a)}
$$

where $N$ is the visit count and $W$ is the sum of returns.

One classic selection rule is UCB:

$$
Q(s,a)+c\sqrt{\frac{\ln N(s)}{N(s,a)}}
$$

The first term favours high returns; the second gives less-visited actions a chance.

---

# A selection check

At a node, suppose $N(s)=16$ and $c=1$.

| Action | $Q(s,a)$ | $N(s,a)$ | UCB score |
|---|---:|---:|---:|
| Left | 0.60 | 8 | $0.60+\sqrt{\ln(16)/8}\approx0.97$ |
| Right | 0.50 | 2 | $0.50+\sqrt{\ln(16)/2}\approx1.33$ |

Right is explored next despite its lower mean: it has fewer visits.

**Check:** increasing $N(s,a)$ must reduce its exploration bonus.

---

# Back up the result

For a sampled path, update every visited edge:

$$
N(s,a)\leftarrow N(s,a)+1,\qquad W(s,a)\leftarrow W(s,a)+G
$$

Then $Q(s,a)=W(s,a)/N(s,a)$.

Example: a win (+1) and a loss (-1) give $W=0$, $N=2$, and $Q=0$.

**Check:** the mean must stay between the smallest and largest observed return.

---

# Which action do we play?

Search estimates values for root actions. A simple choice is:

$$
a_t=\arg\max_a N(s_t,a)
$$

Choosing the most visited action uses the search budget as evidence. Choosing the largest estimated $Q$ is another option.

During self-play training, the visit counts can also define a softened action distribution, encouraging varied play.

---

# What does the search assume?

- We can simulate a transition from the state and action.
- Search can evaluate or continue from the resulting state.
- The state contains enough information to make future transitions meaningful.

In a board game, a state is the board plus whose turn it is and any relevant rule information.

For an unfamiliar or learned environment, model errors can make a confidently searched plan wrong.

---

<!-- _class: section -->
<!-- _paginate: false -->

# AlphaZero

Combine tree search with a neural network

---

# Search plus a learned value

At a new position, a neural network can provide two useful estimates:

- **Policy prior** $P(s,a)$: which actions look promising to try first?
- **Value** $v(s)$: how good does this position look from here?

The tree search uses the prior to guide exploration and the value to evaluate a leaf without simulating all the way to the end.

$$
\text{network intuition} + \text{search over consequences}
$$

---

# PUCT: a policy-guided selection rule

A commonly used form is:

$$
a^*=\arg\max_a\left[Q(s,a)+c_{\rm puct}P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}\right]
$$

The prior steers early visits. As an action is visited, its exploration bonus falls and the backed-up value matters more.

**Check:** if visit counts are equal, a larger prior should give an action a larger exploration bonus.

---

# The network does not replace search

The value network gives a quick estimate at a leaf.

Search then tests consequences along selected paths and combines many estimates into action values.

```text
policy prior ──► where to search
value estimate ─► how a leaf looks
tree backups ───► what the search found for each root action
```

Search can correct a weak first impression; it can also amplify errors if the model or value estimates are poor.

---

# AlphaZero’s learning loop

1. Use the current policy-value network to guide MCTS.
2. Play a self-play game using the search’s move distribution.
3. Save positions, search visit distributions, and game outcomes.
4. Train the network to predict the search distribution and final outcome.
5. Repeat with the updated network.

Search improves move choice; self-play outcomes train the estimates that guide later search.

---

# A value backup check

In a two-player, zero-sum game, keep the value perspective consistent.

If a position is worth $+0.7$ for the player to move, then after a move to the opponent’s turn its value is $-0.7$ from the opponent’s perspective.

$$
v_{\rm parent}\approx -v_{\rm child}
$$

**Check:** a winning outcome for one player must be a losing outcome for the other. Sign mistakes can make search prefer blunders.

---

# What MCTS adds to the policy

The network supplies a general starting point learned across many positions.

The tree supplies local computation focused on the current decision.

| Network | Search |
|---|---|
| Fast generalisation | Deliberation for this position |
| Policy prior and value | Visit counts and backed-up action values |
| Trained from self-play | Uses a simulation budget at play time |

The combination is useful when we can simulate the rules but cannot search every line.

---

<!-- _class: section -->
<!-- _paginate: false -->

# Reaching states again

Search and exploration need useful starting points

---

# The reset problem

An interesting state may be hard to reach again by random exploration.

If the agent forgets how it got there, it may spend most of its budget repeating unproductive paths.

In a simulator, saving a state can make it possible to restart there and explore what follows.

This is a capability of the environment or simulator, not a property guaranteed by MCTS.

---

# Go-Explore: return, then explore

Go-Explore addresses hard-exploration tasks by remembering promising states:

1. Save an interesting state (or a way to reach it).
2. Return to that state.
3. Explore from there to find something new.

This separates **getting back** from **trying new actions**. It is especially natural when a simulator can restore saved states.

---

# A state must be restartable

To resume a simulation faithfully, save enough information to reconstruct the future:

- environment state, not only the visible screen
- whose turn it is and any counters or rule flags
- randomness state when reproducibility matters

**Check:** restore a saved state, take the same action, and verify the same transition when the environment is deterministic.

---

# What to remember

- MCTS spends simulations selectively from the current state.
- Visit counts and backed-up returns give checks on the tree’s behaviour.
- AlphaZero combines tree search with a policy-value network: search improves decisions; self-play trains the network.
- Saved states let exploration return to promising points and continue from there.

Equations are useful here: they make signs, averages, and exploration bonuses testable.

---

# Reading

- David Silver, *Lecture 8: Integrating Learning and Planning* (source lecture PDF in this folder)
- Silver et al., [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://arxiv.org/abs/1712.01815)
- Ecoffet et al., [Go-Explore: a New Approach for Hard-Exploration Problems](https://arxiv.org/abs/1901.10995)
