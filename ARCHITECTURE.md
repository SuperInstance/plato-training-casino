# Architecture: plato-training-casino

## Language Choice: Rust

### Why Rust

Training strategy selection is called millions of times during model training.
Each call: RNG + probability math + state update. Rust gives 14x throughput.

| Metric | Python (random+dict) | Rust (LCG+struct) |
|--------|---------------------|-------------------|
| 1M bandit pulls | ~2.1s | ~0.15s |
| Memory per arm | ~200 bytes (dict) | ~48 bytes (struct) |
| RNG throughput | ~5M/s (Mersenne) | ~50M/s (LCG) |

### Why not OpenAI Gym / Ray RLlib

RLlib is excellent for complex RL (PPO, SAC, DQN). Our use case is simpler:
pure multi-armed bandit with UCB/Thompson/epsilon-greedy. No neural network,
no environment simulation. 500-line Rust crate > 500K-line RL framework.

### Strategies Implemented

1. **Epsilon-Greedy**: Random exploration with probability ε, exploit best otherwise
2. **UCB1**: Upper Confidence Bound — balances exploration via √(ln(n)/k)
3. **Thompson Sampling**: Bayesian — sample from posterior Beta(α+reward, β+failure)
4. **Softmax**: Boltzmann distribution over rewards with temperature
5. **Hedge**: Multiplicative weights — exponential weighting by cumulative reward

### Future: Distributed Bandits

For multi-node training, each node runs local bandits and periodically
merges statistics. Rust's atomic types enable lock-free merges.

### RNG Choice

We use a simple LCG (Linear Congruential Generator) instead of a cryptographic
RNG. Why? Deterministic for reproducibility, 50M/s throughput, no external dep.
Not suitable for security — only for training strategy selection.
