# 02 — Training Methods: Deep Analysis of Every Major Approach

## Overview

Modern LLM training has three distinct phases:
1. **Pretraining** — teach the model language and world knowledge
2. **Supervised Fine-Tuning (SFT)** — teach the model to follow instructions
3. **Alignment** — teach the model to be helpful, harmless, and honest

Each phase has multiple competing methods. Below is a deep analysis of every major one.

---

## Phase 1: Pretraining Methods

### Standard Causal Language Modeling (CLM)
**What it is:** Predict the next token given all previous tokens. The foundation of every GPT-style model.

**How it works:**
```
Input:  "The cat sat on the"
Target: "cat sat on the mat"
Loss:   Cross-entropy on each token
```

**Strengths:**
- Scales perfectly — more data + more parameters = better performance
- No labels needed — all text is self-supervised
- Learns language, facts, and reasoning implicitly

**Weaknesses:**
- Does not teach the model to be *helpful* — just to predict text
- Learns bad patterns alongside good ones
- Does not generalize instruction-following from raw text

**Cost:** Extremely expensive at scale. GPT-2 costs ~$50K to pretrain from scratch. GPT-4 cost estimated $50–100M.

**Orb relevance:** GPT-2 was trained this way. We skip pretraining and start from these weights.

---

### Masked Language Modeling (MLM) — BERT-style
**What it is:** Predict randomly masked tokens from bidirectional context.

**Strengths:** Better for understanding tasks (classification, extraction)
**Weaknesses:** Cannot generate text — not a language model in the generative sense

**Orb relevance:** Not applicable. Orb is a generative model.

---

## Phase 2: Supervised Fine-Tuning (SFT)

### Standard SFT
**What it is:** Train on (instruction, response) pairs using the same CLM loss, restricted to the response tokens.

**Dataset format:**
```
System: You are Orb, a helpful AI assistant.
User: What is the capital of France?
Assistant: The capital of France is Paris.
```

**Loss is computed only on the Assistant tokens.** The model learns to produce the response given the instruction.

**Strengths:**
- Simple, stable, well-understood
- Works with small datasets (1K–100K examples)
- Dramatically improves instruction following

**Weaknesses:**
- Quality entirely dependent on dataset quality
- Model memorizes responses rather than reasoning patterns
- Prone to "sycophancy" — agreeing with users even when wrong

**Key datasets for Orb:**
- `OpenHermes-2.5` — 1M diverse, high-quality instructions (best single dataset)
- `ShareGPT` — real ChatGPT conversations (diverse but noisy)
- `Alpaca-52k` — Stanford's instruction set (dated but clean)
- `WizardLM-Evol-Instruct` — evolved complexity instructions

**Orb relevance:** PRIMARY step. Must do this before any alignment method.

---

### Chat/Multi-Turn SFT
**What it is:** SFT on multi-turn conversations, not just single exchanges.

**Why it matters:** Models trained only on single-turn data fail to maintain coherent, context-aware conversations. They "forget" what was said 3 turns ago.

**Technique:** Pack multiple turns into a single training example. Compute loss only on assistant turns.

**Key datasets:** UltraChat-200k, ShareGPT-cleaned

---

## Phase 3: Alignment Methods

### RLHF — Reinforcement Learning from Human Feedback
**Origin:** OpenAI (InstructGPT, 2022). Used in ChatGPT, GPT-4.

**How it works — 3 stages:**

**Stage 1: SFT** (as above)

**Stage 2: Reward Model Training**
- Collect pairs of model outputs for the same prompt
- Humans rank which output is better
- Train a separate "reward model" to predict human preferences
- The reward model outputs a scalar score for any (prompt, response) pair

**Stage 3: RL Optimization (PPO)**
- Use Proximal Policy Optimization (PPO) to maximize reward model score
- Add a KL-divergence penalty to prevent the model from "gaming" the reward model
- Iterate until the model consistently produces high-reward outputs

**Strengths:**
- Most proven alignment method at scale
- Captures nuanced human preferences
- Dramatically improves helpfulness and safety simultaneously

**Weaknesses:**
- Expensive: requires large human annotation workforce
- Reward hacking: model learns to game the reward model (produces responses that score high but are actually bad)
- Training instability: PPO is notoriously finicky to tune
- Requires 3 separate models in memory simultaneously (huge compute cost)
- Human annotators introduce biases and inconsistencies

**Cost estimate for Orb-scale:**
- Human annotations: ~$5,000–50,000 depending on dataset size
- Compute: ~$500–2,000 (GPU) for the RL training itself
- Engineering complexity: Very high

**Rating:** ⭐⭐⭐⭐ (best alignment quality, worst cost/complexity ratio)

---

### Constitutional AI (CAI) / RLAIF
**Origin:** Anthropic (2022). Used in Claude 1, 2, 3.

**Core insight:** Replace human raters with an AI rater guided by a written "constitution" (a list of principles). This is called **RLAIF** — Reinforcement Learning from AI Feedback.

**The Constitution (Anthropic's actual principles, simplified):**
```
1. Choose the response that is least likely to be harmful
2. Choose the response that is most honest
3. Choose the response that a thoughtful senior Anthropic employee would prefer
4. Choose the response that best supports human autonomy
```

**How it works — 4 stages:**

**Stage 1: SFT** (standard)

**Stage 2: Critique + Revision (supervised)**
- Prompt the model: *"Here is a response. Using principle X from the constitution, identify any issues."*
- Prompt the model: *"Revise the response to address those issues."*
- Fine-tune on the (original, critique, revised) triplets

**Stage 3: AI Preference Data Generation**
- Generate multiple responses to many prompts
- Ask the AI to compare pairs using constitution principles
- Collect AI preferences as preference dataset (no humans needed)

**Stage 4: RL from AI Feedback (same as RLHF Stage 2-3)**
- Train reward model on AI preferences
- Run PPO

**Strengths:**
- Dramatically cheaper than RLHF (no human raters)
- More consistent and principled than human feedback
- Scales with model capability — better model = better AI rater
- Produces genuinely helpful AND safe models
- The constitution is inspectable and adjustable

**Weaknesses:**
- Requires a capable base model to generate good critiques (GPT-2 is marginal)
- AI preferences inherit AI biases
- Still requires PPO (complex RL training)
- Constitution must be carefully crafted

**Orb relevance:** HIGH. This is the basis for the OASIS method. We can implement a simplified version on Orb.

**Rating:** ⭐⭐⭐⭐⭐ (best alignment quality + reasonable cost)

---

### DPO — Direct Preference Optimization
**Origin:** Rafailov et al., Stanford (2023).

**Core insight:** The RLHF optimization problem has a closed-form solution. You don't need a separate reward model or RL training at all. You can directly optimize the language model on preference pairs.

**Mathematical insight:**
The reward model in RLHF is implicitly defined by the optimal policy. DPO derives a training objective that directly optimizes the language model on (preferred, rejected) pairs:

```
Loss = -log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))

Where:
  y_w = preferred (winning) response
  y_l = rejected (losing) response
  π_θ = model being trained
  π_ref = reference model (frozen)
  β = temperature controlling deviation from reference
```

**In plain English:** Push up the probability of preferred responses AND push down the probability of rejected responses, while staying close to the original model's distribution.

**Strengths:**
- No reward model needed (saves memory and compute)
- No RL training — just supervised learning
- Much more stable than PPO
- Nearly as good as RLHF on most benchmarks
- Can be run on a single GPU

**Weaknesses:**
- Requires preference pairs (still needs human or AI preference data)
- Can cause "forgetting" if β is too high
- Does not explicitly model the reward signal — harder to debug
- Slightly worse than RLHF at the extreme high end

**Datasets for DPO:**
- `UltraFeedback` — 64K prompts with 4 responses each, rated by GPT-4
- `HH-RLHF` — Anthropic's human preference dataset (harmless + helpful)
- `OpenHermes-Preferences` — curated preference pairs

**Orb relevance:** HIGH. DPO is the right alignment method for Orb. It's the most practical.

**Rating:** ⭐⭐⭐⭐⭐ (best cost/quality tradeoff for alignment)

---

### ORPO — Odds Ratio Preference Optimization
**Origin:** Hong et al. (2024).

**Core insight:** Combine SFT and alignment into a single training objective. No need for a separate SFT phase.

**Loss function:**
```
Loss = SFT_Loss + λ · OR_Loss

Where OR_Loss penalizes the odds ratio between winning and losing responses
```

**Strengths:**
- Single-stage training (SFT + alignment simultaneously)
- No reference model needed (unlike DPO)
- Memory efficient
- Empirically matches or exceeds DPO on many benchmarks

**Weaknesses:**
- Less studied than DPO
- λ hyperparameter sensitive
- Less intuitive to debug

**Orb relevance:** MEDIUM. Could replace separate SFT + DPO with a single ORPO run.

**Rating:** ⭐⭐⭐⭐ (excellent for resource-constrained settings)

---

### GRPO — Group Relative Policy Optimization
**Origin:** DeepSeek AI (2025, used in DeepSeek-R1).

**Core insight:** Remove the critic model from PPO entirely. Instead of a value function, estimate advantages using group statistics: generate N responses to the same prompt, compare them to each other, use the relative quality as the reward signal.

**How it works:**
1. For each prompt, sample G responses (e.g., G=8)
2. Score all G responses using a reward function (rule-based for math/code: is the answer correct?)
3. Compute advantage for each response relative to the group mean
4. Optimize policy using these group-relative advantages
5. No value model, no critic network needed

```
Advantage(i) = (reward(i) - mean(rewards)) / std(rewards)
```

**Strengths:**
- No critic model (saves 50% memory vs PPO)
- Works excellently with verifiable rewards (math, code)
- Produces exceptional reasoning models (DeepSeek-R1 uses this)
- Simple to implement correctly
- Highly stable training

**Weaknesses:**
- Requires verifiable reward signals (hard for open-ended tasks)
- G responses per prompt = G× inference cost during training
- Less proven for general helpfulness (vs math/code)

**Key insight from DeepSeek-R1:** GRPO with pure rule-based rewards (correct/incorrect) on math problems produced reasoning chains that *spontaneously* developed self-correction, backtracking, and "aha moments" — without explicitly training for them. **This is the closest thing to emergent self-awareness demonstrated in 2025.**

**Orb relevance:** HIGH for math/code tasks. MEDIUM for general use.

**Rating:** ⭐⭐⭐⭐⭐ (best method for reasoning tasks)

---

### STaR — Self-Taught Reasoner
**Origin:** Zeiler et al., Google (2022).

**Core insight:** Bootstrap reasoning ability without human-labeled reasoning chains. Let the model generate its own chains, filter for ones that reach correct answers, and train on them.

**Algorithm:**
```
1. Start with dataset of (question, answer) pairs — NO reasoning chains
2. Prompt model: "Answer this step-by-step: {question}"
3. Check if the reasoning chain reaches the correct final answer
4. Keep (question, chain, answer) triplets where the answer is correct
5. Fine-tune the model on the kept triplets
6. Repeat from step 2 with the improved model
7. Handle failures: "rationalization" — show the correct answer, ask model to reason backwards
```

**Strengths:**
- No human annotation of reasoning chains needed
- Self-improving: each iteration produces a better model that generates better training data
- Works even with small models (originally demonstrated on GPT-2-scale)
- Directly teaches chain-of-thought reasoning

**Weaknesses:**
- Requires verifiable ground truth answers
- Reasoning chains can be "post-hoc" rationalization rather than genuine reasoning
- Slow convergence on hard problems
- Quality ceiling at the model's existing capability — can't learn what it can't yet do

**Orb relevance:** VERY HIGH. This is buildable on Replit CPU today. Core component of OASIS.

**Rating:** ⭐⭐⭐⭐⭐ (for self-improvement without human labels)

---

### SPIN — Self-Play Fine-Tuning
**Origin:** Chen et al., UCLA (2024).

**Core insight:** Use the model's own previous-checkpoint outputs as "negative examples." The model learns to produce better outputs than a slightly older version of itself, using a game-theoretic (self-play) framework.

**Algorithm:**
```
1. Fine-tune model M_0 with SFT → M_1
2. For each training example:
   - Generate response using M_0 (previous checkpoint) → "losing" response
   - Use ground truth SFT response → "winning" response
3. Train M_1 using DPO-style loss on (prompt, winning, losing) triplets
4. M_1 → M_2, repeat
```

**Why it works:** The model is always playing against a slightly weaker version of itself. This creates a stable, progressive improvement signal without any external feedback.

**Strengths:**
- Fully self-supervised — no human labels needed
- Generates its own negative examples automatically
- Stable convergence (unlike RL methods)
- Provable convergence in theory (Nash equilibrium)

**Weaknesses:**
- Quality bounded by SFT data quality
- Can converge to local optima if starting point is too poor
- Slower than supervised methods

**Orb relevance:** HIGH. Can implement this after initial SFT.

**Rating:** ⭐⭐⭐⭐ (excellent autonomous improvement method)

---

### ReST — Reinforced Self-Training
**Origin:** Gulcehre et al., Google DeepMind (2023).

**Core insight:** Alternate between "Grow" (generate lots of candidate responses) and "Improve" (filter by quality, train on the good ones). Simpler than RL.

**Algorithm:**
```
Repeat:
  GROW:    Generate N responses per prompt using current model
  IMPROVE: Filter responses using a quality threshold (reward model, rule-based, or human)
           Fine-tune on the filtered, high-quality subset
```

**Analogy:** Like evolution — generate variation, select the fittest, repeat.

**Strengths:**
- Simple: just inference + filtered fine-tuning
- No RL instability
- Works with any reward signal (rule-based, human, or AI)
- Proven effective on translation, summarization, math

**Weaknesses:**
- Can overfit if grow/improve cycles are too frequent
- Still requires a quality signal
- Slower than RL methods at equivalent compute

**Orb relevance:** MEDIUM-HIGH. Good complement to STaR.

**Rating:** ⭐⭐⭐⭐

---

### Process Reward Models (PRM)
**Origin:** Lightman et al., OpenAI (2023). Used in OpenAI's math models.

**Core insight:** Instead of rewarding only the final answer (Outcome Reward Model / ORM), reward each individual reasoning step. This gives much denser training signal.

**How it works:**
```
Step 1: "Let x = 5"         → Score: 0.9 (correct setup)
Step 2: "Then 5 × 3 = 15"   → Score: 0.8 (correct calculation)
Step 3: "So the answer is 25"→ Score: 0.1 (wrong conclusion)
```
The model learns that Step 3 is where the error occurred, not just that the final answer is wrong.

**Strengths:**
- Much better at identifying *where* the model goes wrong
- Enables "best-of-N" sampling with step-level verification
- Dramatically improves math and code reasoning
- Can be used for test-time compute scaling (more thinking = better answers)

**Weaknesses:**
- Requires step-level human annotation (expensive)
- Hard to apply to open-ended tasks
- Requires carefully designed step representations

**Orb relevance:** LOW for now (requires extensive annotation). Important for Orb-7B+ phase.

**Rating:** ⭐⭐⭐⭐ (excellent for reasoning tasks, high annotation cost)

---

## Efficiency Methods (Not Training Algorithms, But Critical)

### LoRA — Low-Rank Adaptation
**Origin:** Hu et al., Microsoft (2021).

**Core insight:** The weight updates during fine-tuning are intrinsically low-rank. Instead of updating all 117M parameters, decompose the update matrix into two small matrices:

```
ΔW = A × B    where A ∈ ℝ^(d×r), B ∈ ℝ^(r×k), r << min(d,k)
```

With rank r=8, this reduces trainable parameters by ~99.9% while preserving 90%+ of fine-tuning quality.

**For GPT-2:**
- Full fine-tuning: 117M trainable parameters
- LoRA (r=8): ~200K trainable parameters
- Memory reduction: ~99.8%

**This is what makes Replit fine-tuning possible.**

**QLoRA:** Quantize the base model to 4-bit, fine-tune LoRA adapters in 16-bit. Further reduces memory by 4×.

**Orb relevance:** CRITICAL. LoRA is how we fine-tune Orb on Replit.

---

### Mixture of Experts (MoE)
**Origin:** Shazeer et al., Google (2017). Revived in Mixtral (2023), DeepSeek V3 (2024).

**Core insight:** Instead of all parameters active for every token, use a router to activate only a subset of "expert" feed-forward networks per token.

```
Mixtral 8×7B:
- 8 expert FFN networks
- Router selects top-2 experts per token
- Active parameters per token: ~12B (out of 46B total)
- Cost: similar to a 12B model, quality of a 46B model
```

**Why it matters:** MoE models are dramatically more efficient than dense models. DeepSeek V3 (671B MoE) costs $6M to train — GPT-4 (dense, similar capability) cost an estimated $100M.

**Orb relevance:** Highly relevant for Orb-7B+ architecture design. Build Orb-7B as a 7B×8-expert MoE for 40× cost savings.

---

## Comparison Table

| Method | Needs Human Labels | Needs GPU | Complexity | Quality | Orb Priority |
|--------|-------------------|-----------|------------|---------|--------------|
| SFT | Dataset only | Low | Low | Medium | ★★★★★ |
| DPO | Preference pairs | Medium | Medium | High | ★★★★★ |
| RLHF | Human raters | High | Very High | Highest | ★★ |
| Constitutional AI | Just a constitution | Medium | High | Very High | ★★★★ |
| STaR | Verifiable answers | Low | Medium | High | ★★★★★ |
| SPIN | None | Low | Medium | High | ★★★★ |
| ReST | Quality signal | Low | Low | Medium-High | ★★★★ |
| GRPO | Verifiable answers | Medium | Medium | Very High | ★★★★ |
| ORPO | Preference pairs | Low | Low | High | ★★★★ |
| LoRA | — | Very Low | Low | — | ★★★★★ |

---

## What This Means for Orb (Replit CPU)

**Immediately buildable (today):**
- LoRA fine-tuning on OpenHermes-2.5 subset (10K examples)
- STaR loop on simple reasoning tasks
- Constitutional self-critique via prompting

**Buildable with Colab free tier:**
- Full SFT on OpenHermes-2.5 (1M examples)
- DPO on UltraFeedback
- SPIN for autonomous improvement

**Requires paid GPU:**
- GRPO for math reasoning
- Full Constitutional AI pipeline
- RLHF

The OASIS method (Document 04) is designed to get maximum value from the "immediately buildable" category.
