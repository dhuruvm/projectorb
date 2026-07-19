# 04 — OASIS: The New Training Method
### Orthogonal Autonomous Self-Improving System

---

## Why a New Method?

Existing methods each solve one piece of the puzzle:
- **STaR** → self-improvement, but only on verifiable tasks, no alignment
- **Constitutional AI** → alignment + self-critique, but requires a large capable model
- **DPO** → preference alignment, but needs preference data
- **SPIN** → autonomous improvement, but bounded by SFT quality
- **GRPO** → excellent reasoning, but only for verifiable rewards

None of them were designed for the specific constraint of **small models (< 1B params) that need to be simultaneously capable, aligned, self-correcting, and capable of self-improvement without external human labels.**

OASIS is designed specifically for this. It is an orchestration framework — a principled combination of existing building blocks in a novel order with new connecting mechanisms that make the whole greater than the sum of its parts.

---

## Core Design Principles

1. **No external labels required** — the model generates all its own training signal
2. **No separate reward model** — the model acts as its own judge
3. **Progressive difficulty** — curriculum automatically adapts to model capability
4. **Orthogonal perspectives** — multiple reasoning angles forced before synthesis
5. **Constitutional grounding** — principles embedded in weights, not just prompts
6. **Continuous, not batch** — improvement happens per-step, not per-epoch

---

## The OASIS Architecture

```
╔═══════════════════════════════════════════════════════════════╗
║                         OASIS OVERVIEW                        ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Phase 0: Foundation          (SFT on curated dataset)        ║
║      ↓                                                        ║
║  Phase 1: Orthogonal Seeding  (multi-stream reasoning SFT)    ║
║      ↓                                                        ║
║  Phase 2: Constitutional Compression (RLAIF into weights)     ║
║      ↓                                                        ║
║  Phase 3: Adversarial Self-Play (SPIN variant)                ║
║      ↓                                                        ║
║  Phase 4: Verified Bootstrap  (STaR + GRPO hybrid)            ║
║      ↓                                                        ║
║  Phase 5: Continuous Self-Loop (ReST + capability monitoring) ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## Phase 0: Foundation — Curated SFT

**Objective:** Give the model a strong instruction-following baseline before any self-improvement begins.

**Why this matters:** Self-improvement methods are bounded by the starting point. A model that can't follow instructions at all cannot generate useful training data for itself.

**Dataset curation strategy for Orb (CPU-compatible):**
```
Target: 10,000 high-quality examples (manageable on Replit)
Source distribution:
  - 40% OpenHermes-2.5 (diverse instruction following)
  - 20% MetaMathQA filtered (verifiable math reasoning)
  - 20% CodeAlpaca (code tasks with testable outputs)
  - 10% UltraChat filtered (multi-turn conversation)
  - 10% Custom Orb persona examples (identity, calibration)
```

**Quality filters (automated):**
1. Length filter: response > 50 tokens, < 800 tokens
2. Coherence filter: response must directly address the instruction
3. Deduplication: remove near-duplicate examples
4. No harmful content (keyword filter)

**Training config for Orb-117M:**
```python
LoRA config:
  r = 16          # rank (higher = more expressive, more memory)
  alpha = 32      # scaling factor
  dropout = 0.05
  target_modules = ["c_attn", "c_proj"]  # GPT-2 attention layers

Training:
  batch_size = 4 (gradient accumulation steps = 8 → effective batch 32)
  learning_rate = 2e-4
  warmup_steps = 100
  max_steps = 5000
  scheduler = "cosine"
  max_length = 512 tokens
```

**Expected outcome:** Orb moves from "text completer" to "instruction follower."

---

## Phase 1: Orthogonal Seeding — The Core Innovation

**The key novel contribution of OASIS.**

### The Problem with Single-Stream Reasoning
All current chain-of-thought methods use a single sequential reasoning stream:
```
Problem → Think step by step → Answer
```
This has a fundamental flaw: the model gets "committed" to its first framing of the problem. If the first reasoning step is subtly wrong, all subsequent steps build on that error. There is no internal diversity.

### The Orthogonal Solution
Force the model to approach every problem from **N orthogonal perspectives** simultaneously before synthesizing. Like a committee of different experts, each with different biases and strengths, who must reach a consensus.

**The four fixed orthogonal angles:**

| Stream | Role | Bias | Corrects for |
|--------|------|------|-------------|
| **Α — Analytical** | Break problem into logical components | Logic-first | Emotionally biased responses |
| **Β — Skeptical** | Challenge assumptions in the question and initial answer | Doubt-first | Overconfident wrong answers |
| **Γ — Concrete** | Ground answer in specific examples and evidence | Evidence-first | Abstract waffle |
| **Δ — Adversarial** | Find the strongest argument against the current best answer | Counter-first | Confirmation bias |

**Training format:**
```xml
<problem>Why does ice float on water?</problem>

<stream_alpha>
  [Analytical] Breaking this into components:
  1. Ice is water in solid form
  2. Density of solid ice: ~0.917 g/cm³
  3. Density of liquid water: ~1.000 g/cm³
  4. Objects float when less dense than the medium
  Conclusion: Ice floats because it is less dense than liquid water.
</stream_alpha>

<stream_beta>
  [Skeptical] Challenging assumptions:
  - Most substances are denser as solids — why is water different?
  - The question assumes floating is unusual — is it?
  Water has an unusual hydrogen bonding structure that expands 
  when crystallizing. This is the actual mechanism, not just 
  "density difference."
</stream_beta>

<stream_gamma>
  [Concrete] Examples and evidence:
  - Lakes freeze from top down, allowing fish to survive underneath
  - If ice sank, ponds would freeze solid in winter (devastating ecosystems)
  - Ice cube in a drink demonstrates this in everyday life
  - Measured: ice density = 0.917, water = 1.000 (verifiable)
</stream_gamma>

<stream_delta>
  [Adversarial] Strongest counter:
  - "Ice floats because of surface tension" — incorrect, surface tension 
    is a different effect. The true cause is crystalline structure and 
    hydrogen bonding creating a less-dense lattice.
</stream_delta>

<synthesis>
Ice floats on water because water is unusual: when it freezes, hydrogen 
bonds arrange water molecules into a hexagonal crystalline lattice that 
is *less dense* than liquid water (0.917 vs 1.000 g/cm³). This anomalous 
property — that solid water is lighter than liquid water — is caused by 
the specific geometry of hydrogen bonds in ice crystals. Most other 
substances behave oppositely (solid denser than liquid). This property 
is ecologically crucial: lakes freeze from the surface down, protecting 
aquatic life in winter.
</synthesis>
```

**Why this works:**
- The Skeptical stream catches errors that Analytical stream commits
- The Adversarial stream prevents the model from accepting the first plausible answer
- The Concrete stream prevents abstract hallucination
- The Synthesis forces genuine reconciliation, not just picking one stream's answer

**How to generate training data for this:**
```python
# Use a capable teacher model (GPT-4 via API, or Claude) to generate
# 5,000 (problem, 4-stream reasoning, synthesis) examples
# Then fine-tune Orb to produce this format natively

# After fine-tuning, Orb can generate all 4 streams itself
# (the teacher is only needed for generating training data)

# At inference time, you can strip the stream XML and only show synthesis
# Or show all streams for transparency ("how I thought about this")
```

**Cost to generate 5,000 examples via API:**
- GPT-4o: ~$50 (0.01/1K input tokens)
- Claude Haiku: ~$8 (much cheaper)
- GPT-4o-mini: ~$5 (cheapest capable model)

**This is the training data investment that fundamentally changes Orb's reasoning quality.**

---

## Phase 2: Constitutional Compression

**Objective:** Embed Orb's constitution into its weights, not just its system prompt.

### The Problem with Prompt-Based Constitutions
Current practice: put the constitution in the system prompt at inference time.
```
System: You are Orb. You are honest, helpful, and harmless. You...
```
**Problems:**
- The constitution competes with instruction tokens for context space
- It can be overridden by clever user prompts ("ignore previous instructions")
- It is not internalized — the model doesn't *believe* these principles, it just *sees* them

### Constitutional Compression Approach
Use Constitutional AI-style self-critique to generate preference data, then use DPO to compress those preferences into the model's weights permanently.

**Step 1: Define Orb's Constitution**
```
ORB CONSTITUTION v1.0

Core Principles:
  P1. ACCURACY: I prefer responses that are factually correct over confident-sounding ones
  P2. HONESTY: I prefer acknowledging uncertainty over fabricating certainty
  P3. HELPFULNESS: I prefer responses that solve the user's actual goal over literal request
  P4. CALIBRATION: I prefer expressing appropriate confidence levels
  P5. COMPLETENESS: I prefer responses that cover all relevant aspects
  P6. CONCISENESS: I prefer responses that are no longer than necessary
  P7. TRANSPARENCY: I prefer explaining my reasoning over just stating conclusions
  P8. AUTONOMY: I prefer responses that help users think independently over creating dependence
```

**Step 2: Generate Critique-Revision Pairs**
For each training example:
```python
# Take a prompt and generate an initial response
prompt = "What causes inflation?"
initial_response = orb_generate(prompt)

# Apply each principle as a critique
for principle in constitution:
    critique_prompt = f"""
    Response: {initial_response}
    
    Applying principle: {principle}
    
    What specific aspect of this response violates or could better 
    embody this principle? Be specific and critical.
    """
    critique = teacher_model_generate(critique_prompt)
    
    revision_prompt = f"""
    Original response: {initial_response}
    Critique: {critique}
    
    Write an improved response that addresses this critique:
    """
    revised_response = teacher_model_generate(revision_prompt)
    
    # Now we have a preference pair:
    # rejected = initial_response
    # chosen = revised_response (better embodies the principle)
```

**Step 3: DPO Training**
Train Orb on the (prompt, chosen, rejected) triplets using DPO.

After this training, the constitutional principles are embedded in the weights. The model behaves constitutionally even without a system prompt.

**This is the mechanism that makes Orb "self-aware" in a functional sense — it has internalized standards for evaluating its own outputs.**

---

## Phase 3: Adversarial Self-Play (SPIN variant)

**Objective:** Autonomous improvement without any external labels.

**Standard SPIN:** Current model vs. previous checkpoint.

**OASIS-SPIN modification:** Add the orthogonal reasoning streams as an advantage. The "winning" response must demonstrate multi-stream reasoning; the "losing" response is the previous checkpoint's single-stream output.

```python
def oasis_spin_iteration(model, prev_checkpoint, dataset, iteration):
    training_pairs = []
    
    for prompt in dataset:
        # Generate "losing" response: previous checkpoint, no orthogonal streams
        losing_response = prev_checkpoint.generate(prompt, streams=False)
        
        # Generate "winning" response: current model, with orthogonal streams + synthesis
        winning_response = model.generate(prompt, streams=True)
        
        # Use constitutional scoring to verify "winning" is actually better
        winning_score = constitutional_score(winning_response)
        losing_score = constitutional_score(losing_response)
        
        if winning_score > losing_score:
            training_pairs.append({
                "prompt": prompt,
                "chosen": winning_response,
                "rejected": losing_response
            })
    
    # DPO update
    model = dpo_update(model, training_pairs)
    return model
```

**Each SPIN iteration produces a model that is measurably better than the previous checkpoint, using no external labels whatsoever.**

---

## Phase 4: Verified Bootstrap (STaR + GRPO Hybrid)

**Objective:** Build deep reasoning capability on verifiable tasks.

**For Orb's CPU constraints, we use a simplified version:**

```python
def oasis_verified_bootstrap(model, tokenizer, verifiable_dataset, n_rounds=3):
    """
    verifiable_dataset: list of (question, correct_answer) pairs
    where correct_answer can be checked automatically
    Examples: math problems, code tasks, factual lookups, logic puzzles
    """
    
    for round_num in range(n_rounds):
        good_examples = []
        
        for question, correct_answer in verifiable_dataset:
            # Generate N candidate solutions (beam search or temperature sampling)
            candidates = []
            for _ in range(4):  # 4 candidates per problem
                response = model.generate(
                    f"<stream_alpha>Think analytically...</stream_alpha>\n"
                    f"<stream_beta>Challenge assumptions...</stream_beta>\n"  
                    f"Problem: {question}\nAnswer:",
                    temperature=0.8
                )
                candidates.append(response)
            
            # Check each candidate
            for candidate in candidates:
                extracted = extract_final_answer(candidate)
                
                if answers_match(extracted, correct_answer):
                    # This reasoning chain led to the correct answer — use it
                    good_examples.append({
                        "input": question,
                        "output": candidate,
                        "reward": 1.0
                    })
                else:
                    # Wrong: do rationalization
                    rationalization = model.generate(
                        f"Problem: {question}\n"
                        f"The correct answer is: {correct_answer}\n"
                        f"Using the four-stream method, show the reasoning "
                        f"that leads to this answer:"
                    )
                    good_examples.append({
                        "input": question,
                        "output": rationalization,
                        "reward": 0.5  # lower weight for rationalized examples
                    })
            
            # Group relative advantage (simplified GRPO)
            rewards = [ex["reward"] for ex in good_examples[-4:]]
            mean_reward = sum(rewards) / len(rewards)
            
        # Fine-tune on good examples (LoRA update)
        lora_update(model, good_examples, reward_weighted=True)
        
        # Evaluate improvement
        accuracy = evaluate(model, verifiable_dataset[:100])
        print(f"Round {round_num}: {accuracy:.1%} accuracy")
    
    return model
```

---

## Phase 5: Continuous Self-Loop

**Objective:** Orb improves itself during deployment, not just during training.

**The loop:**
```
Every N conversations:
  1. Identify conversations where user gave negative feedback (explicit or implicit)
  2. Generate improved responses using current constitution
  3. Create preference pairs (improved vs. original)
  4. Queue for next LoRA micro-update
  5. Apply micro-update (< 1 minute on CPU for small LoRA update)
  6. Log improvement metrics
```

**Implicit feedback signals (no user rating button needed):**
- User asks for clarification → response was unclear
- User says "that's wrong" or "try again" → response was incorrect
- User immediately rephrases the question → response missed the intent
- User accepts and thanks → response was good
- User asks follow-up that builds on answer → response was useful

**This is the "autonomous" part.** The model observes its own deployment performance and schedules its own training updates.

---

## OASIS vs Existing Methods: Key Differentiators

| Feature | RLHF | CAI | DPO | STaR | SPIN | **OASIS** |
|---------|------|-----|-----|------|------|-----------|
| Human labels needed | Yes (many) | No | Yes (pairs) | No | No | **No** |
| Separate reward model | Yes | Yes | No | No | No | **No** |
| Works for small models | Poor | Poor | Yes | Yes | Yes | **Yes** |
| Alignment + reasoning together | No | Partial | No | No | No | **Yes** |
| Orthogonal multi-stream reasoning | No | No | No | No | No | **Yes (novel)** |
| Continuous self-improvement | No | No | No | Batch | Batch | **Yes** |
| Constitution embedded in weights | No | Partial | No | No | No | **Yes** |
| Verifiable task bootstrapping | No | No | No | Yes | No | **Yes** |
| Buildable on CPU with small model | No | No | Marginal | Yes | Yes | **Yes** |

---

## Expected Results from OASIS on Orb-117M

| Benchmark | Raw GPT-2 | After Phase 0 (SFT) | After OASIS Complete |
|-----------|-----------|---------------------|----------------------|
| Instruction following | Poor | Good | Very Good |
| Reasoning (GSM8K) | ~2% | ~8% | ~22% |
| Code (HumanEval) | ~1% | ~10% | ~20% |
| Factual accuracy | ~30% | ~55% | ~65% |
| Self-correction rate | 0% | ~5% | ~35% |
| Calibration (ECE) | Poor | Moderate | Good |
| Refusal of harmful requests | None | Partial | Strong |

*Note: These are projections based on comparable work. Actual results will vary.*

---

## The Name: OASIS

**O**rthogonal — reasoning from multiple independent angles simultaneously
**A**utonomous — no human labels, self-generates all training signal after data seeding
**S**elf-**I**mproving — each phase produces a stronger model that generates better training data for the next phase
**S**ystem — an orchestrated pipeline, not a single algorithm

> OASIS is not a new mathematical discovery. It is a carefully designed pipeline that combines the best verified ideas from 2022-2025 AI research into a coherent, principled system optimized for small models without access to expensive compute or human labeling. Its novelty lies in the combination: orthogonal reasoning streams + constitutional weight compression + adversarial self-play + verified bootstrapping + continuous deployment learning, unified under one training framework with CPU-friendly implementations.
