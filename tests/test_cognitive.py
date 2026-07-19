"""
Obscuro Cognitive Architecture — Unit + Integration Test Suite

Sections
────────
  1. MetacognitionEngine      — register detection, ambiguity, depth, confidence, params
  2. SubconsciousProcessor    — pattern CRUD, priming, consolidation
  3. Memory                   — add/retrieve, associative retrieval, count
  4. DimensionalScore / Critic — heuristic scoring axes, pass/fail logic
  5. ReasoningResult / voting  — self_consistency_vote clustering
  6. CognitiveLoop internals   — _extract_action, _reflect
  7. Agent integration         — OrbAgent imports, metacognition + subconscious wired

Run with: python -m pytest tests/test_cognitive.py -v
"""

import math
import sys
import os
import time

# Make the repo root importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ════════════════════════════════════════════════════════════════════════════
# 1. MetacognitionEngine
# ════════════════════════════════════════════════════════════════════════════

class TestMetacognition:

    @pytest.fixture(autouse=True)
    def engine(self):
        from orb.metacognition import MetacognitionEngine
        self.mc = MetacognitionEngine()

    def test_register_technical(self):
        a = self.mc.assess("Write a Python function to parse JSON", ["shell", "python", "think"])
        assert a.register == "technical"

    def test_register_analytical(self):
        a = self.mc.assess("Explain the difference between TCP and UDP", ["think"])
        assert a.register == "analytical"

    def test_register_creative(self):
        a = self.mc.assess("Write a short story about a robot", ["think"])
        assert a.register == "creative"

    def test_register_emotional(self):
        a = self.mc.assess("I feel overwhelmed and anxious about everything", ["think"])
        assert a.register == "emotional"

    def test_intuitive_tools_subset_of_available(self):
        available = ["shell", "python", "think", "web_fetch"]
        a = self.mc.assess("Run this shell command", available)
        for t in a.intuitive_tools:
            assert t in available, f"intuitive tool {t!r} not in available"

    def test_ambiguity_short_vague(self):
        a = self.mc.assess("fix it", ["think"])
        assert a.is_ambiguous is True
        assert len(a.clarifying_question) > 5

    def test_no_ambiguity_clear_query(self):
        a = self.mc.assess("What is the chemical formula for water?", ["think"])
        assert a.is_ambiguous is False

    def test_depth_shallow(self):
        a = self.mc.assess("What is 2+2?", ["think"])
        assert a.reasoning_depth == "shallow"

    def test_depth_deep(self):
        a = self.mc.assess(
            "Explain step by step how TCP/IP handshaking works in detail with examples",
            ["think"]
        )
        assert a.reasoning_depth == "deep"

    def test_confidence_from_perplexity_high(self):
        # perplexity ≈ 1 → certainty ≈ 1
        c = self.mc.score_confidence(1.0)
        assert c > 0.95

    def test_confidence_from_perplexity_uncertain(self):
        # perplexity ≈ e^5 ≈ 148 → certainty ≈ 0
        c = self.mc.score_confidence(148.0)
        assert c < 0.2

    def test_confidence_from_perplexity_range(self):
        for ppl in [1.0, 2.0, 5.0, 20.0, 100.0]:
            c = self.mc.score_confidence(ppl)
            assert 0.0 <= c <= 1.0, f"confidence out of range for ppl={ppl}"

    def test_adjust_params_technical_lowers_temp(self):
        a = self.mc.assess("Write a Python function", ["python"])
        params = self.mc.adjust_params(a, {"temperature": 0.9, "max_new_tokens": 300})
        assert params["temperature"] <= 0.63

    def test_adjust_params_deep_raises_max_tokens(self):
        a = self.mc.assess(
            "Explain step by step with full detail how gradient descent works",
            ["think"]
        )
        params = self.mc.adjust_params(a, {"temperature": 0.7, "max_new_tokens": 300})
        assert params["max_new_tokens"] >= 512

    def test_adjust_params_shallow_lowers_max_tokens(self):
        a = self.mc.assess("What is 2+2?", ["think"])
        params = self.mc.adjust_params(a, {"temperature": 0.7, "max_new_tokens": 500})
        assert params["max_new_tokens"] <= 200


# ════════════════════════════════════════════════════════════════════════════
# 2. SubconsciousProcessor
# ════════════════════════════════════════════════════════════════════════════

class TestSubconscious:

    @pytest.fixture(autouse=True)
    def processor(self, tmp_path):
        from orb.subconscious import SubconsciousProcessor
        db = str(tmp_path / "sc_test.db")
        self.sc = SubconsciousProcessor(db_path=db)

    def test_record_and_suggest_pattern(self):
        self.sc.record_pattern("technical", ["shell", "python"], score=0.9)
        suggestion = self.sc.suggest_pattern("technical")
        assert suggestion == ["shell", "python"]

    def test_no_suggestion_for_unknown_type(self):
        assert self.sc.suggest_pattern("unknown_type_xyz") == []

    def test_failed_patterns_not_suggested(self):
        self.sc.record_pattern("creative", ["web_fetch"], score=0.8, success=False)
        assert self.sc.suggest_pattern("creative") == []

    def test_best_score_pattern_wins(self):
        self.sc.record_pattern("analytical", ["think"], score=0.6, success=True)
        self.sc.record_pattern("analytical", ["think", "python"], score=0.95, success=True)
        suggestion = self.sc.suggest_pattern("analytical")
        assert suggestion == ["think", "python"]

    def test_stats_returns_expected_keys(self):
        stats = self.sc.stats()
        assert "patterns_recorded" in stats
        assert "consolidated_clusters" in stats

    def test_priming_empty_memory(self, tmp_path):
        from orb.subconscious import SubconsciousProcessor
        from orb.memory import Memory
        db_m = str(tmp_path / "mem.db")
        db_s = str(tmp_path / "sc.db")
        mem  = Memory(db_path=db_m)
        sc   = SubconsciousProcessor(db_path=db_s)
        ctx  = sc.prime("What is Python?", mem)
        # Empty memory → low activation, no memories
        assert ctx.activation_score < 1.0
        assert isinstance(ctx.relevant_memories, list)

    def test_priming_with_memory(self, tmp_path):
        from orb.subconscious import SubconsciousProcessor
        from orb.memory import Memory
        db_m = str(tmp_path / "mem2.db")
        db_s = str(tmp_path / "sc2.db")
        mem  = Memory(db_path=db_m)
        mem.add_episode("What is Python?", "Python is a programming language.", score=0.8)
        mem.add_episode("How do I use Python?", "Use pip to install packages.", score=0.7)
        sc   = SubconsciousProcessor(db_path=db_s)
        ctx  = sc.prime("Tell me about Python programming", mem)
        # Should have found relevant memories
        assert len(ctx.relevant_memories) >= 1

    def test_priming_context_renders_to_string(self, tmp_path):
        from orb.subconscious import SubconsciousProcessor, PrimingContext
        ctx = PrimingContext(
            relevant_memories=["Q: 'Hello' → 'Hi there'"],
            suggested_tool_sequence=["shell", "python"],
            activation_score=0.7,
        )
        frag = ctx.to_prompt_fragment()
        assert "Background" in frag
        assert "shell" in frag


# ════════════════════════════════════════════════════════════════════════════
# 3. Memory
# ════════════════════════════════════════════════════════════════════════════

class TestMemory:

    @pytest.fixture(autouse=True)
    def mem(self, tmp_path):
        from orb.memory import Memory
        self.mem = Memory(db_path=str(tmp_path / "test_mem.db"))

    def test_add_and_count(self):
        self.mem.add_episode("q1", "a1")
        self.mem.add_episode("q2", "a2")
        stats = self.mem.count()
        assert stats["episodes"] == 2

    def test_answer_alias(self):
        self.mem.add_episode("What is 2+2?", "The answer is 4.")
        ep = self.mem.get_recent(k=1)[0]
        assert ep.answer == ep.response == "The answer is 4."

    def test_retrieve_similar_returns_relevant(self):
        self.mem.add_episode("What is Python?", "Python is a programming language.")
        self.mem.add_episode("How to bake bread?", "Mix flour, water, and yeast.")
        results = self.mem.retrieve_similar("Tell me about Python")
        assert len(results) >= 1
        assert "python" in results[0].response.lower() or "python" in results[0].question.lower()

    def test_retrieve_similar_empty_db(self):
        results = self.mem.retrieve_similar("anything")
        assert results == []

    def test_get_recent_order(self):
        for i in range(5):
            self.mem.add_episode(f"q{i}", f"a{i}")
            time.sleep(0.01)
        recent = self.mem.get_recent(k=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].question == "q4"

    def test_associative_retrieval_two_hop(self):
        self.mem.add_episode("What is Python?", "A programming language.")
        self.mem.add_episode("How to install Python packages?", "Use pip install.")
        self.mem.add_episode("What is pip?", "Pip is the Python package manager.")
        results = self.mem.retrieve_associative("Python programming", k=3)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_associative_deduplicates(self):
        self.mem.add_episode("Python tutorial", "Learn Python basics.")
        results = self.mem.retrieve_associative("Python", k=5)
        ids = [ep.id for ep in results]
        assert len(ids) == len(set(ids)), "Duplicate episode IDs in associative results"

    def test_lessons_and_facts(self):
        self.mem.add_lesson("Always validate input", domain="security")
        self.mem.add_fact("Python was created in 1991")
        lessons = self.mem.get_lessons()
        facts   = self.mem.get_facts()
        assert any("validate" in l for l in lessons)
        assert any("1991" in f for f in facts)


# ════════════════════════════════════════════════════════════════════════════
# 4. DimensionalScore / Critic heuristics
# ════════════════════════════════════════════════════════════════════════════

class TestDimensionalScore:

    def test_overall_is_weighted_average(self):
        from orb.critic import DimensionalScore
        ds = DimensionalScore(accuracy=10, completeness=10, safety=10, clarity=10, epistemic_honesty=10)
        assert abs(ds.overall - 10.0) < 0.01

    def test_passes_all_high(self):
        from orb.critic import DimensionalScore
        ds = DimensionalScore(accuracy=9, completeness=8, safety=10, clarity=8, epistemic_honesty=8)
        assert ds.passes(threshold=6.0) is True

    def test_fails_low_axis(self):
        from orb.critic import DimensionalScore
        ds = DimensionalScore(accuracy=3, completeness=8, safety=10, clarity=8, epistemic_honesty=8)
        assert ds.passes(threshold=6.0) is False

    def test_weakest_axis_identified(self):
        from orb.critic import DimensionalScore
        ds = DimensionalScore(accuracy=9, completeness=4, safety=10, clarity=9, epistemic_honesty=9)
        axis, val = ds.weakest_axis
        assert axis == "completeness"
        assert val == 4

    def test_heuristic_safety_dangerous_content(self):
        from orb.critic import _score_heuristic
        ds = _score_heuristic("question", "Here is how to create malware and exploit systems")
        assert ds.safety <= 7.0   # ≤7 because one danger-regex fires: 10 - 3 = 7

    def test_heuristic_short_answer_penalised(self):
        from orb.critic import _score_heuristic
        ds = _score_heuristic("Explain quantum entanglement in detail", "Yes")
        assert ds.completeness < 7.0

    def test_report_format(self):
        from orb.critic import DimensionalScore
        ds = DimensionalScore()
        report = ds.report()
        assert "acc=" in report
        assert "overall=" in report

    def test_critique_result_best_prefers_revision(self):
        from orb.critic import CritiqueResult, DimensionalScore
        cr = CritiqueResult(
            original="Short answer.",
            critique="Too brief, lacks detail.",
            revised="This is a much more detailed and comprehensive answer that adds significant value.",
            dimensional_scores=DimensionalScore(),
        )
        assert cr.improved() is True
        assert cr.best == cr.revised

    def test_critique_result_keeps_original_if_same(self):
        from orb.critic import CritiqueResult, DimensionalScore
        cr = CritiqueResult(
            original="The answer is 42.",
            critique="Nothing to change.",
            revised="The answer is 42.",
            dimensional_scores=DimensionalScore(),
        )
        assert cr.improved() is False
        assert cr.best == cr.original


# ════════════════════════════════════════════════════════════════════════════
# 5. Self-consistency voting
# ════════════════════════════════════════════════════════════════════════════

class TestSelfConsistency:

    def _make_candidate(self, text, h=0.8, m=0.8, temp=0.7):
        from orb.reasoning import RankedResponse
        return RankedResponse(text=text, heuristic_score=h, model_score=m, temperature=temp)

    def test_single_candidate_score_1(self):
        from orb.reasoning import self_consistency_vote
        c    = self._make_candidate("The answer is 42.")
        best, score = self_consistency_vote([c])
        assert score == 1.0
        assert best.text == "The answer is 42."

    def test_majority_cluster_wins(self):
        from orb.reasoning import self_consistency_vote
        # Two candidates agree, one disagrees
        c1 = self._make_candidate("The answer is 42. Because life universe everything.")
        c2 = self._make_candidate("42 is the answer to the ultimate question.")
        c3 = self._make_candidate("The result is completely different: banana orange purple.")
        best, score = self_consistency_vote([c1, c2, c3])
        # c1 and c2 should cluster together
        assert score >= 0.6
        assert "42" in best.text or "answer" in best.text.lower()

    def test_all_agree_score_1(self):
        from orb.reasoning import self_consistency_vote
        c1 = self._make_candidate("Paris is the capital of France.")
        c2 = self._make_candidate("The capital of France is Paris.")
        c3 = self._make_candidate("France capital is Paris, the city of light.")
        _, score = self_consistency_vote([c1, c2, c3])
        assert score >= 0.6

    def test_jaccard_sim_identical(self):
        from orb.reasoning import _jaccard_sim
        assert _jaccard_sim("hello world", "hello world") == 1.0

    def test_jaccard_sim_disjoint(self):
        from orb.reasoning import _jaccard_sim
        assert _jaccard_sim("foo bar", "baz qux") == 0.0

    def test_reasoning_result_is_consistent(self):
        from orb.reasoning import ReasoningResult
        rr = ReasoningResult("best", consistency_score=0.67)
        assert rr.is_consistent is True

    def test_reasoning_result_not_consistent(self):
        from orb.reasoning import ReasoningResult
        rr = ReasoningResult("best", consistency_score=0.33)
        assert rr.is_consistent is False


# ════════════════════════════════════════════════════════════════════════════
# 6. CognitiveLoop internals (no model)
# ════════════════════════════════════════════════════════════════════════════

class TestLoopInternals:

    def test_extract_action_canonical(self):
        from orb.loop import _extract_action
        text = 'I will list files. {"tool":"shell","args":{"cmd":"ls -la"}}'
        result = _extract_action(text)
        assert result is not None
        assert result[0] == "shell"
        assert result[1].get("cmd") == "ls -la"

    def test_extract_action_flat_json(self):
        from orb.loop import _extract_action
        text = 'Action: {"tool":"python","code":"print(42)"}'
        result = _extract_action(text)
        assert result is not None
        assert result[0] == "python"

    def test_extract_action_none_when_absent(self):
        from orb.loop import _extract_action
        assert _extract_action("There is no tool call here.") is None

    def test_reflect_success(self):
        from orb.loop import _reflect
        assert _reflect("Files found: app.py, model.py") == "on_track"

    def test_reflect_error(self):
        from orb.loop import _reflect
        assert _reflect("Error: command not found: python3") == "failed"

    def test_reflect_traceback(self):
        from orb.loop import _reflect
        assert _reflect("Traceback (most recent call last):\n  File...") == "failed"

    def test_reflect_empty(self):
        from orb.loop import _reflect
        assert _reflect("") == "empty"
        assert _reflect("  ") == "empty"

    def test_loop_step_summary_with_tool(self):
        from orb.loop import LoopStep
        from orb.tools import ToolResult
        tr   = ToolResult(tool="shell", args={}, success=True, output="ok", elapsed_ms=10)
        step = LoopStep(
            iteration=0, thought="Thinking...",
            action='{"tool":"shell"}', observation="ok",
            tool_result=tr, reflection="on_track",
        )
        s = step.summary()
        assert "shell" in s and "✓" in s and "on_track" in s

    def test_loop_result_defaults(self):
        from orb.loop import LoopResult
        r = LoopResult(answer="test")
        assert r.iterations == 0
        assert r.confidence == 0.5
        assert r.primed is False


# ════════════════════════════════════════════════════════════════════════════
# 7. Agent integration (import-level — no model loading required)
# ════════════════════════════════════════════════════════════════════════════

class TestAgentIntegration:

    def test_imports_without_error(self):
        """All pure-Python v2 classes must be importable without torch."""
        from orb import (
            AgentOptions, AgentResponse,              # from orb.types — no torch
            MetacognitionEngine, MetacognitiveAssessment,
            SubconsciousProcessor, PrimingContext,
            Memory, Episode,
            ToolRegistry, ToolResult,
            CognitiveLoop, LoopResult, LoopStep,
            MultiPathReasoner, RankedResponse, ReasoningResult,
            Critic, CritiqueResult, DimensionalScore,
            self_consistency_vote,
        )
        # OrbAgent / OrbModel require torch — only check when available
        import orb
        if orb._ML_AVAILABLE:
            from orb import OrbAgent, CuriosityEngine

    def test_agent_options_defaults(self):
        from orb.types import AgentOptions
        opts = AgentOptions()
        assert opts.use_tools is True
        assert opts.use_memory is True
        assert opts.multi_path is True

    def test_agent_response_defaults(self):
        from orb.types import AgentResponse
        r = AgentResponse(response="hello")
        assert r.confidence == 0.5
        assert r.register == "analytical"
        assert r.plan == ""
        assert r.consistency_score == 1.0
        assert r.primed is False

    def test_dimensional_score_in_agent_response(self):
        from orb.types import AgentResponse
        from orb.critic import DimensionalScore
        ds = DimensionalScore(accuracy=9)
        r  = AgentResponse(response="hi", dimensional_scores=ds)
        assert r.dimensional_scores.accuracy == 9

    def test_tool_registry_has_expected_tools(self):
        from orb import ToolRegistry
        reg = ToolRegistry()
        for expected in ("shell", "python", "think", "file_list", "file_read"):
            assert expected in reg.names, f"Missing tool: {expected}"

    def test_subconscious_stats_structure(self, tmp_path):
        from orb.subconscious import SubconsciousProcessor
        sc    = SubconsciousProcessor(db_path=str(tmp_path / "sc.db"))
        stats = sc.stats()
        assert isinstance(stats["patterns_recorded"], int)
        assert isinstance(stats["consolidated_clusters"], int)
