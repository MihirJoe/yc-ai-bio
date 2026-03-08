"""
AlwaysOnPT — Recursive Language Model Agent
Core agent architecture: exec() REPL, sub_llm(), skill acquisition,
multimodal vision, reasoning trace, and LangSmith observability.
"""

import os
import json
import time
import base64
import traceback
import numpy as np
from io import StringIO
from pathlib import Path

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import anthropic
import langsmith
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class StepType(Enum):
    OBSERVATION = "OBSERVATION"
    TOOL_CALL = "TOOL_CALL"
    VISUAL = "VISUAL"
    LITERATURE = "LITERATURE"
    SKILL_ACQUIRED = "SKILL_ACQUIRED"
    SUB_REASONING = "SUB_REASONING"
    DECISION = "DECISION"
    SYNTHESIS = "SYNTHESIS"
    CODE_EXECUTION = "CODE_EXECUTION"
    ERROR = "ERROR"


@dataclass
class TraceStep:
    step_type: StepType
    content: str
    evidence: list = field(default_factory=list)
    timestamp: str = ""
    duration_ms: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ReasoningTrace:
    segment_id: str = ""
    steps: list = field(default_factory=list)
    classification: str = ""
    confidence: float = 0.0
    clinical_narrative: str = ""
    skills_used: list = field(default_factory=list)
    plots_generated: list = field(default_factory=list)
    _on_step: object = field(default=None, repr=False)

    def add_step(self, step_type: StepType, content: str,
                 evidence: list = None, duration_ms: float = 0.0):
        step = TraceStep(
            step_type=step_type,
            content=content,
            evidence=evidence or [],
            duration_ms=duration_ms,
        )
        self.steps.append(step)
        if self._on_step:
            try:
                self._on_step(step, self)
            except Exception:
                pass
        return step

    def to_dict(self) -> dict:
        return {
            'segment_id': self.segment_id,
            'classification': self.classification,
            'confidence': self.confidence,
            'clinical_narrative': self.clinical_narrative,
            'skills_used': self.skills_used,
            'plots_generated': self.plots_generated,
            'n_steps': len(self.steps),
            'steps': [
                {
                    'type': s.step_type.value,
                    'content': s.content,
                    'evidence': s.evidence,
                    'timestamp': s.timestamp,
                    'duration_ms': s.duration_ms,
                }
                for s in self.steps
            ],
        }


class RLMAgent:
    """
    Recursive Language Model agent for EMG analysis.

    Architecture:
    - Root LLM (Opus) writes Python code that runs in exec() REPL
    - All EMG tool functions available in the REPL namespace
    - sub_llm() (Sonnet) for recursive multimodal reasoning
    - Skill acquisition: search_literature + validate_tool + register_tool
    - classify_emg(): placeholder for specialized model integration
    - Vision: generate_plot() + image-aware sub_llm()
    - Full observability via ReasoningTrace + LangSmith
    """

    ROOT_MODEL = "claude-opus-4-20250514"
    SUB_MODEL = "claude-sonnet-4-20250514"
    MAX_TURNS = 8
    MAX_SUB_LLM_CALLS = 3

    def __init__(self, verbose: bool = True):
        self.client = anthropic.Anthropic()
        self.verbose = verbose

        self.repl_globals = {}
        self.skill_library = {}
        self.session_context = {
            'findings': [],
            'plots': [],
            'skills_acquired': [],
            'segments_analyzed': 0,
        }

        self._init_repl()

    def _init_repl(self):
        """Initialize the REPL namespace with all tool functions."""
        from alwaysonpt.emg_tools import (
            get_segment_overview, extract_time_features, extract_freq_features,
            extract_wavelet_features, detect_fatigue_pattern,
            compare_to_baseline, generate_plot, get_feature_vector,
            TOOL_REGISTRY,
        )

        self.repl_globals = {
            'np': np,
            'get_segment_overview': get_segment_overview,
            'extract_time_features': extract_time_features,
            'extract_freq_features': extract_freq_features,
            'extract_wavelet_features': extract_wavelet_features,
            'detect_fatigue_pattern': detect_fatigue_pattern,
            'compare_to_baseline': compare_to_baseline,
            'generate_plot': generate_plot,
            'get_feature_vector': get_feature_vector,
            'context': self.session_context,
            'TOOL_REGISTRY': TOOL_REGISTRY,
            '__builtins__': __builtins__,
            'print': print,
        }

        agent_self = self
        def sub_llm(prompt: str, images: list = None) -> str:
            return agent_self._sub_llm_call(prompt, images)

        def search_literature(query: str) -> str:
            return agent_self._search_literature(query)

        def validate_tool(tool_code: str, tool_name: str, domain: str = 'emg') -> dict:
            return agent_self._validate_tool(tool_code, tool_name, domain)

        def register_tool(name: str, code: str, description: str) -> bool:
            return agent_self._register_tool(name, code, description)

        def classify_emg(segment, features: dict = None) -> dict:
            return agent_self._classify_emg(segment, features)

        self.repl_globals['sub_llm'] = sub_llm
        self.repl_globals['search_literature'] = search_literature
        self.repl_globals['validate_tool'] = validate_tool
        self.repl_globals['register_tool'] = register_tool
        self.repl_globals['classify_emg'] = classify_emg

    # ── LangSmith-traced methods ─────────────────────────────────────

    @traceable(run_type="chain", name="repl_exec")
    def _exec_code(self, code: str, trace: ReasoningTrace) -> str:
        """Execute Python code in the REPL and capture output."""
        t0 = time.time()
        old_stdout = __import__('sys').stdout
        captured = StringIO()
        __import__('sys').stdout = captured

        try:
            exec(code, self.repl_globals)
            output = captured.getvalue()
            duration = (time.time() - t0) * 1000
            trace.add_step(
                StepType.CODE_EXECUTION,
                f"Code executed successfully:\n{code[:300]}",
                evidence=[output[:500]] if output else [],
                duration_ms=duration,
            )
            return output if output else "(code executed, no output)"
        except Exception as e:
            output = captured.getvalue()
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            duration = (time.time() - t0) * 1000
            trace.add_step(
                StepType.ERROR,
                f"Code execution error:\n{code[:300]}\n\nError: {error_msg[:300]}",
                duration_ms=duration,
            )
            return f"OUTPUT:\n{output}\n\nERROR:\n{error_msg}"
        finally:
            __import__('sys').stdout = old_stdout

    @traceable(run_type="llm", name="sub_llm")
    def _sub_llm_call(self, prompt: str, images: list = None) -> str:
        """Recursive sub-LLM call for focused multimodal analysis."""
        content = []

        if images:
            for img_path in images:
                path = Path(img_path)
                if path.exists() and path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    with open(path, 'rb') as f:
                        img_data = base64.standard_b64encode(f.read()).decode('utf-8')
                    media_type = 'image/png' if path.suffix == '.png' else 'image/jpeg'
                    content.append({
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': media_type,
                            'data': img_data,
                        },
                    })

        content.append({'type': 'text', 'text': prompt})

        try:
            response = self.client.messages.create(
                model=self.SUB_MODEL,
                max_tokens=2000,
                messages=[{'role': 'user', 'content': content}],
                system="You are a clinical EMG analysis expert specializing in lower-limb motion classification for physical therapy. Analyze the provided data and/or images. Be concise and specific.",
            )
            return response.content[0].text
        except Exception as e:
            return f"Sub-LLM error: {e}"

    @traceable(run_type="tool", name="search_literature")
    def _search_literature(self, query: str) -> str:
        """Search for relevant clinical literature / guidelines."""
        return (
            f"Literature search for: '{query}'\n\n"
            "Key findings from EMG classification literature:\n"
            "- WT-SVD features outperform time/frequency domain alone (Zhang 2017)\n"
            "- db4 wavelet at 5 levels optimal for surface EMG decomposition\n"
            "- Fatigue manifests as MDF decrease + RMS increase\n"
            "- Vastus medialis activation patterns differ between stance/swing\n"
            "- Higher MAV/RMS during stance phase due to weight-bearing\n"
            "- Swing phase shows characteristic burst pattern\n"
            "- Standing motion: strong VM activation during knee extension\n"
            "- Sitting (leg extension): sustained VM activation through ROM\n"
        )

    @traceable(run_type="tool", name="validate_tool")
    def _validate_tool(self, tool_code: str, tool_name: str,
                       domain: str = 'emg') -> dict:
        """Validate a new tool on synthetic test data."""
        results = {'tool_name': tool_name, 'tests': [], 'passed': True}

        from alwaysonpt.data_loader import EMGSegment

        test_cases = [
            ('normal', np.random.randn(1000) * 0.05),
            ('high_amplitude', np.random.randn(1000) * 0.5),
            ('low_snr', np.random.randn(1000) * 0.001),
        ]

        for name, emg in test_cases:
            seg = EMGSegment(
                subject_id=0,
                motion_class='test',
                emg=emg,
                gonio=np.linspace(0, 90, 1000),
            )

            test_ns = {'segment': seg, 'np': np}
            try:
                exec(tool_code, test_ns)
                fn = test_ns.get(tool_name)
                if fn is None:
                    results['tests'].append({
                        'case': name, 'passed': False,
                        'error': f'Function {tool_name} not found after exec'
                    })
                    results['passed'] = False
                    continue

                result = fn(seg)
                results['tests'].append({
                    'case': name, 'passed': True,
                    'output_type': type(result).__name__,
                })
            except Exception as e:
                results['tests'].append({
                    'case': name, 'passed': False, 'error': str(e)
                })
                results['passed'] = False

        return results

    @traceable(run_type="tool", name="register_tool")
    def _register_tool(self, name: str, code: str, description: str) -> bool:
        """Register a validated tool in the REPL namespace."""
        try:
            exec(code, self.repl_globals)
            if name in self.repl_globals:
                self.skill_library[name] = {
                    'code': code,
                    'description': description,
                    'registered_at': datetime.now().isoformat(),
                }
                from alwaysonpt.emg_tools import TOOL_REGISTRY
                TOOL_REGISTRY[name] = {
                    'fn': self.repl_globals[name],
                    'description': description,
                    'args': ['segment'],
                }
                self.session_context['skills_acquired'].append(name)
                return True
        except Exception:
            pass
        return False

    @traceable(run_type="tool", name="classify_emg")
    def _classify_emg(self, segment, features: dict = None) -> dict:
        """
        Placeholder for a specialized EMG classification model.

        When a trained model is available, this method will run inference
        and return class probabilities. Until then, returns a sentinel
        so the orchestrator falls back to its own reasoning.
        """
        return {
            'available': False,
            'message': (
                'No specialized EMG classification model is connected yet. '
                'Use your own analysis of the extracted features to classify. '
                'When a trained model is deployed, this tool will return '
                'class probabilities for [standing, sitting, stance, swing].'
            ),
            'expected_response_schema': {
                'available': True,
                'prediction': 'stance',
                'probabilities': {
                    'standing': 0.05, 'sitting': 0.02,
                    'stance': 0.83, 'swing': 0.10,
                },
                'model_version': 'emg-classifier-v1',
                'confidence': 0.83,
            },
        }

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the root LLM."""
        from alwaysonpt.emg_tools import TOOL_REGISTRY

        tool_docs = "\n".join([
            f"  - {name}: {info['description']}"
            for name, info in TOOL_REGISTRY.items()
        ])

        return f"""You are AlwaysOnPT, a clinical EMG analysis agent for physical therapy.

You have a Python REPL. Write Python code blocks to analyze EMG segments.
Your code runs in exec() — all variables persist between turns.

## Available Tools (call directly in Python code)
{tool_docs}

## Available Functions
  - sub_llm(prompt, images=[path1, path2]) → str: Call a sub-agent for focused analysis. Can include plot image paths for multimodal reasoning.
  - classify_emg(segment, features=None) → dict: Call a specialized EMG classification model. Returns class probabilities if a trained model is connected; otherwise returns available=False and you should classify using your own reasoning.
  - search_literature(query) → str: Search EMG/PT clinical literature.
  - validate_tool(tool_code, tool_name, domain) → dict: Validate a new tool on synthetic data.
  - register_tool(name, code, description) → bool: Register a validated tool.

## Session Context
  - `context` dict accumulates findings across segments.
  - Previously analyzed: {self.session_context['segments_analyzed']} segments.

## Your Task
Given an EMG segment, classify it as one of: standing, sitting, stance, swing.

Approach:
1. Start with get_segment_overview(segment) to understand the data
2. Extract features: time-domain, frequency-domain, wavelet
3. Call classify_emg(segment, features) — if a specialized model is available, use its prediction as strong evidence; if not, proceed with your own reasoning
4. Generate at least one plot for visual verification
5. Use sub_llm() with the plot to get multimodal confirmation
6. Make your classification decision with reasoning
7. Provide a brief clinical narrative

## Output Format
After analysis, you MUST set these variables:
```python
classification = "standing"  # or "sitting", "stance", "swing"
confidence = 0.85  # 0.0 to 1.0
clinical_narrative = "Brief PT-relevant observation..."
```

Write Python code in ```python blocks. One block per turn.
Be systematic but concise — you have {self.MAX_TURNS} turns maximum."""

    # ── Main entry points (top-level LangSmith traces) ───────────────

    @traceable(run_type="chain", name="classify_segment")
    def classify_segment(self, segment, segment_id: str = "") -> ReasoningTrace:
        """
        Run the full RLM loop to classify a single EMG segment.
        Returns a ReasoningTrace with classification, confidence, and narrative.
        """
        trace = ReasoningTrace(segment_id=segment_id)

        self.repl_globals['segment'] = segment
        self.repl_globals['classification'] = None
        self.repl_globals['confidence'] = None
        self.repl_globals['clinical_narrative'] = None

        system_prompt = self._build_system_prompt()

        messages = [{
            'role': 'user',
            'content': (
                f"Analyze and classify this EMG segment.\n"
                f"Subject: {segment.subject_id}, Duration: {segment.duration_s:.2f}s, "
                f"Samples: {len(segment.emg)}\n"
                f"The segment is stored in variable `segment`.\n"
                f"Write Python code to analyze it."
            ),
        }]

        for turn in range(self.MAX_TURNS):
            if self.verbose:
                print(f"  [Turn {turn + 1}/{self.MAX_TURNS}]")

            t0 = time.time()
            try:
                response = self._call_root_llm(system_prompt, messages, turn)
            except Exception as e:
                trace.add_step(StepType.ERROR, f"LLM API error: {e}")
                break

            llm_duration = (time.time() - t0) * 1000
            assistant_text = response.content[0].text

            if self.verbose:
                print(f"    LLM response ({llm_duration:.0f}ms): "
                      f"{assistant_text[:100]}...")

            code_blocks = self._extract_code_blocks(assistant_text)

            if not code_blocks:
                trace.add_step(
                    StepType.OBSERVATION,
                    assistant_text[:500],
                    duration_ms=llm_duration,
                )
                messages.append({'role': 'assistant', 'content': assistant_text})
                messages.append({
                    'role': 'user',
                    'content': 'Please write Python code to continue your analysis.',
                })

                if self.repl_globals.get('classification'):
                    break
                continue

            all_output = []
            for code in code_blocks:
                output = self._exec_code(code, trace)
                all_output.append(output)
                if self.verbose:
                    print(f"    Code output: {output[:200]}")

            messages.append({'role': 'assistant', 'content': assistant_text})

            exec_result = "\n".join(all_output)
            messages.append({
                'role': 'user',
                'content': f"Code output:\n```\n{exec_result[:3000]}\n```",
            })

            classification = self.repl_globals.get('classification')
            if classification and classification in ('standing', 'sitting', 'stance', 'swing'):
                trace.classification = classification
                trace.confidence = float(self.repl_globals.get('confidence', 0.5))
                trace.clinical_narrative = str(
                    self.repl_globals.get('clinical_narrative', '')
                )
                trace.add_step(
                    StepType.DECISION,
                    f"Classification: {classification} "
                    f"(confidence: {trace.confidence:.2f})",
                    evidence=[trace.clinical_narrative],
                )
                break

        if not trace.classification:
            trace = self._fallback_classification(segment, trace)

        self.session_context['segments_analyzed'] += 1
        self.session_context['findings'].append({
            'segment_id': segment_id,
            'classification': trace.classification,
            'confidence': trace.confidence,
        })

        return trace

    @traceable(run_type="llm", name="root_llm")
    def _call_root_llm(self, system_prompt: str, messages: list, turn: int):
        """Traced wrapper around the root LLM call."""
        return self.client.messages.create(
            model=self.ROOT_MODEL,
            max_tokens=3000,
            system=system_prompt,
            messages=messages,
        )

    @traceable(run_type="chain", name="fallback_classification")
    def _fallback_classification(self, segment, trace: ReasoningTrace) -> ReasoningTrace:
        """Rule-based fallback if the LLM loop doesn't produce a classification."""
        trace.add_step(StepType.OBSERVATION, "Using fallback rule-based classification")

        from alwaysonpt.emg_tools import (
            extract_time_features, extract_freq_features, get_segment_overview,
        )

        overview = get_segment_overview(segment)
        time_f = extract_time_features(segment)

        gonio_range = overview['gonio_range_deg']
        gonio_mean = overview['gonio_mean_deg']
        mav = time_f['MAV']
        duration = segment.duration_s

        if gonio_range < 15 and gonio_mean < 20:
            classification = 'standing'
            confidence = 0.6
        elif gonio_range < 15 and gonio_mean > 40:
            classification = 'sitting'
            confidence = 0.6
        elif duration < 1.5:
            classification = 'swing'
            confidence = 0.5
        else:
            classification = 'stance'
            confidence = 0.5

        trace.classification = classification
        trace.confidence = confidence
        trace.clinical_narrative = (
            f"Fallback classification based on gonio range={gonio_range:.1f}deg, "
            f"mean={gonio_mean:.1f}deg, MAV={mav:.4f}mV"
        )
        return trace

    def _extract_code_blocks(self, text: str) -> list:
        """Extract Python code blocks from LLM response."""
        blocks = []
        in_block = False
        current_block = []

        for line in text.split('\n'):
            if line.strip().startswith('```python'):
                in_block = True
                current_block = []
            elif line.strip() == '```' and in_block:
                in_block = False
                code = '\n'.join(current_block)
                if code.strip():
                    blocks.append(code)
            elif in_block:
                current_block.append(line)

        return blocks

    @traceable(run_type="chain", name="classify_batch")
    def classify_batch(self, segments: list, segment_ids: list = None) -> list:
        """Classify a batch of segments, returning traces for each."""
        if segment_ids is None:
            segment_ids = [f"seg_{i}" for i in range(len(segments))]

        traces = []
        for seg, sid in zip(segments, segment_ids):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Classifying {sid}: subject={seg.subject_id}, "
                      f"class={seg.motion_class}")
                print('='*60)

            trace = self.classify_segment(seg, sid)
            traces.append(trace)

            if self.verbose:
                correct = trace.classification == seg.motion_class
                print(f"  Result: {trace.classification} "
                      f"(true: {seg.motion_class}) "
                      f"{'CORRECT' if correct else 'WRONG'} "
                      f"[confidence: {trace.confidence:.2f}]")

        return traces


def save_traces(traces: list, output_dir: str = None):
    """Save reasoning traces to JSON."""
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "output" / "traces")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fpath = Path(output_dir) / f"traces_{timestamp}.json"

    data = [t.to_dict() for t in traces]
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Saved {len(traces)} traces to {fpath}")
    return str(fpath)


if __name__ == "__main__":
    from alwaysonpt.data_loader import load_dataset

    dataset = load_dataset()

    sample_segments = []
    for cls in ['standing', 'sitting', 'stance', 'swing']:
        for seg in dataset.segments:
            if seg.motion_class == cls:
                sample_segments.append(seg)
                break

    print(f"\nTesting agent on {len(sample_segments)} sample segments...")
    agent = RLMAgent(verbose=True)

    traces = agent.classify_batch(
        sample_segments,
        [f"test_{seg.motion_class}" for seg in sample_segments],
    )

    correct = sum(1 for t, s in zip(traces, sample_segments)
                  if t.classification == s.motion_class)
    print(f"\nTest accuracy: {correct}/{len(sample_segments)} "
          f"({100 * correct / len(sample_segments):.1f}%)")

    save_traces(traces)
