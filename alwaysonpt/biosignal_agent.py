"""
BioSignalAgent — generalized RLM agent for any physiological signal.
Task-type prompt architecture: the agent loop is general, the prompt specializes.
"""

import os
import json
import time
import base64
import inspect
import traceback
import numpy as np
from io import StringIO
from pathlib import Path

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import anthropic
import langsmith
from langsmith import traceable
from datetime import datetime

from alwaysonpt.rlm_agent import ReasoningTrace, StepType, save_traces
from alwaysonpt.datasets.base import BioSignalRecord
from alwaysonpt.task_prompts import get_task_config, TASK_REGISTRY
from alwaysonpt.signal_tools import SIGNAL_TOOL_REGISTRY


class BioSignalAgent:
    """
    Generalized recursive LLM agent for any biosignal domain.

    Same architecture as RLMAgent (exec() REPL, sub_llm, skill acquisition),
    but parameterized by task_type which selects the prompt template.
    """

    ROOT_MODEL = "claude-sonnet-4-6"
    SUB_MODEL = "claude-sonnet-4-6"
    MAX_TURNS = 6
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
            'records_analyzed': 0,
        }
        self._sub_llm_count = 0

    def _init_repl(self, record: BioSignalRecord, domain_hint: str = None):
        """Initialize the REPL namespace with signal tools and record data."""
        from alwaysonpt.signal_tools import (
            compute_statistics, compute_psd, compute_wavelet,
            compute_variability, compute_cross_correlation,
            compute_symmetry, detect_peaks, segment_signal,
            generate_signal_plot,
        )

        tool_fns = {
            'compute_statistics': compute_statistics,
            'compute_psd': compute_psd,
            'compute_wavelet': compute_wavelet,
            'compute_variability': compute_variability,
            'compute_cross_correlation': compute_cross_correlation,
            'compute_symmetry': compute_symmetry,
            'detect_peaks': detect_peaks,
            'segment_signal': segment_signal,
            'generate_signal_plot': generate_signal_plot,
        }

        self.repl_globals = {
            'np': np,
            'record': record,
            'signals': record.signals,
            'context': self.session_context,
            '__builtins__': __builtins__,
            'print': print,
        }
        self.repl_globals.update(tool_fns)

        if domain_hint == 'emg':
            try:
                from alwaysonpt.data_loader import EMGSegment
                from alwaysonpt.emg_tools import (
                    get_segment_overview, extract_time_features,
                    extract_freq_features, extract_wavelet_features,
                    detect_fatigue_pattern, compare_to_baseline,
                    generate_plot, get_feature_vector,
                )
                emg_arr = record.signals.get('emg', np.array([]))
                gonio_arr = record.signals.get('gonio', np.zeros_like(emg_arr))
                sid = record.metadata.get('subject_id', 0)
                try:
                    sid = int(sid)
                except (ValueError, TypeError):
                    sid = 0
                segment = EMGSegment(
                    subject_id=sid,
                    motion_class=record.metadata.get('motion_class',
                                  record.metadata.get('exercise', 'unknown')),
                    emg=emg_arr,
                    gonio=gonio_arr,
                    fs=record.fs,
                )
                emg_fns = {
                    'get_segment_overview': get_segment_overview,
                    'extract_time_features': extract_time_features,
                    'extract_freq_features': extract_freq_features,
                    'extract_wavelet_features': extract_wavelet_features,
                    'detect_fatigue_pattern': detect_fatigue_pattern,
                    'compare_to_baseline': compare_to_baseline,
                    'generate_plot': generate_plot,
                    'get_feature_vector': get_feature_vector,
                }
                self.repl_globals['segment'] = segment
                self.repl_globals.update(emg_fns)
                tool_fns.update(emg_fns)
            except ImportError:
                pass

        tools_str = self._build_tool_signatures(tool_fns)
        all_fns = dict(tool_fns)

        agent_self = self

        def sub_llm(prompt: str, images: list = None) -> str:
            """Call a sub-agent for focused analysis. Can include plot image paths."""
            return agent_self._sub_llm_call(prompt, images)

        all_fns['sub_llm'] = sub_llm

        def help_tool(name: str) -> str:
            """Return the full docstring and signature for any available tool."""
            fn = all_fns.get(name)
            if fn is None:
                return f"Unknown tool: '{name}'. Use print(TOOLS) to see available tools."
            sig = inspect.signature(fn)
            doc = inspect.getdoc(fn) or "No documentation."
            return f"{name}{sig}\n\n{doc}"

        tools_str += f"\n\n  sub_llm(prompt: str, images: list = None) -> str  — Call a sub-agent for focused analysis"
        tools_str += f"\n  help_tool(name: str) -> str  — Get full docstring for any tool"

        self.repl_globals['sub_llm'] = sub_llm
        self.repl_globals['help_tool'] = help_tool
        self.repl_globals['TOOLS'] = tools_str

    @traceable(run_type="chain", name="biosignal_analyze")
    def analyze(self, record: BioSignalRecord, task_type: str,
                question: str = None, record_id: str = "",
                on_step=None) -> ReasoningTrace:
        """
        Run the full RLM loop on a BioSignalRecord.

        Args:
            record: any BioSignalRecord
            task_type: key into TASK_REGISTRY for prompt selection
            question: optional free-form question (for 'open_analysis')
            record_id: identifier for the trace
            on_step: optional callback(step, trace) called after each step
        """
        task_config = get_task_config(task_type)
        trace = ReasoningTrace(segment_id=record_id or record.record_id,
                               _on_step=on_step)
        self._sub_llm_count = 0

        self._init_repl(record, task_config.get('domain_hint'))

        for var in task_config['output_vars']:
            self.repl_globals[var] = None

        system_prompt = self._build_system_prompt(task_config, record, question)
        user_msg = self._build_user_message(record, task_type, question)

        messages = [{'role': 'user', 'content': user_msg}]

        for turn in range(self.MAX_TURNS):
            if self.verbose:
                print(f"  [Turn {turn + 1}/{self.MAX_TURNS}]")

            t0 = time.time()
            try:
                response = self.client.messages.create(
                    model=self.ROOT_MODEL,
                    max_tokens=1500,
                    system=system_prompt,
                    messages=messages,
                )
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
                    StepType.OBSERVATION, assistant_text,
                    duration_ms=llm_duration,
                )
                messages.append({'role': 'assistant', 'content': assistant_text})
                messages.append({
                    'role': 'user',
                    'content': 'Please write Python code to continue your analysis.',
                })
                if self._check_completion(task_config):
                    break
                continue

            code = code_blocks[0]
            output = self._exec_code(code, trace)
            if self.verbose:
                print(f"    Code output: {output[:200]}")

            messages.append({'role': 'assistant', 'content': assistant_text})
            messages.append({
                'role': 'user',
                'content': f"Code output:\n```\n{output[:3000]}\n```",
            })

            if self._check_completion(task_config):
                self._extract_results(trace, task_config)
                break

        if not trace.classification and not any(
            self.repl_globals.get(v) is not None
            for v in task_config['output_vars']
        ):
            trace.add_step(StepType.OBSERVATION,
                           "Agent did not produce a result within turn limit")
            trace.classification = "inconclusive"
            trace.confidence = 0.0
            trace.clinical_narrative = "Analysis did not converge within turn limit."

        self.session_context['records_analyzed'] += 1
        self.session_context['findings'].append({
            'record_id': record.record_id,
            'classification': trace.classification,
            'confidence': trace.confidence,
        })

        return trace

    @traceable(run_type="chain", name="classify_segment")
    def classify_segment(self, segment, segment_id: str = "") -> ReasoningTrace:
        """Backward-compatible wrapper for existing EMG eval harness."""
        record = BioSignalRecord.from_emg_segment(segment)
        trace = self.analyze(record, task_type='emg_classification',
                             record_id=segment_id)
        return trace

    def _check_completion(self, task_config: dict) -> bool:
        """Check if all required output variables have been set."""
        output_vars = task_config['output_vars']
        required = [v for v in output_vars if v != 'clinical_narrative']
        return all(self.repl_globals.get(v) is not None for v in required)

    def _extract_results(self, trace: ReasoningTrace, task_config: dict):
        """Pull results from REPL namespace into the trace."""
        output_vars = task_config['output_vars']

        primary_var = output_vars[0] if output_vars else None
        primary_val = self.repl_globals.get(primary_var, '')
        trace.classification = str(primary_val) if primary_val else ''

        trace.confidence = float(self.repl_globals.get('confidence', 0.5))
        trace.clinical_narrative = str(
            self.repl_globals.get('clinical_narrative', '')
        )

        trace.add_step(
            StepType.DECISION,
            f"{primary_var}={primary_val} (confidence: {trace.confidence:.2f})",
            evidence=[trace.clinical_narrative],
        )

    def _build_tool_signatures(self, tool_fns: dict) -> str:
        """Build a human-readable string of all tool signatures from actual functions."""
        lines = []
        for name, fn in tool_fns.items():
            try:
                sig = inspect.signature(fn)
                doc_first = (inspect.getdoc(fn) or '').split('\n')[0]
                lines.append(f"  {name}{sig}  — {doc_first}")
            except (ValueError, TypeError):
                lines.append(f"  {name}(...)  — (signature unavailable)")
        return "\n".join(lines)

    def _build_system_prompt(self, task_config: dict,
                              record: BioSignalRecord,
                              question: str = None) -> str:
        """Build the system prompt from task config."""
        prompt_template = task_config['system_prompt']
        return prompt_template.format(
            max_turns=self.MAX_TURNS,
            question=question or '',
        )

    def _build_user_message(self, record: BioSignalRecord,
                             task_type: str, question: str = None) -> str:
        """Build the initial user message describing the data."""
        channels = ", ".join(
            f"{k} ({len(v)} samples)" for k, v in record.signals.items()
        )
        msg = (
            f"Analyze this {record.domain} recording.\n"
            f"Record ID: {record.record_id}\n"
            f"Domain: {record.domain}, Sampling rate: {record.fs}Hz, "
            f"Duration: {record.duration_s:.1f}s\n"
            f"Available channels: {channels}\n"
        )
        if record.metadata:
            safe_meta = {k: v for k, v in record.metadata.items()
                         if not isinstance(v, (np.ndarray,))}
            msg += f"Metadata: {json.dumps(safe_meta, default=str)}\n"

        msg += (
            f"\nThe record is stored in variable `record`, "
            f"and signal arrays in `signals` dict.\n"
            f"Write Python code to analyze it."
        )

        if question:
            msg += f"\n\nSpecific question: {question}"

        return msg

    @traceable(run_type="chain", name="repl_exec")
    def _exec_code(self, code: str, trace: ReasoningTrace) -> str:
        """Execute Python code in the REPL and capture output."""
        t0 = time.time()
        import sys
        old_stdout = sys.stdout
        captured = StringIO()
        sys.stdout = captured

        try:
            exec(code, self.repl_globals)
            output = captured.getvalue()
            duration = (time.time() - t0) * 1000
            trace.add_step(
                StepType.CODE_EXECUTION,
                code,
                evidence=[output] if output else [],
                duration_ms=duration,
            )
            return output if output else "(code executed, no output)"
        except Exception as e:
            output = captured.getvalue()
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            duration = (time.time() - t0) * 1000
            trace.add_step(
                StepType.ERROR,
                code,
                evidence=[error_msg],
                duration_ms=duration,
            )
            return f"OUTPUT:\n{output}\n\nERROR:\n{error_msg}"
        finally:
            sys.stdout = old_stdout

    @traceable(run_type="llm", name="sub_llm")
    def _sub_llm_call(self, prompt: str, images: list = None) -> str:
        """Recursive sub-LLM call for focused analysis."""
        if self._sub_llm_count >= self.MAX_SUB_LLM_CALLS:
            return "Sub-LLM call limit reached. Proceed with available information."

        self._sub_llm_count += 1
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
                system=(
                    "You are a biomedical signal analysis expert. "
                    "Analyze the provided data and/or images. "
                    "Be concise and specific. Ground conclusions in evidence."
                ),
            )
            return response.content[0].text
        except Exception as e:
            return f"Sub-LLM error: {e}"

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
