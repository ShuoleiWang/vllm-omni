# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage bridge from complete JoyAI actions to Qwen3-TTS."""

from copy import copy
from dataclasses import replace

from vllm.outputs import RequestOutput

from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import (
    parse_action,
)
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors.aura_omni import aura2tts

_TTS_DEFAULTS = {
    "tts_task_type": "CustomVoice",
    "tts_language": "Auto",
    "tts_speaker": "Vivian",
}


def _extract_text(source_output: RequestOutput) -> str:
    output = source_output.outputs[0]
    text = getattr(output, "cumulative_text", None) or output.text
    return text if isinstance(text, str) else ""


def _tts_prompt(prompt: object) -> dict[str, object]:
    result = dict(prompt) if isinstance(prompt, dict) else {}
    raw_info = result.get("additional_information")
    additional_info = dict(raw_info) if isinstance(raw_info, dict) else {}
    for key, value in _TTS_DEFAULTS.items():
        additional_info.setdefault(key, value)
    result["additional_information"] = additional_info
    return result


def _tts_source(source_output: RequestOutput, text: str) -> RequestOutput:
    result = copy(source_output)
    result.outputs = [replace(source_output.outputs[0], text=text)]
    return result


def joyai2tts(
    source_outputs: list[RequestOutput],
    prompt: object | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Speak the response/delegate note; silence and empty notes skip TTS."""
    del requires_multimodal_data
    prompts = prompt if isinstance(prompt, list) else [prompt]
    sources: list[RequestOutput] = []
    tts_prompts: list[dict[str, object]] = []
    for index, source_output in enumerate(source_outputs):
        action = parse_action(_extract_text(source_output))
        if not action.spoke or not action.text:
            continue
        sources.append(_tts_source(source_output, action.text))
        prompt_item = prompts[index] if index < len(prompts) else None
        tts_prompts.append(_tts_prompt(prompt_item))

    if not sources:
        return []
    return aura2tts(
        sources,
        prompt=tts_prompts,
        requires_multimodal_data=False,
    )
