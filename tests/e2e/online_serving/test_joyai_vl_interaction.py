# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end coverage for the native JoyAI all-sync speech pipeline."""

import base64
import os
from io import BytesIO

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import (
    _merge_base64_audio_to_segment,
    convert_audio_bytes_to_text,
    generate_synthetic_image,
)
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import (
    Action,
    parse_action,
)
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    USER_QUERY_HEADER,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_MODEL = "jdopensource/JoyAI-VL-Interaction-Preview"
_SERVER = OmniServerParams(
    model=_MODEL,
    stage_config_path=get_deploy_config_path("joyai_vl_interaction.yaml"),
    server_args=["--trust-remote-code", "--no-async-chunk"],
)


def _messages(image_url: str, instruction: str) -> list[dict]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{USER_QUERY_HEADER}\n{instruction}"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]


def _run_stream(
    openai_client,
    model: str,
    image_url: str,
    instruction: str,
) -> tuple[str, list[str], list[str]]:
    stream = openai_client.client.chat.completions.create(
        model=model,
        messages=_messages(image_url, instruction),
        modalities=["text", "audio"],
        stream=True,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
    )
    text = ""
    audio_parts: list[str] = []
    events: list[str] = []
    for chunk in stream:
        modality = getattr(chunk, "modality", None)
        for choice in chunk.choices:
            content = getattr(choice.delta, "content", None)
            if not content:
                continue
            if modality == "audio":
                base64.b64decode(content, validate=True)
                audio_parts.append(content)
                events.append("audio")
            elif modality == "text":
                text += content
                if "action" not in events and any(
                    marker in text for marker in ("</response>", "</silence>")
                ):
                    events.append("action")
    return text, audio_parts, events


def _assert_spoken(audio: list[str], events: list[str]) -> bytes:
    assert audio and all(audio)
    assert events.index("action") < events.index("audio")
    segment = _merge_base64_audio_to_segment(audio)
    assert segment.sample_rate == 24000
    assert segment.data.ndim == 1
    assert segment.data.size > 0
    wav = BytesIO()
    segment.export(wav, format="wav")
    return wav.getvalue()


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [_SERVER], indirect=True)
def test_action_controls_native_all_sync_speech(
    omni_server,
    openai_client,
) -> None:
    image = generate_synthetic_image(224, 224, seed=0)
    image_url = f"data:image/jpeg;base64,{image['base64']}"

    response_text, response_audio, response_events = _run_stream(
        openai_client,
        omni_server.model,
        image_url,
        "Output exactly: </response> Colored squares are visible.",
    )
    response = parse_action(response_text)
    assert response.action is Action.RESPONSE
    assert response.text == "Colored squares are visible."
    _assert_spoken(response_audio, response_events)

    silence_text, silence_audio, silence_events = _run_stream(
        openai_client,
        omni_server.model,
        image_url,
        "Output exactly: </silence>",
    )
    silence = parse_action(silence_text)
    assert silence.action is Action.SILENCE
    assert "action" in silence_events
    assert not silence_audio

    delegate_text, delegate_audio, delegate_events = _run_stream(
        openai_client,
        omni_server.model,
        image_url,
        (
            "Output exactly: </response> I am checking this request now. "
            "</delegation> Why is the purple elephant dancing beside a volcano?"
        ),
    )
    delegate = parse_action(delegate_text)
    assert delegate.action is Action.DELEGATE
    assert delegate.text == "I am checking this request now."
    assert delegate.delegated_question == (
        "Why is the purple elephant dancing beside a volcano?"
    )
    delegate_wav = _assert_spoken(delegate_audio, delegate_events)
    transcript = convert_audio_bytes_to_text(delegate_wav).lower()
    assert "check" in transcript
    assert all(word not in transcript for word in ("purple", "elephant", "volcano"))
