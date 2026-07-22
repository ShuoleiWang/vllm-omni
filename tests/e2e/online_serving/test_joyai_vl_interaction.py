# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end coverage for the native JoyAI speech pipeline."""

import base64
import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    USER_QUERY_HEADER,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_MODEL = "jdopensource/JoyAI-VL-Interaction-Preview"
_SERVER = OmniServerParams(
    model=_MODEL,
    stage_config_path=get_deploy_config_path("joyai_vl_interaction.yaml"),
    server_args=["--trust-remote-code"],
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
    openai_client, model: str, image_url: str, instruction: str
) -> tuple[str, list[bytes], list[str]]:
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
    audio_parts: list[bytes] = []
    events: list[str] = []
    for chunk in stream:
        modality = getattr(chunk, "modality", None)
        for choice in chunk.choices:
            content = getattr(choice.delta, "content", None)
            if not content:
                continue
            if modality == "audio":
                audio_parts.append(base64.b64decode(content, validate=True))
                events.append("audio")
            elif modality == "text":
                text += content
                if "action" not in events and any(
                    marker in text for marker in ("</response>", "</silence>")
                ):
                    events.append("action")
    return text, audio_parts, events


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [_SERVER], indirect=True)
def test_action_controls_native_multistage_speech(omni_server, openai_client) -> None:
    image = generate_synthetic_image(224, 224, seed=0)
    image_url = f"data:image/jpeg;base64,{image['base64']}"

    response_text, response_audio, response_events = _run_stream(
        openai_client,
        omni_server.model,
        image_url,
        "Output exactly: </response> Colored squares are visible.",
    )
    assert "</response>" in response_text
    assert response_audio and all(response_audio)
    assert response_events.index("action") < response_events.index("audio")

    silence_text, silence_audio, _ = _run_stream(
        openai_client,
        omni_server.model,
        image_url,
        "Output exactly: </silence>",
    )
    assert "</silence>" in silence_text
    assert not silence_audio
