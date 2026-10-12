# JoyAI ASR PR 8031 review handoff

这份交接记录对应 2026 年 10 月 12 日完成的 JoyAI ASR reviewer 修复。代码快照和分析放在 `ShuoleiWang/vllm-omni` 的独立分支 `codex/pr-8031-review-handoff-20261012`，供另一台电脑上的 coding agent 检查和继续工作。它不是原 PR 的更新，也不代表已经具备 upstream 合并条件。

当前代码保留了原 PR 的可选 ASR 入口设计，同步了一个明确的 main 快照，并修复了预处理阻塞事件循环、错误分类和副本缓存归属问题。已有 CPU 单元测试证据有明确环境限制；H20、GPU 推理和真实音视频性能尚未验证。

## 分支基线和审阅方式

| 对象 | 修订或 tree | 用途 |
| --- | --- | --- |
| 原 upstream PR | [vllm-project/vllm-omni PR 8031](https://github.com/vllm-project/vllm-omni/pull/8031) | 只作为背景和 review 来源，不更新它 |
| 原 PR head | `5a6c21791cf2f955bef3221b6095d7159f70c46d` | `feature/joyai-vl-interaction-asr` 在交接时的远端 head |
| 集成的 main 快照 | `233f0842136b351ac057c1286d3ef3c4c4a47285` | 本次实际使用的 main，不是对未来最新 main 的承诺 |
| 同步 main 后的交接基线提交 | `2d4ff487c92924a7b04ae762c7c96ec71791ddaa` | 合并原 PR 和上述 main，保留 entry-aware prewarm 和 reference prefetch |
| Reviewer 修复提交 | `2a3fa19c2275fceea47e35da7ba753ff0faad6f7` | 只包含本轮 11 个 reviewer-fix 文件 |
| 修复前基线 tree | `3606eecaf0c30c43f96dfed66c4352acc7ce4def` | 精确记录同步后的原 PR 内容 |
| 已验证代码 tree | `f90530db910273c591556f494f048228e2c36767` | Reviewer 修复提交的完整 tree；交接文档另加在后一个提交中 |

本分支依次包含同步基线、reviewer 修复和交接文档。原先本地用于显示 diff 的同步提交是 `2b23492f7cb4e6c16ce4b64fa51ae45a1c18e87c`；这里使用内容相同、tree 相同的独立交接基线，不要求另一台电脑存在本地 Git refs 或临时文件。

审阅本轮修复时使用以下比较：

```bash
git diff --stat 2d4ff487c92924a7b04ae762c7c96ec71791ddaa 2a3fa19c2275fceea47e35da7ba753ff0faad6f7
git diff 2d4ff487c92924a7b04ae762c7c96ec71791ddaa 2a3fa19c2275fceea47e35da7ba753ff0faad6f7 -- vllm_omni/engine tests/engine
```

预期为 **11 个文件，875 行新增、191 行删除**，其中 5 个业务代码文件、6 个测试文件。与集成 main 比较完整方案时，不含本文的代码提交应为 **29 个文件，1620 行新增、230 行删除**：

```bash
git diff --stat 233f0842136b351ac057c1286d3ef3c4c4a47285 2a3fa19c2275fceea47e35da7ba753ff0faad6f7
```

直接与原 PR head `5a6c217` 比较，会看到同步 main 带来的约 370 个文件差异。这些不能全部归为本轮 reviewer 修改。新电脑 checkout 本交接分支后，工作树应是干净的；11 个文件是修复提交的 diff，不是要求新电脑保留 11 个未提交文件。

## 在另一台电脑获取代码

新 checkout：

```bash
git clone --filter=blob:none --single-branch \
  --branch codex/pr-8031-review-handoff-20261012 \
  https://github.com/ShuoleiWang/vllm-omni.git omni-pr8031-handoff
cd omni-pr8031-handoff
git status --short --branch
git log -3 --oneline
```

已有其他工作的 checkout 时，先检查本地状态；可以把这个分支 fetch 到独立 worktree，不要 reset 或 clean 原来的工作区：

```bash
git status --short --branch
git fetch --no-tags https://github.com/ShuoleiWang/vllm-omni.git \
  refs/heads/codex/pr-8031-review-handoff-20261012
git worktree add --detach ../omni-pr8031-handoff FETCH_HEAD
```

## 请求路径和当前范围

ASR profile 的阶段编号是 `ASR(0) → JoyAI(1) → Talker(2) → Code2Wav(3)`。带音频的请求从 ASR 开始；没有音频的文本、图片或视频请求直接进入 JoyAI，不生成占位音频，也不执行 ASR。

`bypass_without_modalities` 是部署配置声明的入口规则。`AsyncOmniBase.resolve_entry_stage_id()` 决定入口，frontend 使用该结果选择 chat template，并把同一个入口传给 engine。原来的 JoyAI 三阶段 profile 仍然存在。

两个 JoyAI deploy profile 当前都显式设置了 `async_chunk: false`。共享 prewarm 代码支持非零入口，并不意味着已经开启 JoyAI async chunk 或验证了该模式的性能。ASR-first 混合同步和异步边的进一步分类仍不属于这轮已验证能力。

## Review 含义和修改原因

### CPU 预处理与事件循环

[amy 的预处理 review](https://github.com/vllm-project/vllm-omni/pull/8031#discussion_r4244415202) 指出，bypass 入场直接在 orchestrator 事件循环调用 `processor.process_inputs()`。图片和视频处理、媒体 hashing、首次 processor 初始化都可能耗费 CPU 时间。在这段时间里 GPU 可以继续工作，但同一事件循环里的输出接收和转发协程得不到运行机会，从而影响其他请求的首 token、后续 token 或音频输出。

本轮通过 `_run_stage_input_processing()` 把 CPU 工作放进 `asyncio.to_thread()`。副本选择和绑定仍在事件循环执行，hashing、processor 初始化及输入处理在 worker 中执行。它释放事件循环的执行机会，没有把原来的串行请求队列改造成任意并行调度器。

只移入线程还不够，因为 bypass 和 ASR forwarding 使用同一个 stage input processor。发送端缓存可能在实际提交前就已写入：A 处理图片 X 后尚未送达，B 又处理 X 并生成缓存引用，如果 B 先送达，接收端还没有 X。因此 `_stage_input_processing()` 的每 stage 锁覆盖了输入处理到实际提交的整个区间。Forward、bypass、duplex 和管理 RPC 的缓存清理共用该保护，保证发送缓存与提交顺序一致。

主要代码是 `vllm_omni/engine/orchestrator.py` 的 `_run_stage_input_processing()`、`_stage_input_processing()`、`_forward_to_next_stage()` 和 `_handle_add_request()`，以及 `vllm_omni/engine/duplex_orchestrator.py` 的 `submit()`。

### 取消与缓存清理

取消等待 `to_thread()` 的协程不会停止已经开始的线程。代码使用 shield 并等待 worker 完成后才释放 stage 锁，避免一个仍在运行的 worker 与后续请求同时修改缓存。

如果处理完成但请求没有成功提交，发送端缓存中可能存在接收端从未得到的媒体。当前实现调用 `renderer.clear_mm_cache()` 清理整个 stage sender cache，**不是精准驱逐单条未提交媒体**。这个选择优先保证一致性，也可能降低后续请求的缓存命中率，实际性能影响需要另行测量。

本轮加锁的 entry、forward、duplex 提交路径，在获得 stage 锁之后、CPU worker 返回之后、awaited submit 返回之后检查请求状态身份。等待锁时已被取消的请求不能重新绑定副本；这些提交路径发现请求已清理时停止后续 prewarm、输出所有权记录或 request state 更新。这个保护不能概括成 orchestrator 所有路径都具有相同保证，现有 prewarm 循环内部并没有在每次 submit 后都做无条件身份检查。

### 区分输入错误与内部 bug

[amy 的错误分类 review](https://github.com/vllm-project/vllm-omni/pull/8031#discussion_r4244415207) 要求去掉 bypass 路径的宽泛 `except Exception`。原实现会把 prompt 校验错误和 processor 内部 `AttributeError` 一起映射成客户端错误。

现在 entry path 与 forwarded-input path 一样，只把 `ValueError` 和 `VLLMClientError` 作为已知输入错误隔离到单个请求，`StageUnavailableError` 单独处理。未知程序异常继续向上传播，进入现有的服务端故障路径；这可能触发 orchestrator teardown。**没有新增所有内部 bug 都能隔离成单请求 500 的保证。**

相关回归在 `tests/engine/test_orchestrator_error_handling.py`，特别是 `test_bypass_input_validation_is_distinct_from_processor_bugs`。

### 副本缓存约束与正常重路由

[amy 的副本约束 review](https://github.com/vllm-project/vllm-omni/pull/8031#discussion_r4244415209) 指出，之前的校验无差别拒绝预选副本与提交副本不一致的请求，影响了其他多副本 pipeline 原来的健康副本重选能力。

缓存请求的危险情况是：图片 X 已按 replica 0 处理并生成 `stage1:rep0:X` 这样的 key；提交前 replica 0 下线，router 改选 replica 1。发送端可能只发送 replica 0 的缓存引用，而 replica 1 从未收到 X。对此可以重新预处理，也可以明确失败；本轮选择失败，并释放绑定，不把旧引用发送给新副本，没有实现自动重新预处理或重试。

当前 downstream `_preselect_stage_mm_cache_replica()` 只在有真实媒体、`mm_processor_cache_gb > 0`、且有多个 live replicas 时预选。被处理请求保存显式 `expected_scoped_replica_id`，`StagePool.submit_initial()` 只对提供该值的请求禁止换副本，普通请求仍允许健康重路由。即使旧 affinity 已被移除，这个显式 ID 仍能识别旧缓存归属。

Stage 0 的既有 frontend 路径仍可能在缓存关闭时进行 preselection 或 UUID scoping；本轮只让新的固定副本 metadata 在启用 MM cache 时生效。不要把这项修改描述成关闭了所有 cache-disabled scoping。当前判断依据是缓存配置，而非对所有实际 cache 实现逐一探测。

涉及 `stage_pool.py`、`async_omni_engine.py`、`messages.py` 和 orchestrator request state。新增 `tests/engine/test_stage_pool_mm_cache_binding.py` 覆盖正常重路由、缓存请求拒绝换副本、绑定已失效和无可用副本等情况。

### 入口感知 prewarm 与 Code2Wav reference prefetch

[natureofnature 的同步 main 要求](https://github.com/vllm-project/vllm-omni/pull/8031#issuecomment-6116877860) 要求同时保留本 PR 的 entry-stage-aware prewarm 和已合入 main 的 [PR 8198](https://github.com/vllm-project/vllm-omni/pull/8198) reference-prefetch 逻辑。两者都修改共享的 `_prewarm_async_chunk_stages()`，所以冲突解决必须叠加两项行为。

Prewarm 先给接收 async chunk 的下游 scheduler 注册占位请求，让真实 chunk 到达时请求已经存在。循环从 `entry_stage_id + 1` 开始，并跳过不接收 chunk 的 stage；streaming update 使用保存的入口，而不是写死 stage 0。入口已经是最终阶段时不需要 prewarm。

PR 8198 是 MiniCPM-o Code2Wav 已有优化：在真实首个 chunk 到来之前，提前交付 reference audio 并准备参考缓存。占位输入会清掉 `multi_modal_data`，所以 hook 必须读取原始 `req_state.prompt`，不能读取已清理的占位副本。Payload 仅在初次 admission、非 session-owned、非 resumable 时附加；streaming update 和 duplex 沿用默认不附加行为。缺少或无效的 hook payload 仍可退回普通占位路径。

这项保留工作不应被描述成本轮新创造的 JoyAI GPU 性能优化。对应测试在 `tests/engine/test_orchestrator_async_chunk_prewarm.py` 及下述 reference-prefetch 分组。

### 当前 PR 已有的早期修复

[chickeyton 的抽象边界 review](https://github.com/vllm-project/vllm-omni/pull/8031#discussion_r4225982192) 希望入口决策在 engine 内。当前 PR 已把决策放入 `AsyncOmniBase.resolve_entry_stage_id()`，但仍显式传递入口结果，因为 frontend 在 `generate()` 前就需要为该 stage 渲染 prompt。这是当前取舍，不代表 reviewer 已明确认可全部参数设计。

[较早的 LoRA review](https://github.com/vllm-project/vllm-omni/pull/8031#pullrequestreview-5450762480) 指出，跳过 stage 0 时 adapter 可能丢失。原 PR head 已携带 `lora_request` 到真实 entry-stage processor，本轮保留此行为。

## 相关 issue 的边界

| Issue | 讨论内容 | 对当前分支的影响 |
| --- | --- | --- |
| [8803](https://github.com/vllm-project/vllm-omni/issues/8803) | JoyAI unified duplex session，例如持续看视频、播放中接收新输入、管理工具和记忆 | 是后续 RFC，ASR 使用本 PR 的可选 profile；没有在这里实现完整 duplex migration、摘要或工具生命周期 |
| [8580](https://github.com/vllm-project/vllm-omni/issues/8580) | Stage Output Gate，输出产生后决定继续、停止及客户端事件 | 与 admission 时选择入口是不同决策；当前没有加入 proposed gate API，也没有承诺撤回已经发出的音频 |
| [8740](https://github.com/vllm-project/vllm-omni/issues/8740) | 配置固定的非顺序 stage chain，并统一路由、prewarm、完成与清理 | 当前保持已有线性流水线与 leading bypass，没有实现任意 `StageRouting` |
| [8604](https://github.com/vllm-project/vllm-omni/issues/8604) | Bridge 客户端错误导致整个 orchestrator 退出 | 已关闭，main 已包含相关修复；本轮沿用已知输入错误隔离、内部 bug 不掩盖的契约 |
| [5286](https://github.com/vllm-project/vllm-omni/issues/5286) | JoyAI native ASR 后续设计背景 | 提供可选 ASR 和无音频绕过的动机，不是本分支实现全部实时交互能力的证明 |

[8803 的后续评论](https://github.com/vllm-project/vllm-omni/issues/8803#issuecomment-6117830033) 也要求先稳定工具生命周期、history 和 stage-output contract，再确定 JoyAI 实现范围。这些 RFC 的状态可能继续变化，后续工作需重新读当时的讨论。

## CPU 验证结果和限制

本次记录有 **658 个唯一 CPU 测试通过、1 skipped**，这是多次局部运行去重后的统计，不是一次全部通过的完整套件：

| 分组 | 唯一通过数 | 证据说明 |
| --- | --- | --- |
| Config | 285，另有 1 skipped | 初轮 273 通过，12 个本地 localhost 权限失败后重跑通过 |
| AsyncOmni input 与 JoyAI processor | 28 | 独立运行通过 |
| Core、prewarm、StagePool、duplex、lifecycle | 220 | 初轮 215，加两处 fixture 修复后通过 2 个，再加 3 个新 lifecycle case |
| Scheduler、MiniCPM payload、Code2Wav reference prefetch | 125 | 独立运行通过 |

交接前重新执行了最后 37 个定向用例，结果 **37 passed, 3 warnings in 1.34s**。这 37 个都已经计入 658：其中 34 个在更早运行中通过，3 个是后续新增的 lifecycle case。这次重跑不再累加。

以下七个 case 在本机没有通过，仍需合适环境验证：

- `test_frontend_log_stats_flag_is_not_an_unowned_stage_argument[True]` 和 `[False]`，以及 `test_mps_stays_in_runtime_instead_of_engine_arguments`：Mac 平台没有相应 worker-class 实现。
- `test_disabled_stage_metrics_drain_native_cache_on_completion`：本地 vLLM 的 `make_request_output(error=...)` 签名与 Omni 不兼容。
- `test_duplex_import_boundary.py` 的三个子进程 case：子进程不继承下面的临时启动 sentinel，仍会因缺失 vLLM startup API 而 import 失败。

Serving、modality、multistage、sampling 套件另外因缺少 `av` 和启动 API 兼容性问题未完整执行。它们不包含在 658 通过数里，不能写成 serving 验证已完成。

Ruff check、Ruff format check、AST 解析和 diff whitespace 检查已通过。没有声称跑过全部 pre-commit hooks、GPU worker、完整 CLI startup、模型加载或 E2E 服务测试。

### 本机单元测试环境

本机是 macOS arm64、Python 3.12.12、torch 2.11.0，CUDA 不可用；transformers 5.14.1、pytest 9.1.1、pytest-asyncio 1.4.0。实际 editable vLLM 源码修订为 `5a5b7eb07705d28a5576430326290dbfef9b73a0`。安装包元数据的版本字符串已经陈旧，不能据此声称环境匹配 vLLM 0.31。

临时依赖 overlay 包含 aenum 3.1.16、janus 2.0.0、omegaconf 2.4.0、PyYAML 6.0.3、diffusers 0.40.0、pytest-mock 3.15.1、soundfile 0.14.0、prettytable 3.18.0、ruff 0.17.0。没有升级共享 vLLM 环境。

本机缺少 `vllm.v1.engine.utils.get_engine_zmq_addresses`。为了让 fake-stage CPU tests 完成 import，测试启动脚本只为这个缺失 API 加了一个**一旦调用就抛出 RuntimeError 的 sentinel**，并先运行仓库原有的 custom-op bootstrap。它没有模拟 engine startup，也没有替换被测的缓存、路由或错误处理行为。正常 Linux/CUDA runtime 应首先直接运行 pytest；不要把 sentinel 用到 CLI server、GPU、模型加载或性能验证中。

### 可复现的测试命令

在版本匹配的环境中，优先使用以下 wrapper。`TEST_PYTHON` 指向实际测试解释器，`OMNI_REPO` 指向当前 checkout；不要复制其他机器的路径：

```bash
export OMNI_REPO="$PWD"
export TEST_PYTHON=python
run_cpu() {
  PYTHONDONTWRITEBYTECODE=1 "$TEST_PYTHON" -m pytest \
    -q -p no:cacheprovider -o addopts= "$@"
}
```

Config、input 和 processor 分组：

```bash
run_cpu \
  tests/config/test_omni_config.py \
  tests/engine/test_async_omni_engine_input.py \
  tests/model_executor/stage_input_processors/test_joyai_vl_interaction.py
```

Core 分组的十个模块如下。不要用排除失败 case 的结果声称整个分组正常通过：

```bash
run_cpu \
  tests/engine/test_orchestrator.py \
  tests/engine/test_orchestrator_error_handling.py \
  tests/engine/test_orchestrator_stage_input_bridge.py \
  tests/engine/test_orchestrator_async_chunk_prewarm.py \
  tests/engine/test_async_chunk_prewarm_wire.py \
  tests/engine/test_stage_metadata_async_chunk_prewarm.py \
  tests/engine/test_stage_pool_mm_cache_binding.py \
  tests/engine/test_stage_pool_collective_rpc.py \
  tests/engine/test_duplex_orchestrator.py \
  tests/engine/test_duplex_import_boundary.py
```

交接前重跑的 37 个 case：

```bash
run_cpu \
  tests/engine/test_orchestrator.py::test_abort_while_waiting_for_processor_does_not_recreate_replica_binding \
  tests/engine/test_orchestrator.py::test_cleanup_during_entry_submit_does_not_prewarm_orphaned_request \
  tests/engine/test_orchestrator.py::test_aborted_preprocessing_cannot_seed_an_unsent_cache_hit \
  tests/engine/test_orchestrator.py::test_bypass_and_forwarded_processing_wait_for_prior_submit \
  tests/engine/test_orchestrator.py::test_bypass_preprocessing_keeps_stage_outputs_responsive \
  tests/engine/test_orchestrator_async_chunk_prewarm.py
```

Reference-prefetch 分组：

```bash
run_cpu \
  tests/core/sched/test_omni_scheduler_async_chunk_prewarm.py \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_prewarm_payload.py \
  tests/model_executor/models/minicpmo_4_5/test_code2wav_ref_prefetch.py
```

### 仅用于复现本机 import 限制的启动脚本

如果需要复现上述 Mac fake-stage 结果，可以把下面脚本保存到临时目录，通过 `TEST_BOOTSTRAP` 指向它。这是测试环境的明确例外，不是 production 修复；记录使用它的每次运行。它不会自动影响子进程。

```python
from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

bootstrap_vllm_layer_custom_op_modules()

from vllm.v1.engine import utils as engine_utils


def unsupported_startup(*args, **kwargs):
    raise RuntimeError(
        "CPU import sentinel called: this environment does not support real engine startup"
    )


if not hasattr(engine_utils, "get_engine_zmq_addresses"):
    engine_utils.get_engine_zmq_addresses = unsupported_startup

import sys
import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
```

相应 wrapper 为：

```bash
run_cpu() {
  env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$OMNI_TEST_DEPS:$OMNI_REPO" \
    "$TEST_PYTHON" "$TEST_BOOTSTRAP" \
    -q -p no:cacheprovider -o addopts= "$@"
}
```

初轮 config/input/processor 运行是 `301 passed, 1 skipped, 15 failed`，其中 28 个 input/processor case 不能再次计数。下面四个单 case 加一个八参数 case 的本地 localhost 权限重跑为 `12 passed`，其余三个 Mac worker-class case 仍受限：

```bash
run_cpu \
  tests/config/test_omni_config.py::test_diffusion_config_none_values_preserve_dataclass_defaults \
  tests/config/test_omni_config.py::test_from_pipeline_config_normalizes_diffusion_config_aliases_from_engine_args \
  tests/config/test_omni_config.py::test_diffusion_stage_payload_keys_roundtrip \
  tests/config/test_omni_config.py::test_omni_diffusion_config_normalizes_video_output_transport_mapping \
  tests/config/test_omni_config.py::test_direct_diffusion_aliases_promote_none_and_reject_real_conflicts
```

## 后续工作和授权边界

下一台 agent 应先确认修订和 diff，再在匹配的 runtime 上重跑相关 CPU 与 serving 用例，记录真实结果。重点继续审阅缓存清理成本、异常分类、processor 共享、cancel/submit 竞态，以及多副本缓存实现的边界。

**用户尚未批准 H20 执行。**本轮没有连接 H20，也没有在 H20 运行命令。本文是交接上下文，不能替代用户对后续远端操作的授权；不要依据旧 session 中的机器名、容器状态或 GPU 占用做假设。

得到明确批准后，H20 验证可以按以下顺序提出具体执行计划：先只读识别 host/container、runtime 版本、模型缓存、空间、GPU 占用和端口；使用隔离 source tree；再在空闲 GPU 上串行验证原 JoyAI profile 与 ASR profile。应覆盖音频和无音频、streaming、混合并发、未缓存的大视觉输入、取消，以及首 token/连续输出/事件循环响应。冷启动成本和稳态处理成本需要分别记录。

不要把 CPU fake-stage 测试、原 PR 描述中旧版 H20 结果或 PR 8198 作者的其他硬件数据当成本分支的 GPU 性能证据。不要自动升级 runtime、重启容器或停止其他人的 GPU 进程。

本次发布范围只包括这个新的 fork 分支。没有授权更新原 `feature/joyai-vl-interaction-asr`、创建或修改 PR、发表评论、resolve review thread，或覆盖其他远端分支。之后若需要这些操作，应以用户新的明确指令为准。

## AI assistance

Codex 实现了本轮 reviewer 修复，使用独立子 agent 检查缓存、取消和 prewarm 边界，并执行上述有环境限制的 CPU 验证。交接文档由 Codex 整理。本文没有声称人类已经完成最终代码审阅，也没有声称本分支已经通过完整服务或 GPU 验证。
