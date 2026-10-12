# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Replica-scoped multimodal cache keys must survive route invalidation safely."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm_omni.distributed.omni_coordinator import ReplicaInfo, ReplicaStatus
from vllm_omni.distributed.omni_coordinator.load_balancer import RoundRobinBalancer
from vllm_omni.engine.stage_pool import StagePool, StageUnavailableError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_pool(distributed: bool) -> tuple[StagePool, list[SimpleNamespace], list[ReplicaInfo]]:
    clients = [
        SimpleNamespace(
            stage_type="llm",
            client_addresses={"input_address": f"tcp://replica-{replica_id}:1000"},
            add_request_async=AsyncMock(),
        )
        for replica_id in range(2)
    ]
    pool = StagePool(0, clients, output_processor=Mock())  # type: ignore[arg-type]
    replicas = [
        ReplicaInfo(
            input_addr=client.client_addresses["input_address"],
            output_addr=f"tcp://replica-{replica_id}:1001",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=0,
            last_heartbeat=0.0,
            registered_at=0.0,
        )
        for replica_id, client in enumerate(clients)
    ]
    if distributed:
        hub = Mock()
        hub.get_replicas_for_stage.return_value = SimpleNamespace(replicas=replicas)
        pool.attach_hub(hub)
        pool.attach_load_balancer(RoundRobinBalancer())
    return pool, clients, replicas


def _make_unavailable(pool: StagePool, replicas: list[ReplicaInfo], replica_id: int) -> None:
    if pool.is_distributed:
        replicas[replica_id].status = ReplicaStatus.DOWN
    else:
        pool.clients[replica_id] = None


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("binding_invalidated", [False, True])
def test_scoped_cache_request_rejects_replica_change(distributed: bool, binding_invalidated: bool):
    async def run() -> None:
        pool, clients, replicas = _make_pool(distributed)
        assert pool.preselect_replica_id("req") == 0
        _make_unavailable(pool, replicas, 0)
        if binding_invalidated:
            pool.release_binding("req")

        with pytest.raises(StageUnavailableError, match="scoped multimodal cache keys"):
            await pool.submit_initial(
                "req",
                SimpleNamespace(sampling_params_list=[object()]),
                object(),
                expected_scoped_replica_id=0,
            )

        assert pool.get_bound_replica_id("req") is None
        pool.output_processor.add_request.assert_not_called()
        for client in clients:
            client.add_request_async.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("distributed", [False, True])
def test_ordinary_binding_reroutes_to_healthy_replica(distributed: bool):
    async def run() -> None:
        pool, clients, replicas = _make_pool(distributed)
        assert pool.preselect_replica_id("req") == 0
        _make_unavailable(pool, replicas, 0)
        request = object()

        replica_id = await pool.submit_initial("req", SimpleNamespace(sampling_params_list=[object()]), request)

        assert replica_id == 1
        assert pool.get_bound_replica_id("req") == 1
        pool.output_processor.add_request.assert_called_once()
        clients[0].add_request_async.assert_not_awaited()
        clients[1].add_request_async.assert_awaited_once_with(request)

    asyncio.run(run())


@pytest.mark.parametrize("distributed", [False, True])
def test_scoped_cache_request_reuses_healthy_replica(distributed: bool):
    async def run() -> None:
        pool, clients, _replicas = _make_pool(distributed)
        assert pool.preselect_replica_id("req") == 0
        request = object()

        replica_id = await pool.submit_initial(
            "req",
            SimpleNamespace(sampling_params_list=[object()]),
            request,
            expected_scoped_replica_id=0,
        )

        assert replica_id == 0
        assert pool.get_bound_replica_id("req") == 0
        clients[0].add_request_async.assert_awaited_once_with(request)
        clients[1].add_request_async.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("distributed", [False, True])
def test_scoped_cache_request_releases_binding_when_no_replica_is_up(distributed: bool):
    async def run() -> None:
        pool, clients, replicas = _make_pool(distributed)
        assert pool.preselect_replica_id("req") == 0
        for replica_id in range(len(clients)):
            _make_unavailable(pool, replicas, replica_id)
        pool.DISPATCH_WAIT_TIMEOUT_S = 0.0

        with pytest.raises(StageUnavailableError, match="no (UP|live) replica"):
            await pool.submit_initial(
                "req",
                SimpleNamespace(sampling_params_list=[object()]),
                object(),
                expected_scoped_replica_id=0,
            )

        assert pool.get_bound_replica_id("req") is None
        pool.output_processor.add_request.assert_not_called()
        for client in clients:
            client.add_request_async.assert_not_awaited()

    asyncio.run(run())
