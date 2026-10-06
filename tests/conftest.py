import pytest

from imagegen_mcp import devices


@pytest.fixture(autouse=True)
def _no_host_gpu_uuids(monkeypatch):
    """Device plans pin GPUs by UUID from nvidia-smi; tests must not depend on the machine's GPUs."""
    monkeypatch.setattr(devices, "gpu_uuids", lambda: {})
