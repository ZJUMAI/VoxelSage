from types import SimpleNamespace

from Tool_Box import gpu_manager


def test_query_gpu_info_parses_csv_without_spaces(monkeypatch):
    monkeypatch.setattr(
        gpu_manager.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="0,NVIDIA RTX Test,24564,1024,23540\n",
        ),
    )

    result = gpu_manager.query_gpu_info(refresh=True)

    assert result == [
        {
            "index": 0,
            "name": "NVIDIA RTX Test",
            "memory_total_mb": 24564,
            "memory_used_mb": 1024,
            "memory_free_mb": 23540,
        }
    ]
