from pathlib import Path
from unittest.mock import patch

from meeting_transcriber.deepfilternet import (
    DESKTOP_DEEP_FILTER_ASSETS,
    DEEPFILTERNET_VERSION,
    deep_filter_asset_name,
    deep_filter_asset_url,
    ensure_deep_filter_asset,
    ensure_deep_filter_desktop_binaries,
    local_deep_filter_asset_path,
    local_deep_filter_path,
    resolve_deep_filter_binary,
)


def test_deep_filter_asset_names_cover_desktop_platforms():
    assert deep_filter_asset_name("linux", "x86_64") == f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-unknown-linux-musl"
    assert deep_filter_asset_name("darwin", "arm64") == f"deep-filter-{DEEPFILTERNET_VERSION}-aarch64-apple-darwin"
    assert deep_filter_asset_name("darwin", "x86_64") == f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-apple-darwin"
    assert deep_filter_asset_name("win32", "AMD64") == f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-pc-windows-msvc.exe"


def test_deep_filter_asset_url_points_to_github_release():
    asset = deep_filter_asset_name("linux", "x86_64")

    assert deep_filter_asset_url(asset) == (
        f"https://github.com/Rikorose/DeepFilterNet/releases/download/v{DEEPFILTERNET_VERSION}/{asset}"
    )


def test_local_deep_filter_path_uses_platform_asset_name(tmp_path):
    assert local_deep_filter_path(tmp_path, platform="win32", machine="amd64") == (
        tmp_path / f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-pc-windows-msvc.exe"
    )


def test_local_deep_filter_asset_path_uses_release_asset_name(tmp_path):
    asset = f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-apple-darwin"

    assert local_deep_filter_asset_path(asset, tmp_path) == tmp_path / asset


def test_resolve_deep_filter_binary_prefers_environment_path(tmp_path, monkeypatch):
    binary = tmp_path / "deep-filter"
    binary.write_text("bin", encoding="utf-8")
    monkeypatch.setenv("DEEP_FILTER_BINARY", str(binary))

    assert resolve_deep_filter_binary(tmp_path) == binary


def test_resolve_deep_filter_binary_uses_path_when_local_missing(tmp_path):
    with patch("meeting_transcriber.deepfilternet.shutil.which", return_value="/usr/bin/deep-filter"):
        assert resolve_deep_filter_binary(tmp_path) == Path("/usr/bin/deep-filter")


def test_ensure_deep_filter_asset_downloads_and_marks_executable(tmp_path):
    asset = f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-apple-darwin"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"binary"

    with patch("meeting_transcriber.deepfilternet.urllib.request.urlopen", return_value=Response()) as urlopen:
        path = ensure_deep_filter_asset(asset, tmp_path)

    assert path == tmp_path / asset
    assert path.read_bytes() == b"binary"
    assert path.stat().st_mode & 0o111
    assert urlopen.call_args.args[0] == deep_filter_asset_url(asset)


def test_ensure_deep_filter_desktop_binaries_prepares_all_release_assets(tmp_path):
    with patch("meeting_transcriber.deepfilternet.ensure_deep_filter_asset") as ensure:
        ensure.side_effect = lambda asset, base_dir: base_dir / asset

        paths = ensure_deep_filter_desktop_binaries(tmp_path)

    assert paths == tuple(tmp_path / asset for asset in DESKTOP_DEEP_FILTER_ASSETS)
