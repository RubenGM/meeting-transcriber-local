import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.bootstrap import (
    BootstrapPaths,
    build_pip_install_command,
    should_run_setup,
    venv_python_path,
    _prepare_deepfilternet_optional,
    _runtime_env,
)


class BootstrapTests(unittest.TestCase):
    def test_venv_python_path_uses_windows_layout(self):
        paths = BootstrapPaths(project_dir=Path("app"), venv_dir=Path("app") / ".venv")

        result = venv_python_path(paths, "win32")

        self.assertEqual(result, Path("app") / ".venv" / "Scripts" / "python.exe")

    def test_venv_python_path_uses_posix_layout(self):
        paths = BootstrapPaths(project_dir=Path("app"), venv_dir=Path("app") / ".venv")

        result = venv_python_path(paths, "linux")

        self.assertEqual(result, Path("app") / ".venv" / "bin" / "python")

    def test_build_pip_install_command_installs_project_with_dependencies(self):
        command = build_pip_install_command(Path(".venv") / "bin" / "python", Path("project"))

        self.assertEqual(
            command,
            [
                str(Path(".venv") / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "wheel",
                "setuptools",
                "project",
            ],
        )

    def test_should_run_setup_when_forced_even_if_environment_is_ready(self):
        self.assertTrue(should_run_setup(["bootstrap.py", "--setup"], environment_ready=True))

    def test_should_skip_setup_when_environment_is_ready(self):
        self.assertFalse(should_run_setup(["bootstrap.py"], environment_ready=True))

    def test_should_run_setup_when_environment_is_not_ready(self):
        self.assertTrue(should_run_setup(["bootstrap.py"], environment_ready=False))

    def test_runtime_env_prefers_project_sources_over_installed_copy(self):
        paths = BootstrapPaths(project_dir=Path("/tmp/app"), venv_dir=Path("/tmp/app") / ".venv")

        result = _runtime_env(paths, base_env={})

        self.assertEqual(result["PYTHONPATH"], str(Path("/tmp/app") / "src"))

    def test_prepare_deepfilternet_optional_does_not_raise_on_failure(self):
        with patch("scripts.bootstrap.subprocess.run", return_value=Mock(returncode=1)):
            _prepare_deepfilternet_optional(Path("python"), {})


if __name__ == "__main__":
    unittest.main()
