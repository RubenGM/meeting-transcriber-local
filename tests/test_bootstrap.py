import unittest
from pathlib import Path

from scripts.bootstrap import (
    BootstrapPaths,
    build_pip_install_command,
    should_run_setup,
    venv_python_path,
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


if __name__ == "__main__":
    unittest.main()
