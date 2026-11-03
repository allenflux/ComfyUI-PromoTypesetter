"""Installer isolation and failure behavior without touching host site-packages."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("promo_setup_test", ROOT / "scripts" / "setup_ocr.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class OCRSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="promo-private-ocr-setup-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "requirements-ocr.txt").write_text("rapidocr-onnxruntime==1.4.4\n")
        (self.root / "promo_typesetter").mkdir()
        (self.root / "promo_typesetter" / "ocr_worker.py").write_text("# test worker\n")
        self.directory = self.root / ".venv-ocr"
        self.executable = setup.runtime_python(self.directory)
        self.commands = []
        self.host_env = {"PIP_CONSTRAINT": "/platform/constraints.txt",
                         "PIP_BUILD_CONSTRAINT": "/platform/build.txt",
                         "PIP_TARGET": "/host/site-packages", "PIP_PREFIX": "/host/prefix",
                         "PIP_ROOT": "/host", "PIP_PYTHON": "/host/other-python",
                         "PIP_USER": "1", "PIP_NO_DEPS": "1", "PYTHONPATH": "/host/path",
                         "PYTHONHOME": "/host/python", "HTTPS_PROXY": "https://proxy.test",
                         "PIP_INDEX_URL": "https://private-index.test/simple", "SSL_CERT_FILE": "/host/ca.pem"}
        self.response = {"status": "ok", "protocol": "1", "provider": "CPUExecutionProvider",
                         "detector_version": "1.4.4", "detection_regions": 2,
                         "versions": {"python": "3.12.11", "numpy": "2.2.6",
                                      "opencv-python": "4.12.0.88", "onnxruntime": "1.31.1",
                                      "rapidocr-onnxruntime": "1.4.4"}}

    def create(self, isolated=True):
        self.executable.parent.mkdir(parents=True, exist_ok=True)
        self.executable.write_text("# fake virtualenv interpreter\n")
        (self.directory / "pyvenv.cfg").write_text("include-system-site-packages = " + str(not isolated).lower() + "\n")

    def runner(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if command[1:4] == ["-I", "-m", "venv"]:
            self.create()
            return subprocess.CompletedProcess(command, 0, "", "")
        if "-c" in command:
            return subprocess.CompletedProcess(command, 0, json.dumps({"prefix": str(self.directory),
                                               "base_prefix": "/host/python", "user_site": False}), "")
        if command[-1] == "--self-check":
            return subprocess.CompletedProcess(command, 0, json.dumps(self.response), "")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    def run_setup(self, **kwargs):
        options = {"root": self.root, "python": "/host/comfy-python", "version_info": (3, 12, 11),
                   "runner": self.runner, "environ": self.host_env, "progress": lambda message: None}
        options.update(kwargs)
        return setup.setup_runtime(**options)

    def test_private_environment_does_not_mutate_or_reuse_host_package_settings(self):
        original = dict(self.host_env)
        env = setup.private_environment(self.directory, self.host_env)
        self.assertEqual(self.host_env, original)
        for name in ("PIP_CONSTRAINT", "PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_PYTHON", "PYTHONHOME", "PYTHONPATH"):
            self.assertNotIn(name, env)
        self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)
        self.assertEqual(env["PIP_USER"], "0")
        self.assertEqual(env["PIP_NO_DEPS"], "0")
        self.assertEqual(env["PIP_REQUIRE_VIRTUALENV"], "1")
        for name in ("PIP_BUILD_CONSTRAINT", "HTTPS_PROXY", "PIP_INDEX_URL", "SSL_CERT_FILE"):
            self.assertEqual(env[name], self.host_env[name])

    def test_new_venv_install_uses_target_interpreter_and_records_checked_runtime(self):
        result = self.run_setup()
        creation = self.commands[0][0]
        self.assertEqual(creation, ["/host/comfy-python", "-I", "-m", "venv", str(self.directory)])
        self.assertNotIn("--system-site-packages", creation)
        pip_commands = [command for command, _ in self.commands if "pip" in command]
        self.assertEqual(len(pip_commands), 2)
        for command in pip_commands:
            self.assertEqual(command[0], str(self.executable))
        install = next(command for command in pip_commands if "install" in command)
        self.assertIn("--only-binary=:all:", install)
        self.assertEqual(install[-2:], ["-r", str(self.root / "requirements-ocr.txt")])
        self.assertEqual(result["versions"]["numpy"], "2.2.6")
        marker = json.loads((self.directory / setup.MARKER).read_text())
        self.assertEqual(marker["python"], str(self.executable))
        self.assertTrue(marker["isolated"])
        self.assertFalse((self.directory / (setup.MARKER + ".tmp")).exists())
        self.assertEqual(self.host_env["PIP_CONSTRAINT"], "/platform/constraints.txt")

    def test_existing_private_environment_is_reused_without_recreating(self):
        self.create()
        self.run_setup()
        self.assertFalse(any(command[1:4] == ["-I", "-m", "venv"] for command, _ in self.commands))

    def test_unsupported_python_is_rejected_before_any_process(self):
        for version in ((3, 9), (3, 13)):
            with self.subTest(version=version), self.assertRaises(setup.SetupError) as caught:
                self.run_setup(version_info=version)
            self.assertEqual(caught.exception.code, "OCR_SETUP_PYTHON_VERSION")
        self.assertEqual(self.commands, [])

    def test_system_site_packages_or_wrong_prefix_cannot_install(self):
        self.create(isolated=False)
        with self.assertRaises(setup.SetupError) as caught:
            self.run_setup()
        self.assertEqual(caught.exception.code, "OCR_SETUP_NOT_ISOLATED")
        self.assertEqual(self.commands, [])
        self.create()

        def wrong_prefix(command, **kwargs):
            result = self.runner(command, **kwargs)
            if "-c" in command:
                result.stdout = json.dumps({"prefix": "/host/python", "base_prefix": "/host/python", "user_site": False})
            return result

        with self.assertRaises(setup.SetupError) as caught:
            self.run_setup(runner=wrong_prefix)
        self.assertEqual(caught.exception.code, "OCR_SETUP_NOT_ISOLATED")
        self.assertFalse(any("pip" in command for command, _ in self.commands))

    @unittest.skipIf(os.name == "nt", "test uses a directory symbolic link")
    def test_symlink_to_another_environment_is_rejected(self):
        foreign = self.root / "foreign"
        foreign.mkdir()
        self.directory.symlink_to(foreign, target_is_directory=True)
        with self.assertRaises(setup.SetupError) as caught:
            self.run_setup()
        self.assertEqual(caught.exception.code, "OCR_SETUP_UNSAFE_DIRECTORY")
        self.assertEqual(self.commands, [])

    def test_failed_dependency_install_removes_stale_marker_and_redacts_index_urls(self):
        self.create()
        marker = self.directory / setup.MARKER
        marker.write_text('{"old":true}')

        def failing(command, **kwargs):
            if "install" in command:
                return subprocess.CompletedProcess(command, 1, "", "Could not reach https://user:secret@index.test/simple")
            return self.runner(command, **kwargs)

        with self.assertRaises(setup.SetupError) as caught:
            self.run_setup(runner=failing)
        self.assertEqual(caught.exception.code, "OCR_SETUP_FAILED")
        self.assertNotIn("secret", json.dumps(caught.exception.as_dict()))
        self.assertNotIn("index.test", json.dumps(caught.exception.as_dict()))
        self.assertFalse(marker.exists())

    def test_failed_worker_selfcheck_never_creates_marker(self):
        for response in ({"status": "failed"}, dict(self.response, detection_regions=0),
                         dict(self.response, provider="CUDAExecutionProvider")):
            with self.subTest(response=response):
                self.response = response
                with self.assertRaises(setup.SetupError):
                    self.run_setup()
                self.assertFalse((self.directory / setup.MARKER).exists())

    def test_malformed_worker_reply_cannot_mark_install_as_successful(self):
        def malformed(command, **kwargs):
            if command[-1] == "--self-check":
                return subprocess.CompletedProcess(command, 0, "worker returned no JSON", "")
            return self.runner(command, **kwargs)

        with self.assertRaises(setup.SetupError) as caught:
            self.run_setup(runner=malformed)
        self.assertEqual(caught.exception.code, "OCR_SETUP_INVALID_RESULT")
        self.assertFalse((self.directory / setup.MARKER).exists())

    def test_worker_versions_cannot_claim_success_with_host_numpy(self):
        self.response["versions"]["numpy"] = "2.3.5"
        with self.assertRaises(setup.SetupError) as caught:
            self.run_setup()
        self.assertEqual(caught.exception.code, "OCR_SETUP_VERSION_MISMATCH")
        self.assertFalse((self.directory / setup.MARKER).exists())


if __name__ == "__main__":
    unittest.main()
