import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("cobalt_runner", Path(__file__).parents[1] / "reaxkit_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        lines = ["synthetic\n", "1 ! Nr of atoms\n", "header\n", "header\n", "header\n"]
        lines += ["Co " + " ".join(["2"] * 8) + "\n"] + [" ".join(["2"] * 8) + "\n"] * 3
        lines += ["1 ! Nr of bonds\n", "header\n", "1 1 " + " ".join(["2"] * 8) + "\n", " ".join(["2"] * 8) + "\n"]
        (self.root / "ffield").write_text("".join(lines))
        (self.root / "params").write_text("2 1 1 0.1 1 3\n3 1 9 0.1 1 3\n")
        (self.root / "trainset").write_text("ENERGY\n# General Co_clusters\n2 + pair/2 - atom/1 -3\n# Volume excluded\n1 + ignored/1 - atom/1 0\nENDENERGY\n")
        (self.root / "geo").write_text("DESCRP pair\nHETATM 1 Co -1 0 0\nHETATM 2 Co 1 0 0\nEND\nDESCRP atom\nHETATM 1 Co 0 0 0\nEND\n")
        self.request = dict(contract="reaxkit-external-runner-v1", optimizer="ezff", seed=None,
                            inputs={key: str(self.root / key) for key in ("ffield", "params", "geo", "trainset")},
                            params_bounds_order="min_max", native_directory=str(self.root / "native"),
                            settings=dict(algorithm="ONEPLUSONE_SO", num_epochs=1, population_size=2, num_errors=1))
        self.config = self.root / "run.json"
        self.config.write_text(json.dumps(self.request))

    def test_inputs_and_template(self):
        request, template, bounds, initial, targets, geometries = runner.load_request(self.config)
        self.assertEqual(len(targets), 1)
        self.assertEqual(set(geometries), {"pair", "atom"})
        self.assertEqual(initial, {"p_2_1_1": 2, "p_3_1_9": 2})
        self.assertIn("<<p_3_1_9>>", template)
        self.assertEqual(bounds["p_2_1_1"], [1, 3])
        (self.root / "params").write_text("2 1 1 0.1 3 1\n")
        with self.assertRaises(ValueError):
            runner.load_request(self.config)
        self.request["params_bounds_order"] = "max_min"
        self.config.write_text(json.dumps(self.request))
        self.assertEqual(runner.load_request(self.config)[2]["p_2_1_1"], [1, 3])

    def test_rejects_unsupported_modes(self):
        for key, value in (("seed", 42), ("optimizer", "other"), ("contract", "other")):
            with self.subTest(key=key):
                changed = dict(self.request, **{key: value})
                self.config.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.load_request(self.config)

    def test_missing_geometry(self):
        (self.root / "geo").write_text("")
        with self.assertRaises(ValueError):
            runner.load_request(self.config)

    def test_invalid_parameters_and_targets(self):
        for params in ("2 1 1 0.1 1 3\n2 1 1 0.1 1 3\n", "2 1 33 0.1 1 3\n",
                       "2 1 1 0.1 nan 3\n", "2 1 1 0.1 4 5\n"):
            with self.subTest(params=params):
                (self.root / "params").write_text(params)
                with self.assertRaises(ValueError):
                    runner.load_request(self.config)
        (self.root / "trainset").write_text("ENERGY\n# General Co_clusters\n0 + pair/2 - atom/1 -3\n")
        with self.assertRaises(ValueError):
            runner.read_targets(self.root / "trainset")

    def test_engine_warning_is_not_success(self):
        def fake_run(command, cwd, **kwargs):
            kwargs["stdout"].write("WARNING: charge equilibration did not converge\n")
            (cwd / "energy.txt").write_text("EZFF_ENERGY -7\n")
        with patch.object(runner.subprocess, "run", side_effect=fake_run):
            with self.assertRaisesRegex(ValueError, "warning requires review"):
                runner.lammps_energy(self.root / "warning", [[0, 0, 0]], "lmp", 12)

    def test_objective_preserves_reports_and_fails_closed(self):
        _, template, bounds, initial, targets, geometries = runner.load_request(self.config)
        objective = runner.CobaltObjective(self.root, bounds, targets, geometries, "lmp", 10)
        with patch.object(runner, "lammps_energy", side_effect=lambda directory, positions, *args: -10 if len(positions) == 2 else -1):
            self.assertEqual(objective(initial, template), [0.5])
        report = json.loads((self.root / "evaluations/candidate-0000/report.json").read_text())
        self.assertEqual(report["items"][0]["prediction"], -4)
        self.assertEqual(report["items"][0]["weighted_squared_error"], 0.25)
        self.assertNotIn("<<", (self.root / "evaluations/candidate-0000/ffield").read_text())
        with patch.object(runner, "lammps_energy", side_effect=RuntimeError("engine failed")):
            with self.assertRaises(RuntimeError):
                objective(initial, template)
        self.assertTrue((self.root / "evaluations/candidate-0001/failure.json").exists())
        self.assertFalse((self.root / "evaluations/candidate-0001/report.json").exists())

    def test_engine_command_and_output_validation(self):
        def fake_run(command, cwd, **kwargs):
            self.assertEqual(command, ["lmp", "-in", "in.lmp"])
            self.assertTrue(kwargs["check"])
            self.assertEqual(kwargs["timeout"], 12)
            (cwd / "energy.txt").write_text("EZFF_ENERGY -7.125\n")
        with patch.object(runner.subprocess, "run", side_effect=fake_run):
            energy = runner.lammps_energy(self.root / "engine", [[-1, 0, 0]], "lmp", 12)
        self.assertEqual(energy, -7.125)
        script = (self.root / "engine/in.lmp").read_text()
        self.assertIn("units real", script)
        self.assertIn("boundary f f f", script)
        self.assertIn("qeq/reaxff", script)
        with patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("lmp", 12)):
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.lammps_energy(self.root / "timeout", [[0, 0, 0]], "lmp", 12)

    def test_check_only_does_not_create_outputs(self):
        with patch.object(runner.shutil, "which", return_value="lmp"):
            self.assertEqual(runner.main([str(self.config), "--acknowledge-cobalt-subset", "--check-only"]), 0)
        self.assertFalse((self.root / "native").exists())


if __name__ == "__main__":
    unittest.main()
