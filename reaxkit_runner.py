"""Serial cobalt-subset LAMMPS smoke runner for ReaxKit's file contract."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess


GROUPS = {"Co_clusters", "Co2_cluster"}
OBJECTIVE = "sqrt_sum_squared_accuracy_divided_energy_residuals"


def number(value):
    result = float(str(value).replace("D", "E").replace("d", "e"))
    if not math.isfinite(result):
        raise ValueError("Nonfinite numeric input")
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_targets(path):
    targets = []
    active = False
    energy = False
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if line == "ENERGY":
            energy = True
        elif line == "ENDENERGY":
            energy = active = False
        elif line.startswith("#"):
            active = line in {"# General " + group for group in GROUPS}
        elif active and energy and line:
            fields = line.split()
            if len(fields) != 6 or fields[1] != "+" or fields[3] != "-":
                raise ValueError(f"Unsupported selected energy target at line {line_number}")
            first, first_scale = fields[2].rsplit("/", 1)
            second, second_scale = fields[4].rsplit("/", 1)
            tolerance, first_scale, second_scale, target = map(number, (fields[0], first_scale, second_scale, fields[5]))
            if min(tolerance, first_scale, second_scale) <= 0:
                raise ValueError("Target divisors and tolerance must be positive")
            targets.append(dict(line=line_number, first=first, second=second,
                                first_divisor=first_scale, second_divisor=second_scale,
                                tolerance=tolerance, reference=target, expression=f"{fields[2]} - {fields[4]}"))
    if not targets:
        raise ValueError("No Co_clusters/Co2_cluster energy targets found")
    return targets


def read_geometries(path, required):
    geometries = {}
    name, positions = None, []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "DESCRP":
            name = " ".join(fields[1:])
            positions = []
        elif name in required and fields[0] == "HETATM":
            if len(fields) < 6 or fields[2] != "Co":
                raise ValueError(f"Only cobalt geometries are supported: {name}")
            positions.append([number(value) for value in fields[3:6]])
        elif fields[0] == "END" and name in required:
            if not positions or name in geometries:
                raise ValueError(f"Empty or duplicate geometry: {name}")
            geometries[name] = positions
            name = None
    if required != geometries.keys():
        raise ValueError(f"Missing geometries: {sorted(required - geometries.keys())}")
    return geometries


def make_template(ffield, params, bounds_order):
    lines = ffield.read_text().splitlines(keepends=True)
    atom_header = next(index for index, line in enumerate(lines) if "Nr of atoms" in line)
    bond_header = next(index for index, line in enumerate(lines) if "Nr of bonds" in line)
    if int(lines[atom_header].split()[0]) != 1 or lines[atom_header + 4].split()[0] != "Co":
        raise ValueError("This smoke runner requires a one-element cobalt force field")
    if int(lines[bond_header].split()[0]) != 1:
        raise ValueError("This smoke runner requires one Co-Co bond entry")
    bounds, initial, replacements = {}, {}, {}
    for line in params.read_text().splitlines():
        fields = re.split(r"[!#]", line, maxsplit=1)[0].split()
        if not fields:
            continue
        if len(fields) != 6:
            raise ValueError("params requires six columns")
        section, row, column = map(int, fields[:3])
        number(fields[3])
        lower, upper = map(number, fields[4:])
        if bounds_order == "max_min":
            lower, upper = upper, lower
        if lower >= upper or row != 1 or section not in (2, 3) or not 1 <= column <= {2: 32, 3: 16}[section]:
            raise ValueError("Unsupported parameter address or invalid bounds")
        offset, token = divmod(column - 1, 8)
        line_index = {2: atom_header + 4, 3: bond_header + 2}[section] + offset
        token += ({2: 1, 3: 2}[section] if offset == 0 else 0)
        name = f"p_{section}_{row}_{column}"
        if name in bounds:
            raise ValueError("Duplicate parameter")
        matches = list(re.finditer(r"\S+", lines[line_index].split("!", 1)[0]))
        match = matches[token]
        value = number(match.group())
        if not lower <= value <= upper:
            raise ValueError(f"Initial parameter outside bounds: {name}")
        bounds[name], initial[name] = [lower, upper], value
        replacements.setdefault(line_index, []).append((match.start(), match.end(), f"<<{name}>>"))
    if not bounds:
        raise ValueError("No parameters selected")
    for line_index, edits in replacements.items():
        for start, end, text in sorted(edits, reverse=True):
            lines[line_index] = lines[line_index][:start] + text + lines[line_index][end:]
    return "".join(lines), bounds, initial


def load_request(path):
    request = json.loads(path.read_text())
    if request.get("contract") != "reaxkit-external-runner-v1" or request.get("optimizer") != "ezff":
        raise ValueError("Expected ReaxKit external-runner contract for ezff")
    settings = request["settings"]
    allowed = {"algorithm", "num_epochs", "population_size", "num_errors"}
    if set(request.get("explicit_settings", settings)) - allowed:
        raise ValueError("This serial smoke runner supports only algorithm, num_epochs, population_size, num_errors")
    if settings.get("algorithm", "").upper() not in {"ONEPLUSONE_SO", "RANDOMSEARCH_SO"}:
        raise ValueError("Supported smoke algorithms: ONEPLUSONE_SO, RANDOMSEARCH_SO")
    for key in ("population_size", "num_epochs"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings.get("num_errors") != 1 or request.get("seed") is not None:
        raise ValueError("Use num_errors: 1 and seed: null; RNG forwarding is not verified")
    if request["inputs"].get("control"):
        raise ValueError("Use control: null; this LAMMPS profile performs static evaluations")
    order = request.get("params_bounds_order")
    if order not in ("min_max", "max_min"):
        raise ValueError("Explicit params_bounds_order is required")
    inputs = {key: Path(request["inputs"][key]) for key in ("ffield", "params", "geo", "trainset")}
    template, bounds, initial = make_template(inputs["ffield"], inputs["params"], order)
    targets = read_targets(inputs["trainset"])
    required = {target[key] for target in targets for key in ("first", "second")}
    geometries = read_geometries(inputs["geo"], required)
    return request, template, bounds, initial, targets, geometries


def lammps_energy(directory, positions, executable, timeout):
    directory.mkdir()
    axes = list(zip(*positions))
    limits = [(min(axis) - 20, max(axis) + 20) for axis in axes]
    data = ["Cobalt static evaluation\n", f"{len(positions)} atoms\n1 atom types\n"]
    data.extend(f"{lower:.12g} {upper:.12g} {axis}lo {axis}hi" for axis, (lower, upper) in zip("xyz", limits))
    data.extend(["\nMasses\n", "1 58.933194", "\nAtoms # charge\n"])
    data.extend(f"{index} 1 0 " + " ".join(f"{value:.12g}" for value in position)
                for index, position in enumerate(positions, 1))
    (directory / "atoms.data").write_text("\n".join(data) + "\n")
    (directory / "in.lmp").write_text(
        "units real\natom_style charge\nboundary f f f\nread_data atoms.data\n"
        "pair_style reaxff NULL\npair_coeff * * ../ffield Co\n"
        "neighbor 2.0 bin\nneigh_modify every 1 delay 0 check yes\n"
        "fix charges all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff\n"
        "fix integrator all nve\n"
        "thermo 1\nthermo_style custom step atoms pe\nrun 0\n"
        'print "EZFF_ENERGY $(pe:%.17g)" file energy.txt\n')
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        subprocess.run([executable, "-in", "in.lmp"], cwd=directory, stdout=stdout,
                       stderr=stderr, timeout=timeout, check=True)
    if "WARNING" in (directory / "stdout.log").read_text():
        raise ValueError(f"LAMMPS warning requires review: {directory / 'stdout.log'}")
    fields = (directory / "energy.txt").read_text().split()
    if len(fields) != 2 or fields[0] != "EZFF_ENERGY":
        raise ValueError("Invalid LAMMPS energy output")
    return number(fields[1])


class CobaltObjective:
    def __init__(self, root, bounds, targets, geometries, executable, timeout):
        self.root, self.bounds, self.targets, self.geometries = root, bounds, targets, geometries
        self.executable, self.timeout = executable, timeout
        self.count = 0
        self.clock = None
        self.best_loss = math.inf

    def __call__(self, values, template):
        if self.clock is not None and self.clock.should_stop():
            raise OptimizationTimeLimit()
        directory = self.root / "evaluations" / f"candidate-{self.count:04d}"
        directory.mkdir(parents=True, exist_ok=False)
        candidate = self.count
        self.count += 1
        text = template
        for name, (lower, upper) in self.bounds.items():
            value = number(values[name])
            rendered = f"{value:.6f}"
            if not lower <= number(rendered) <= upper:
                raise ValueError(f"Candidate outside bounds after rendering: {name}")
            text = text.replace(f"<<{name}>>", rendered)
        if "<<" in text:
            raise ValueError("Unresolved template variable")
        (directory / "ffield").write_text(text)
        write_json(directory / "parameters.json", {name: number(value) for name, value in values.items()})
        try:
            energies = {name: lammps_energy(directory / f"geometry-{index:04d}", positions, self.executable, self.timeout)
                        for index, (name, positions) in enumerate(sorted(self.geometries.items()))}
            items = []
            for target in self.targets:
                predicted = energies[target["first"]] / target["first_divisor"] - energies[target["second"]] / target["second_divisor"]
                items.append(dict(target, prediction=predicted, unit="kcal/mol",
                                  weighted_squared_error=((predicted - target["reference"]) / target["tolerance"]) ** 2))
            loss = number(math.sqrt(sum(item["weighted_squared_error"] for item in items)))
            report = dict(candidate=candidate, loss=loss, objective=OBJECTIVE, items=items,
                          engine="LAMMPS ReaxFF", dataset_scope="Co_clusters/Co2_cluster energy targets only",
                          geometry_relaxation=False, boundary="nonperiodic", energies=energies)
            write_json(directory / "report.json", report)
        except Exception as error:
            write_json(directory / "failure.json", {"error": str(error)})
            raise
        print(f"Candidate {candidate}: {loss:.12g}", flush=True)
        if loss < self.best_loss:
            temporary = self.root / "ffield_best.tmp"
            temporary.write_text(text)
            temporary.replace(self.root / "ffield_best")
            self.best_loss = loss
        if self.clock is not None:
            self.clock.checkpoint(loss)
        return [loss]


class OptimizationTimeLimit(Exception):
    pass


def main(argv=None):
    from run_budget import Budget
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_config", type=Path)
    parser.add_argument("--acknowledge-cobalt-subset", action="store_true", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--lammps", default="lmp")
    parser.add_argument("--engine-timeout", type=float, default=60)
    parser.add_argument("--time-limit-seconds", type=float)
    args = parser.parse_args(argv)
    request, template, bounds, initial, targets, geometries = load_request(args.run_config)
    clock = None if args.check_only else Budget(request["native_directory"], args.time_limit_seconds)
    executable = shutil.which(args.lammps)
    if not executable or not math.isfinite(args.engine_timeout) or args.engine_timeout <= 0:
        raise ValueError("LAMMPS executable and a positive finite timeout are required")
    print(f"Cobalt subset: {len(bounds)} parameters, {len(targets)} targets, {len(geometries)} geometries", flush=True)
    if args.check_only:
        print("Input checks passed; engine execution remains unverified")
        return 0
    import ezff

    root = Path(request["native_directory"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("evaluations", "results", "cobalt-run.json"):
        if (root / name).exists():
            raise ValueError("Existing run outputs found; prepare a new campaign")
    settings = request["settings"]
    write_json(root / "cobalt-run.json", dict(objective=OBJECTIVE, engine="LAMMPS ReaxFF", executable=executable,
                                             dataset_scope="Co_clusters/Co2_cluster energy targets only",
                                             ezff_version=ezff.__version__, seed_forwarded=None, settings=settings))
    (root / "ffield.template").write_text(template)
    (root / "variable_bounds").write_text("".join(f"{name} {lower:.17g} {upper:.17g}\n" for name, (lower, upper) in bounds.items()))
    objective = CobaltObjective(root, bounds, targets, geometries, executable, args.engine_timeout)
    optimizer = ezff.FFParam(error_function=objective, num_errors=1)
    optimizer.read_variable_bounds(str(root / "variable_bounds"))
    optimizer.read_forcefield_template(str(root / "ffield.template"))
    optimizer.variables.append([initial[name] for name in optimizer.variable_names])
    optimizer.errors.append(objective(initial, template))
    clock.begin()
    objective.clock = clock
    previous = Path.cwd()
    try:
        os.chdir(root)
        optimizer.set_algorithm(settings["algorithm"], population_size=settings["population_size"])
        optimizer.parameterize(num_epochs=settings["num_epochs"])
    except OptimizationTimeLimit:
        pass
    finally:
        os.chdir(previous)
    clock.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
