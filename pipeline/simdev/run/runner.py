from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from simdev.run.parsers import find_fatal_errors


class StageError(Exception):
    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


@dataclass(frozen=True)
class CommandResult:
    name: str
    argv: list[str]
    returncode: int
    log_path: Path
    fatal: list[str] = field(default_factory=list)


def parallel_argv(argv: list[str], n_ranks: int) -> list[str]:
    return ["mpirun", "-np", str(n_ranks), *argv, "-parallel"]


class Runner:
    """Runs OpenFOAM utilities, captures logs, and fails loudly."""

    def __init__(self, case_dir: Path) -> None:
        self.case_dir = Path(case_dir)
        self.log_dir = self.case_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self, argv: list[str], name: str | None = None, check: bool = True
    ) -> CommandResult:
        name = name or Path(argv[0]).name
        log_path = self.log_dir / f"log.{name}"

        completed = subprocess.run(
            argv,
            cwd=self.case_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        output = completed.stdout + completed.stderr
        log_path.write_text(output, encoding="utf-8")

        fatal = find_fatal_errors(output)
        result = CommandResult(
            name=name,
            argv=argv,
            returncode=completed.returncode,
            log_path=log_path,
            fatal=fatal,
        )

        if check:
            reasons: list[str] = []
            if completed.returncode != 0:
                reasons.append(f"{name} failed with exit code {completed.returncode}")
            if fatal:
                # Exit code 0 is not proof of success in OpenFOAM.
                reasons.append(f"{name} logged a fatal error: {fatal[0]}")
            if reasons:
                reasons.append(f"see {log_path}")
                raise StageError(reasons)

        return result

    def run_parallel(
        self,
        argv: list[str],
        n_ranks: int,
        name: str | None = None,
        check: bool = True,
    ) -> CommandResult:
        return self.run(
            parallel_argv(argv, n_ranks), name=name or Path(argv[0]).name, check=check
        )
