from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from simdev.run.parsers import is_fatal_line


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


# Rank placement, which is not a detail on a memory-bound solver.
#
# Open MPI's default above two ranks is --bind-to socket: a rank may migrate
# between the cores of its socket, and the pages it first-touched do not
# migrate with it. --bind-to core pins it. --map-by socket is Open MPI's own
# default policy and is kept deliberately rather than replaced by --map-by
# core: filling one socket before starting the next would put a run using
# fewer ranks than cores entirely on one memory controller, which on this
# machine is half of the bandwidth it is already short of.
#
# Override with SIMDEV_MPI_ARGS - empty string for stock behaviour.
DEFAULT_MPI_ARGS = ("--bind-to", "core", "--map-by", "socket")


def mpi_args() -> list[str]:
    override = os.environ.get("SIMDEV_MPI_ARGS")
    if override is None:
        return list(DEFAULT_MPI_ARGS)
    return shlex.split(override)


def parallel_argv(argv: list[str], n_ranks: int) -> list[str]:
    return ["mpirun", "-np", str(n_ranks), *mpi_args(), *argv, "-parallel"]


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

        # Streamed to the file line by line rather than captured and written
        # at exit. Two reasons, both learned the hard way on this case:
        #
        #  - A run stopped any way other than returning - a timeout above all
        #    - used to leave NO log at all, because the whole thing lived in
        #    a Python string until the process ended. A 12-hour solve died at
        #    iteration 3,012 and left nothing to read.
        #  - A multi-hour solve cannot be watched. `tail -f logs/log.simpleFoam`
        #    now works while it runs, which is the difference between noticing
        #    a diverging case at ten minutes and at six hours.
        #
        # Fatal-error scanning moves onto the stream for the same reason: it
        # no longer needs the complete text to exist.
        fatal: list[str] = []
        with open(log_path, "w", encoding="utf-8", buffering=1, newline="") as log:
            process = subprocess.Popen(
                argv,
                cwd=self.case_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                errors="replace",
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    log.write(line)
                    if is_fatal_line(line):
                        fatal.append(line.strip())
            finally:
                process.stdout.close()
                returncode = process.wait()

        result = CommandResult(
            name=name,
            argv=argv,
            returncode=returncode,
            log_path=log_path,
            fatal=fatal,
        )

        if check:
            reasons: list[str] = []
            if returncode != 0:
                reasons.append(f"{name} failed with exit code {returncode}")
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
