"""
Reproduction smoke check of an environment that holds dolfinx.

It solves -Δu = f on the unit square with the exact solution u = 1 + x² + 2y², so
f = -6 and the Dirichlet data come from u. The solution lies in the degree-2
Lagrange space, so the discrete solution equals u up to rounding. The script
imports neither ``advent`` nor ``advent_dev``, so it runs in every environment and
in the public repository.
"""

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import basix
import dolfinx
import ffcx
import numpy as np
import petsc4py
import tyro
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

logger = logging.getLogger(__name__)
"""The logger of this script."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Parameters:
    """Parameters of the smoke check."""

    TOLERANCE: ClassVar[float] = 1e-8
    """
    Largest passing relative forward error, in the max norm over the dof vector.

    Admitted, not derived: many orders above the rounding error of a direct solve and
    many orders below the O(1) error of a wrong assembly or solve. A NaN fails.
    """

    env_file: Path
    """The environment file whose ``fenics-dolfinx`` pin the series must match."""

    def __post_init__(self) -> None:
        """
        Refuse an environment file that does not exist.

        :raises ValueError: If ``env_file`` is not a file.
        """
        if not self.env_file.is_file():
            raise ValueError(f"env_file is not a file: {self.env_file}")


def pinned_series(env_file: Path) -> str:
    """
    Return the ``major.minor`` series of the ``fenics-dolfinx`` pin of a file.

    :param env_file: The environment file.
    :returns: The series, for example ``0.11``.
    :raises RuntimeError: If the file has no ``fenics-dolfinx=`` pin.
    """
    match = re.search(
        r"^\s*-\s*fenics-dolfinx=(\d+\.\d+)", env_file.read_text(), re.MULTILINE
    )
    if match is None:
        raise RuntimeError(f"no fenics-dolfinx= pin in {env_file}")
    return match.group(1)


def relative_error() -> float:
    """
    Solve the problem and return the relative forward error of the dof vector.

    The error is the max norm of ``u_h - I u`` over the max norm of ``I u``, with
    ``I u`` the interpolant of the exact solution; degree-2 Lagrange dofs are point
    values, so ``I u`` holds the exact solution at every dof.

    :returns: The error.
    """
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    space = fem.functionspace(domain, ("Lagrange", 2))
    x = ufl.SpatialCoordinate(domain)
    exact_ufl = 1 + x[0] ** 2 + 2 * x[1] ** 2
    exact = fem.Function(space)
    exact.interpolate(fem.Expression(exact_ufl, space.element.interpolation_points))

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(space, domain.topology.dim - 1, facets)
    boundary = fem.dirichletbc(exact, dofs)

    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    problem = LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        -ufl.div(ufl.grad(exact_ufl)) * v * ufl.dx,
        bcs=[boundary],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="advent_smoke_",
    )
    solution = problem.solve()

    error = domain.comm.allreduce(
        float(np.max(np.abs(solution.x.array - exact.x.array))), op=MPI.MAX
    )
    scale = domain.comm.allreduce(float(np.max(np.abs(exact.x.array))), op=MPI.MAX)
    return error / scale


def main(params: Parameters) -> None:
    """
    Print the version table, then fail unless the environment is the one pinned.

    :param params: The parameters of the run.
    :raises RuntimeError: If the installed dolfinx series differs from the pin, the
        default scalar type is not ``numpy.float64``, or the solve is not accurate to
        ``Parameters.TOLERANCE``.
    """
    logger.info("running with %s", params)

    table = {
        "python": sys.version.split()[0],
        "dolfinx": dolfinx.__version__,
        "basix": basix.__version__,
        "ufl": ufl.__version__,
        "ffcx": ffcx.__version__,
        "petsc4py": petsc4py.__version__,
        "default_scalar_type": np.dtype(default_scalar_type).name,
    }
    for name, version in table.items():
        print(f"{name:<20}{version}")

    series = pinned_series(params.env_file)
    if ".".join(dolfinx.__version__.split(".")[:2]) != series:
        raise RuntimeError(
            f"dolfinx {dolfinx.__version__} is installed, "
            f"but {params.env_file} pins the series {series}"
        )
    if default_scalar_type is not np.float64:
        raise RuntimeError(f"default scalar type is {default_scalar_type}, not float64")

    error = relative_error()
    print(f"{'relative error':<20}{error:.3e}")
    if not error <= Parameters.TOLERANCE:
        raise RuntimeError(
            f"relative error {error:.3e} exceeds {Parameters.TOLERANCE:.0e}"
        )


if __name__ == "__main__":
    main(tyro.cli(Parameters))
