"""
Order smoke of the ``advent-cut`` env: CutFEMx P1 converges at order 2 on a cut disk.

Solves -Δu = f on the disk of radius 0.4 centred at (1/2, 1/2), cut out of the
unit-square background mesh, with u = 1 + x² + 2y² and f = -6. Dirichlet data are
imposed by Nitsche's method on the embedded boundary, with a ghost penalty on the
facets of the cut cells. The background meshes are an N by N grid of squares, with
N = 8, 16, 32, 64, so there are three refinements; the whole check runs twice, once
with the squares split into triangles and once with quadrilateral cells. The error is
the L² error over the discrete cut domain, integrated with the runtime quadrature of
CutFEMx. The level set is a P1 function, so the discrete domain is a polygon and u
is not reproduced exactly.

Inactive dofs, the background dofs whose support misses the cut domain, give empty
matrix rows. This script keeps the full P1 space on the background mesh and replaces
each inactive row by the identity with a zero right-hand side
(``cutfemx.petsc.deactivate_outside``), the second of the two ways of section 4.2 of
``docs/design/setup.md``.
The solve is serial and direct: PETSc ``preonly`` with LU from MUMPS.

The checks bound the order of the error only. With the ghost penalty switched off
the order check still passes, so this script does not test the stabilisation.

The precision contract is float64 throughout, checked against
``default_scalar_type``. The script prints the version table and the error table,
and raises ``RuntimeError`` with the measured value on the first failed check, never
by ``assert``. It imports neither ``advent`` nor ``advent_dev``, so it runs in every
env that has CutFEMx.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import ClassVar

import basix
import cutfemx
import cutfemx.petsc as cutfemx_petsc
import dolfinx
import ffcx
import numpy as np
import tyro
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from numpy.typing import NDArray
from petsc4py import PETSc

logger = logging.getLogger(__name__)
"""The script's logger; the caller decides where its records go."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Parameters:
    """Parameters of the order smoke; its class variables fix the problem."""

    env_file: Path
    """The env file whose ``fenics-dolfinx=`` pin the installed dolfinx must match."""

    CENTRE: ClassVar[float] = 0.5
    """Both coordinates of the centre of the disk."""

    RADIUS: ClassVar[float] = 0.4
    """Radius of the disk; it lies inside the unit square, centred at (1/2, 1/2)."""

    SOURCE: ClassVar[float] = -6.0
    """The source term f, equal to -Δu for u = 1 + x² + 2y²."""

    MESH_SIZES: ClassVar[tuple[int, ...]] = (8, 16, 32, 64)
    """Squares per side of the four background meshes; each doubles the last."""

    CELL_TYPES: ClassVar[tuple[mesh.CellType, ...]] = (
        mesh.CellType.triangle,
        mesh.CellType.quadrilateral,
    )
    """Background cell types; CutCells cuts both with its own tables, without Algoim."""

    QUADRATURE_ORDER: ClassVar[int] = 4
    """
    Degree requested of the runtime rules; the squared error has degree 4.

    Observed, not proved: degree 8 gives the same errors to the printed digits.
    """

    NITSCHE_PENALTY: ClassVar[float] = 40.0
    """Nitsche penalty gamma in gamma / h; the value of the CutFEMx Poisson demo."""

    GHOST_PENALTY: ClassVar[float] = 0.1
    """Ghost-penalty weight gamma_g in gamma_g * h * [grad u . n][grad v . n]."""

    SLOPE_WINDOW: ClassVar[tuple[float, float]] = (1.8, 2.2)
    """
    Admitted window for the slope of log e against log h, not a derived bound.

    Three refinements separate order near 2 from near 0 or 1; no error value would.
    """

    RATE_MINIMUM: ClassVar[float] = 1.5
    """
    Admitted lower bound for each consecutive rate, not a derived bound.

    Its reason is that of ``SLOPE_WINDOW``: a rate this far under 2 is a broken method.
    """

    DOLFINX_PIN: ClassVar[str] = r"^\s*-\s*fenics-dolfinx=(\d+)\.(\d+)"
    """Pattern of the dolfinx pin in the env file; captures major and minor version."""

    def __post_init__(self) -> None:
        """
        Refuse an env file that does not exist.

        :raises FileNotFoundError: If ``env_file`` is not a file.
        """
        if not self.env_file.is_file():
            raise FileNotFoundError(f"env file not found: {self.env_file}")


def _check_versions(parameters: Parameters) -> None:
    """
    Print the version table and check the dolfinx series and the scalar type.

    :param parameters: The settled command line.
    :raises RuntimeError: If the env file has no dolfinx pin, if the installed dolfinx
        differs from the pinned series in major or minor version, or if the default
        scalar type is not float64.
    """
    rows = [
        ("python", sys.version.split()[0]),
        ("dolfinx", dolfinx.__version__),
        ("basix", basix.__version__),
        ("ufl", ufl.__version__),
        ("ffcx", ffcx.__version__),
        ("petsc4py", metadata.version("petsc4py")),
        ("default_scalar_type", dolfinx.default_scalar_type.__name__),
        ("CutFEMx", metadata.version("cutfemx")),
        ("runintgen", metadata.version("runintgen")),
        ("CutCells", metadata.version("cutcells")),
    ]
    for name, value in rows:
        print(f"{name:<22}{value}")

    match = re.search(
        Parameters.DOLFINX_PIN,
        parameters.env_file.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"no fenics-dolfinx= pin in {parameters.env_file}")
    installed = tuple(dolfinx.__version__.split(".")[:2])
    if installed != match.groups():
        raise RuntimeError(
            f"dolfinx {dolfinx.__version__} is installed; the env file pins the "
            f"{'.'.join(match.groups())} series"
        )
    if dolfinx.default_scalar_type is not np.float64:
        raise RuntimeError(
            f"default_scalar_type is {dolfinx.default_scalar_type}, not float64"
        )


def _solve_error(n: int, cell_type: mesh.CellType) -> float:
    """
    Solve the cut problem on an ``n`` by ``n`` grid mesh; return its L² error.

    :param n: Squares per side of the background mesh.
    :param cell_type: Cell type of the background mesh.
    :returns: The L² norm of ``u_h - u`` over the discrete cut domain.
    :raises RuntimeError: If the direct solve does not converge.
    """
    msh = mesh.create_unit_square(MPI.COMM_SELF, n, n, cell_type=cell_type)
    space = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(space, name="phi")
    phi.interpolate(
        lambda x: (
            np.sqrt((x[0] - Parameters.CENTRE) ** 2 + (x[1] - Parameters.CENTRE) ** 2)
            - Parameters.RADIUS
        )
    )

    cut_data = cutfemx.cut(phi)
    order = Parameters.QUADRATURE_ORDER
    dx_omega = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[
            cutfemx.locate_entities(cut_data, "phi<0"),
            cutfemx.runtime_quadrature(cut_data, "phi<0", order),
        ],
    )
    dx_gamma = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=1,
        subdomain_data=cutfemx.runtime_quadrature(cut_data, "phi=0", order),
    )
    ds_ghost = ufl.Measure(
        "dS",
        domain=msh,
        subdomain_id=2,
        subdomain_data=cutfemx.ghost_penalty_facets(cut_data, "phi<0"),
    )

    u = ufl.TrialFunction(space)
    v = ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(msh)
    n_gamma = cutfemx.normal(phi)
    n_facet = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    u_exact = 1.0 + x[0] ** 2 + 2.0 * x[1] ** 2

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
    a += (
        -ufl.dot(ufl.grad(u), n_gamma) * v
        - ufl.dot(ufl.grad(v), n_gamma) * u
        + Parameters.NITSCHE_PENALTY / h * u * v
    ) * dx_gamma
    a += (
        Parameters.GHOST_PENALTY
        * ufl.avg(h)
        * ufl.inner(ufl.jump(ufl.grad(u), n_facet), ufl.jump(ufl.grad(v), n_facet))
        * ds_ghost
    )
    rhs = Parameters.SOURCE * v * dx_omega
    rhs += (
        -ufl.dot(ufl.grad(v), n_gamma) * u_exact
        + Parameters.NITSCHE_PENALTY / h * u_exact * v
    ) * dx_gamma

    a_form = cutfemx.fem.form(a)
    matrix = cutfemx_petsc.assemble_matrix(a_form)
    matrix.assemble()
    b = cutfemx_petsc.assemble_vector(cutfemx.fem.form(rhs))
    b.assemble()
    # The sparsity pattern has no entry in the row of an inactive dof, and PETSc drops a
    # value inserted outside the pattern unless the matrix accepts new nonzeros: without
    # this option deactivate_outside leaves the row empty and the matrix singular.
    matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, flag=True)
    cutfemx_petsc.deactivate_outside(matrix, b, cutfemx.fem.active_domain(a_form))
    matrix.assemble()
    b.assemble()

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    x_h = matrix.createVecRight()
    ksp.solve(b, x_h)
    if ksp.getConvergedReason() <= 0:
        raise RuntimeError(
            f"MUMPS solve on {cell_type.name} N={n} diverged: "
            f"reason {ksp.getConvergedReason()}"
        )

    u_h = fem.Function(space, name="u_h")
    u_h.x.array[:] = x_h.getArray()
    error_form = cutfemx.fem.form((u_h - u_exact) ** 2 * dx_omega)
    return float(np.sqrt(cutfemx.fem.assemble_scalar(error_form)))


def main(parameters: Parameters) -> None:
    """
    Run the order smoke on each background cell type and print its tables.

    :param parameters: The settled command line.
    :raises RuntimeError: If a version check fails, a solve fails, the slope leaves
        ``SLOPE_WINDOW`` or a rate falls under ``RATE_MINIMUM``.
    """
    logger.info("running with %s", parameters)
    _check_versions(parameters)
    for cell_type in Parameters.CELL_TYPES:
        _check_order(cell_type)


def _check_order(cell_type: mesh.CellType) -> None:
    """
    Print the error table on one background cell type and check its order.

    :param cell_type: Cell type of the background meshes.
    :raises RuntimeError: If a solve fails, the slope leaves ``SLOPE_WINDOW`` or a
        rate falls under ``RATE_MINIMUM``.
    """
    sizes = Parameters.MESH_SIZES
    errors: NDArray[np.float64] = np.array([_solve_error(n, cell_type) for n in sizes])
    log_h = -np.log(np.array(sizes, dtype=np.float64))
    rates = np.diff(np.log(errors)) / np.diff(log_h)
    slope = float(np.polyfit(log_h, np.log(errors), 1)[0])

    print(cell_type.name)
    print(f"{'N':>6}{'L2 error':>16}{'rate':>8}")
    for i, (n, error) in enumerate(zip(sizes, errors, strict=True)):
        rate = f"{rates[i - 1]:.3f}" if i > 0 else "-"
        print(f"{n:>6}{error:>16.6e}{rate:>8}")
    print(f"least-squares slope {slope:.3f}")

    # Written so that NaN fails: every comparison with NaN is false.
    low, high = Parameters.SLOPE_WINDOW
    if not low <= slope <= high:
        raise RuntimeError(
            f"{cell_type.name}: slope {slope:.3f} is outside [{low}, {high}]; "
            f"errors {errors}"
        )
    if not np.all(rates > Parameters.RATE_MINIMUM):
        raise RuntimeError(
            f"{cell_type.name}: a rate is not above {Parameters.RATE_MINIMUM}: "
            f"rates {rates}"
        )


if __name__ == "__main__":
    main(tyro.cli(Parameters))
