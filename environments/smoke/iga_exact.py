"""
Reproduction smoke check of the ``advent-iga`` env: tIGArx solves Poisson exactly.

It solves -Δu = f on the unit square with u = 1 + x² + 2y², so f = -6. The space is
tIGArx's tensor-product B-splines of degree 2, on 8 by 8 uniform elements, with
identity geometry, so the physical mesh is the unit square. The error is measured by
evaluating u_h at an 11 by 11 grid of points. The script imports neither ``advent``
nor ``advent_dev``, so it runs in every env that has tIGArx.

Boundary data. A control value of a B-spline is not a point value, so setting the
boundary control values to u at the control points gives a wrong solution even in a
working env. The data are imposed by the projection of g = u onto the spline space:
tIGArx fixes control values only, and its own demo of nonzero boundary data projects g
the same way. Each boundary control value of u_h is the one of that projection. The
weak (Nitsche) route is not used: the local-extraction assembly of tIGArx handles cell
integrals only, so its facet terms would need the older global-extraction track.

Why u_h equals u up to rounding. Derived, in four steps that use exact integrals. In
exact arithmetic the geometry map is the identity, so every integrand is a polynomial
of degree at most 4 in each variable, which the quadrature of degree 4 integrates
exactly.

1. The space S holds every polynomial of degree at most 2 in each variable, so u is in
   S. The B-splines are linearly independent, so the mass matrix is nonsingular, and
   the L² projection of u onto S returns u's own control values. The projection of
   g = u is therefore exact: g is a quadratic and S holds every quadratic.
2. The interior knots are simple, so S is continuous and lies in H¹. The knot vectors
   are open, so on a side of the square only the basis functions of the control points
   on that side are nonzero. A function of S with zero boundary control values
   therefore has zero trace: the set S₀ of these functions lies in H¹₀.
3. Write a(w, v) for the integral of grad w . grad v and (f, v) for the integral of
   f v. Integrating by parts, a(u, v) = (f, v) for every v in H¹₀, hence for every v in
   S₀. The discrete problem asks the same of u_h, with the same boundary control values.
4. The difference u_h - u lies in S₀, and a(u_h - u, v) = 0 for every v in S₀. Taking
   v = u_h - u gives zero gradient, so u_h - u is constant on the connected square, has
   zero trace and is zero. So the stiffness matrix of the free control values is
   nonsingular: a null vector c gives w in S₀ with a(w, v) = 0 for every v in S₀, so
   w is zero and c is zero by the independence of step 1.

What remains is rounding: in the two direct solves, in the evaluation, in the identity
geometry, which tIGArx computes through its metric tensor, and in the fastmath kernels
below. Admitted, not derived: ``Parameters.TOLERANCE``.

The check does not detect a fault that leaves every identity above true for this u. A
quadrature of degree 2 gives the same answer in a run, observed and not derived: the
integrands of a(u, v) and (f, v) have degree at most 2 in each variable and the
projection applies one rule to both sides of its linear system, but the argument also
needs the matrices of that rule to stay nonsingular, which it does not prove. A constant
factor on the measure, and more fixed control values than the boundary ones, are two
more such faults.

The precision contract is float64 throughout, checked against ``default_scalar_type``.
The numba kernels of tIGArx that extract and assemble are compiled with fastmath,
which licenses reassociation, so the result is reproducible within the tolerance and
not to the last bit. The forms of this script are compiled with dolfinx's default
options, not with the fast-math options in the solver module of tIGArx, which this
script does not use. Both solves are direct: PETSc ``preonly`` with LU from MUMPS.

The script prints the version table and the error, and raises ``RuntimeError`` with the
measured value on the first failed check, never by ``assert``.
"""

import logging
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import ClassVar

import basix
import dolfinx
import ffcx
import numpy as np
import tyro
import ufl
from dolfinx import default_scalar_type, fem, geometry
from petsc4py import PETSc
from tigarx.BSplines import ExplicitBSplineControlMesh, uniform_knots
from tigarx.LocalAssembly import assemble_matrix, assemble_vector
from tigarx.LocalSpline import LocallyConstructedSpline

logger = logging.getLogger(__name__)
"""The logger of this script."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Parameters:
    """Parameters of the smoke check; its class variables fix the problem."""

    env_file: Path
    """The environment file whose ``fenics-dolfinx`` pin the series must match."""

    TOLERANCE: ClassVar[float] = 1e-8
    """
    Largest passing relative error, in the max norm over the 11 by 11 points.

    Admitted, not derived: many orders above the rounding level of a direct solve of
    this size and many orders below the error of wrong boundary data, source or solve.
    """

    DEGREE: ClassVar[int] = 2
    """Degree of the B-splines in each direction."""

    ELEMENTS: ClassVar[int] = 8
    """Uniform elements per direction."""

    SAMPLES: ClassVar[int] = 11
    """Points per direction of the grid on which u_h is evaluated."""

    SOURCE: ClassVar[float] = -6.0
    """The source term f, equal to -Δu for u = 1 + x² + 2y²."""

    def __post_init__(self) -> None:
        """
        Refuse an environment file that does not exist.

        :raises ValueError: If ``env_file`` is not a file.
        """
        if not self.env_file.is_file():
            raise ValueError(f"env_file is not a file: {self.env_file}")

    @property
    def pinned_series(self) -> str:
        """
        The ``major.minor`` series of the ``fenics-dolfinx`` pin of ``env_file``.

        :raises RuntimeError: If the file has no ``fenics-dolfinx=`` pin.
        """
        match = re.search(
            r"^\s*-\s*fenics-dolfinx=(\d+\.\d+)",
            self.env_file.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(f"no fenics-dolfinx= pin in {self.env_file}")
        return match.group(1)


def _direct_solve(matrix: PETSc.Mat, rhs: PETSc.Vec) -> PETSc.Vec:
    """
    Solve ``matrix x = rhs`` by PETSc ``preonly`` with LU from MUMPS.

    :param matrix: The assembled system matrix.
    :param rhs: The right-hand side.
    :returns: The solution.
    :raises RuntimeError: If the solve does not converge.
    """
    ksp = PETSc.KSP().create(matrix.getComm())
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    solution = matrix.createVecRight()
    ksp.solve(rhs, solution)
    if ksp.getConvergedReason() <= 0:
        raise RuntimeError(f"MUMPS solve diverged: reason {ksp.getConvergedReason()}")
    return solution


def _relative_error() -> float:
    """
    Solve the problem and return the relative error of u_h at the sample points.

    The error is the max norm of ``u_h - u`` over the sample grid, over the max norm of
    ``u`` on it.

    :returns: The error.
    :raises RuntimeError: If a direct solve does not converge.
    """
    degree = Parameters.DEGREE
    control_mesh = ExplicitBSplineControlMesh(
        [degree, degree],
        [
            uniform_knots(degree, 0.0, 1.0, Parameters.ELEMENTS),
            uniform_knots(degree, 0.0, 1.0, Parameters.ELEMENTS),
        ],
    )
    spline = LocallyConstructedSpline.get_from_mesh_and_init(
        control_mesh, quad_degree=2 * degree, dofs_per_cp=1
    )
    basis = control_mesh.getScalarSpline()

    u, v = ufl.TrialFunction(spline.V), ufl.TestFunction(spline.V)
    x = spline.get_fe_coordinates()
    exact = 1.0 + x[0] ** 2 + 2.0 * x[1] ** 2
    source = fem.Constant(spline.mesh, default_scalar_type(Parameters.SOURCE))

    # The control values of the projection of u onto the spline space; it is u itself.
    mass = assemble_matrix(fem.form(ufl.inner(u, v) * spline.dx), basis)
    moment = assemble_vector(fem.form(ufl.inner(exact, v) * spline.dx), basis)
    boundary_values = _direct_solve(mass, moment)

    stiffness = assemble_matrix(
        fem.form(ufl.inner(spline.grad(u), spline.grad(v)) * spline.dx), basis
    )
    load = assemble_vector(fem.form(ufl.inner(source, v) * spline.dx), basis)
    boundary_dofs = np.unique(
        np.concatenate(
            [
                basis.getSideDofs(direction, side)
                for direction in (0, 1)
                for side in (0, 1)
            ]
        )
    ).astype(np.int32)
    # Fix the boundary control values and move their share of the matrix to the load.
    stiffness.zeroRowsColumns(boundary_dofs, 1.0, x=boundary_values, b=load)
    control_values = _direct_solve(stiffness, load)

    solution = fem.Function(spline.V)
    spline.extract_cp_solution_to_fe(control_values, solution)

    axis = np.linspace(0.0, 1.0, Parameters.SAMPLES)
    grid_x, grid_y = np.meshgrid(axis, axis)
    points = np.zeros((grid_x.size, 3))
    points[:, 0] = grid_x.ravel()
    points[:, 1] = grid_y.ravel()
    tree = geometry.bb_tree(spline.mesh, spline.mesh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(spline.mesh, candidates, points)
    cells = np.array(
        [colliding.links(i)[0] for i in range(points.shape[0])], dtype=np.int32
    )

    computed = solution.eval(points, cells).ravel()
    reference = 1.0 + points[:, 0] ** 2 + 2.0 * points[:, 1] ** 2
    return float(np.max(np.abs(computed - reference)) / np.max(np.abs(reference)))


def main(parameters: Parameters) -> None:
    """
    Print the version table, then fail unless the environment is the one pinned.

    :param parameters: The settled command line.
    :raises RuntimeError: If the environment file has no ``fenics-dolfinx=`` pin, the
        installed dolfinx series differs from it, the default scalar type is not
        ``numpy.float64``, a solve fails, or the error exceeds ``Parameters.TOLERANCE``.
    """
    logger.info("running with %s", parameters)

    table = {
        "python": sys.version.split()[0],
        "dolfinx": dolfinx.__version__,
        "basix": basix.__version__,
        "ufl": ufl.__version__,
        "ffcx": ffcx.__version__,
        "petsc4py": metadata.version("petsc4py"),
        "default_scalar_type": np.dtype(default_scalar_type).name,
        "tIGArx": metadata.version("tIGArx"),
    }
    for name, version in table.items():
        print(f"{name:<20}{version}")

    pinned_series = parameters.pinned_series
    if ".".join(dolfinx.__version__.split(".")[:2]) != pinned_series:
        raise RuntimeError(
            f"dolfinx {dolfinx.__version__} is installed, but {parameters.env_file} "
            f"pins the series {pinned_series}"
        )
    if default_scalar_type is not np.float64:
        raise RuntimeError(f"default scalar type is {default_scalar_type}, not float64")

    error = _relative_error()
    print(f"{'relative error':<20}{error:.3e}")
    # Written so that NaN fails: every comparison with NaN is false.
    if not error <= Parameters.TOLERANCE:
        raise RuntimeError(
            f"relative error {error:.3e} exceeds {Parameters.TOLERANCE:.0e}"
        )


if __name__ == "__main__":
    main(tyro.cli(Parameters))
