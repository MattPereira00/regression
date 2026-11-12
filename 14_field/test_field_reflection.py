import pybdsim
import os
import pytest
import numpy as np
from pathlib import Path
from typing import Any

SLOPE     = 0.01   # T/cm
AMPLITUDE = 0.5    # T
PERIOD    = 10.0   # cm
LENGTH    = 60.0   # cm, extent of the half map along the reflected axis
NPOINTS   = 120
NQUERY    = 100
QUERY_MAX = 75.0   # cm
TANGENTIAL_SCALE = 0.5

# the box spans the whole query range
BOX_LENGTH = 160   # cm
BOX_WIDTH  = 200   # cm

# axes that are not reflected: map extent (cm, ns), points, field gradient (per cm, per ns)
# and the position they are queried at (cm, ns)
OTHER_RANGE = {"x": 20.0, "y": 20.0, "z": 20.0, "t": 2.0}
NOTHER      = 5
GRADIENT    = {"x": 0.01, "y": 0.02, "z": 0.015, "t": 0.1}
FIXED       = {"x": 5.5, "y": 5.5, "z": 5.5, "t": 0.5}

FIELD_CLASS = {1: pybdsim.Field.Field1D, 2: pybdsim.Field.Field2D,
               3: pybdsim.Field.Field3D, 4: pybdsim.Field.Field4D}


def _ramp(u: np.ndarray) -> np.ndarray:
    return SLOPE * u


def _sine(u: np.ndarray) -> np.ndarray:
    return AMPLITUDE * np.sin(2 * np.pi * u / PERIOD)


# shape -> (function of the axis coordinate, amplitude)
SHAPES = {"ramp": (_ramp, SLOPE * LENGTH), "sine": (_sine, AMPLITUDE)}

# case, dimension, axis, interpolator, shape, relative tolerance on the amplitude
FIELD_REFLECTION_CASES = [
    ("1d_x_linear_ramp", 1, "x", "linear", "ramp", 1e-4),
    ("1d_x_cubic_sine",  1, "x", "cubic",  "sine", 1e-2),
    ("1d_y_linear_ramp", 1, "y", "linear", "ramp", 1e-4),
    ("1d_y_cubic_sine",  1, "y", "cubic",  "sine", 1e-2),
    ("1d_z_linear_ramp", 1, "z", "linear", "ramp", 1e-4),
    ("1d_z_cubic_sine",  1, "z", "cubic",  "sine", 1e-2),
    ("2d_x_linear_ramp", 2, "x", "linear", "ramp", 1e-4),
    ("2d_x_cubic_sine",  2, "x", "cubic",  "sine", 1e-2),
    ("2d_y_linear_ramp", 2, "y", "linear", "ramp", 1e-4),
    ("2d_y_cubic_sine",  2, "y", "cubic",  "sine", 1e-2),
    ("3d_x_linear_ramp", 3, "x", "linear", "ramp", 1e-4),
    ("3d_x_cubic_sine",  3, "x", "cubic",  "sine", 1e-2),
    ("3d_y_linear_ramp", 3, "y", "linear", "ramp", 1e-4),
    ("3d_y_cubic_sine",  3, "y", "cubic",  "sine", 1e-2),
    ("3d_z_linear_ramp", 3, "z", "linear", "ramp", 1e-4),
    ("3d_z_cubic_sine",  3, "z", "cubic",  "sine", 1e-2),
    ("4d_x_linear_ramp", 4, "x", "linear", "ramp", 1e-4),
    ("4d_x_cubic_sine",  4, "x", "cubic",  "sine", 1e-2),
    ("4d_y_linear_ramp", 4, "y", "linear", "ramp", 1e-4),
    ("4d_y_cubic_sine",  4, "y", "cubic",  "sine", 1e-2),
    ("4d_z_linear_ramp", 4, "z", "linear", "ramp", 1e-4),
    ("4d_z_cubic_sine",  4, "z", "cubic",  "sine", 1e-2),
]


def _gradient_factor(coordinates: dict[str, Any], skip: str) -> Any:
    # multilinear dependence on every axis other than the reflected one
    factor = 1.0
    for a, c in coordinates.items():
        if a != skip:
            factor = factor * (1 + GRADIENT[a] * c)
    return factor


def _map_axes(dim: int, axis: str) -> str:
    # a 1D map is along the reflected axis itself, higher dimensions are x, y, z, t in order
    return axis if dim == 1 else "xyzt"[:dim]


def _build_map(dim: int, axis: str, oracle: Any, normal: int, tangential: int) -> np.ndarray:
    # only the u >= 0 half along the reflected axis is written - the reflection supplies the rest
    axes = _map_axes(dim, axis)
    ranges = []
    for a in axes:
        if a == axis:
            ranges.append(np.linspace(0, LENGTH, NPOINTS))
        else:
            ranges.append(np.linspace(-OTHER_RANGE[a], OTHER_RANGE[a], NOTHER))
    grids = np.meshgrid(*ranges, indexing="ij")
    coordinates = dict(zip(axes, grids))

    data = np.zeros(grids[0].shape + (dim + 3,))
    for i, a in enumerate(axes):
        data[..., i] = grids[i] * (1e-9 if a == "t" else 1)  # file: cm and seconds
    u = coordinates[axis]
    scale = oracle(u) * _gradient_factor(coordinates, axis)
    data[..., dim + normal] = scale
    data[..., dim + tangential] = TANGENTIAL_SCALE * scale
    return data


def _write_map(path: Path, dim: int, axis: str, data: np.ndarray) -> None:
    if dim == 1:
        FIELD_CLASS[1](data, column=axis.upper()).Write(str(path))
    else:
        FIELD_CLASS[dim](data).Write(str(path))


def _load_query(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() == "" or line.startswith("!") or ">" in line:
            continue
        rows.append([float(v) for v in line.split()])
    return np.array(rows)


def _query_offset(axis: str) -> float:
    # z is measured from the start of the box, x and y from its centre
    return BOX_LENGTH / 2 if axis == "z" else 0.0


def _query_axes(dim: int, axis: str) -> list[dict[str, Any]]:
    # scan the reflected axis and sit at a fixed point on all the others
    query = []
    for a in "xyzt" if dim == 4 else "xyz":
        offset = _query_offset(a)
        if a == axis:
            query.append({"name": a, "n": NQUERY, "lo": offset - QUERY_MAX,
                          "hi": offset + QUERY_MAX})
        else:
            query.append({"name": a, "n": 1, "lo": offset + FIXED[a], "hi": 0})
    return query


def _run_query(directory: Path, name: str, dim: int, axis: str, map_file: Path,
               interpolator: str, reflection: str, nprimary: int) -> tuple[np.ndarray, float]:
    gmad_name = directory / f"{name}.gmad"
    out_name  = directory / f"{name}.dat"
    params = {
        'DIM': dim,
        'FIELD_FILE': str(map_file),
        'INTERPOLATOR': interpolator,
        'REFLECTION': reflection,
        'BOX_LENGTH': BOX_LENGTH,
        'BOX_WIDTH': BOX_WIDTH,
        'QUERY': _query_axes(dim, axis),
        'OUTFILE': str(out_name),
    }
    pybdsim.Run.RenderGmadJinjaTemplate("field_map_query.tpl", str(gmad_name), params)
    sproc, timing = pybdsim.Run.Bdsim(str(gmad_name), str(directory / name), nprimary,
                                      seed=1, returnTiming=True)
    assert sproc.returncode == 0
    result = _load_query(out_name)
    result[:, 0] -= _query_offset(axis)
    return result, timing


@pytest.mark.parametrize("case, dim, axis, interpolator, shape, rel_tol", FIELD_REFLECTION_CASES)
def test(geant4_version: str, bdsim_version: str,
         test_length: str, testlength_primaries: Any, testdata_store: Any, tmp_path: Path,
         case: str, dim: int, axis: str, interpolator: str, shape: str, rel_tol: float) -> None:

    os.chdir(Path(__file__).resolve().parent)

    nprimary = testlength_primaries.get_nprimary(__file__, test_length)
    oracle, amplitude = SHAPES[shape]
    reflection = f"reflect{axis}"

    # field component along the reflected axis, one transverse component, and one left at zero
    normal     = "xyz".index(axis)
    tangential = "xyz".index("y" if axis != "y" else "x")
    unused     = 3 - normal - tangential

    fixed = {a: FIXED[a] for a in _map_axes(dim, axis)}
    tol = rel_tol * amplitude * _gradient_factor(fixed, axis)

    map_file = tmp_path / f"map_{case}.dat"
    _write_map(map_file, dim, axis, _build_map(dim, axis, oracle, normal, tangential))

    reflected, t_reflected = _run_query(tmp_path, f"reflected_{case}", dim, axis, map_file,
                                        interpolator, reflection, nprimary)
    control, t_control     = _run_query(tmp_path, f"control_{case}", dim, axis, map_file,
                                        interpolator, "", nprimary)

    # columns are the scanned coordinate then Fx, Fy, Fz
    column = {"normal": normal + 1, "tangential": tangential + 1, "unused": unused + 1}

    q = reflected[:, 0]
    assert np.allclose(q, control[:, 0])
    inside   = np.abs(q) < 58
    outside  = np.abs(q) > 61
    positive = (q > 1) & inside
    negative = q < -1

    expected = oracle(np.abs(q)) * _gradient_factor(fixed, axis)

    # reflecting an axis flips the field component along that axis on the reflected side
    error = np.maximum(np.abs(reflected[:, column["normal"]] - np.sign(q) * expected),
                       np.abs(reflected[:, column["tangential"]] - TANGENTIAL_SCALE * expected))
    assert error[inside].max() < tol
    assert np.abs(reflected[:, column["unused"]]).max() < 1e-12
    assert np.abs(reflected[outside, 1:]).max() < 1e-12

    assert np.abs(control[positive, column["normal"]] - expected[positive]).max() < tol
    assert np.abs(control[positive, column["tangential"]] - TANGENTIAL_SCALE * expected[positive]).max() < tol
    assert np.abs(control[negative, 1:]).max() < 1e-12

    te = testdata_store.new_test_entry("14_field/field_reflection_" + case, __file__, nprimary,
                                       t_reflected + t_control)
    te.add_input_parameter_dict({'dimension': dim, 'axis': axis, 'interpolator': interpolator,
                                 'shape': shape, 'reflection': reflection})
    te.add_output_parameter("max_abs_error", float(error[inside].max()))
