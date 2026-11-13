import pybdsim
import os
import pytest
import numpy as np
from pathlib import Path
from typing import Any

SLOPE     = 0.01   # T/cm
AMPLITUDE = 0.5    # T
PERIOD    = 10.0   # cm
LENGTH    = 60.0   # cm, extent of the half map along the scanned reflected axis
NPOINTS   = 120
NQUERY    = 100
QUERY_MAX = 75.0   # cm

# the box spans the whole query range
BOX_LENGTH = 160   # cm
BOX_WIDTH  = 200   # cm

# every field component is non-zero and different, so a wrong sign or a leak between them shows.
# The component along a reflected axis is flipped there, so it has to be zero at that axis' origin.
COMPONENT_SCALE = {"x": 1.0, "y": 0.5, "z": 0.25}

# the axes other than the scanned one: map extent (cm, ns), points, field gradient (per cm, per ns)
# and the position they are queried at (cm, ns). A reflected axis is queried at +/- this position.
OTHER_RANGE = {"x": 20.0, "y": 20.0, "z": 20.0, "t": 2.0}
NOTHER      = 5
GRADIENT    = {"x": 0.01, "y": 0.02, "z": 0.015, "t": 0.1}
FIXED       = {"x": 5.5, "y": 5.5, "z": 5.5, "t": 0.7}

FIELD_CLASS = {1: pybdsim.Field.Field1D, 2: pybdsim.Field.Field2D,
               3: pybdsim.Field.Field3D, 4: pybdsim.Field.Field4D}


def _ramp(u: np.ndarray, period: float) -> np.ndarray:
    return SLOPE * u


def _sine(u: np.ndarray, period: float) -> np.ndarray:
    return AMPLITUDE * np.sin(2 * np.pi * u / period)


# shape -> (function of a coordinate and a period, amplitude)
SHAPES = {"ramp": (_ramp, SLOPE * LENGTH), "sine": (_sine, AMPLITUDE)}

# case, dimension, reflected axes (the first is scanned), interpolator, shape, relative tolerance on the amplitude
FIELD_REFLECTION_CASES = [
    ("1d_x_linear_ramp",    1, "x",    "linear", "ramp", 1e-4),
    ("1d_x_cubic_sine",     1, "x",    "cubic",  "sine", 1e-2),
    ("1d_y_linear_ramp",    1, "y",    "linear", "ramp", 1e-4),
    ("1d_y_cubic_sine",     1, "y",    "cubic",  "sine", 1e-2),
    ("1d_z_linear_ramp",    1, "z",    "linear", "ramp", 1e-4),
    ("1d_z_cubic_sine",     1, "z",    "cubic",  "sine", 1e-2),
    ("2d_x_linear_ramp",    2, "x",    "linear", "ramp", 1e-4),
    ("2d_x_cubic_sine",     2, "x",    "cubic",  "sine", 1e-2),
    ("2d_y_linear_ramp",    2, "y",    "linear", "ramp", 1e-4),
    ("2d_y_cubic_sine",     2, "y",    "cubic",  "sine", 1e-2),
    ("3d_x_linear_ramp",    3, "x",    "linear", "ramp", 1e-4),
    ("3d_x_cubic_sine",     3, "x",    "cubic",  "sine", 1e-2),
    ("3d_y_linear_ramp",    3, "y",    "linear", "ramp", 1e-4),
    ("3d_y_cubic_sine",     3, "y",    "cubic",  "sine", 1e-2),
    ("3d_z_linear_ramp",    3, "z",    "linear", "ramp", 1e-4),
    ("3d_z_cubic_sine",     3, "z",    "cubic",  "sine", 1e-2),
    ("4d_x_linear_ramp",    4, "x",    "linear", "ramp", 1e-4),
    ("4d_x_cubic_sine",     4, "x",    "cubic",  "sine", 1e-2),
    ("4d_y_linear_ramp",    4, "y",    "linear", "ramp", 1e-4),
    ("4d_y_cubic_sine",     4, "y",    "cubic",  "sine", 1e-2),
    ("4d_z_linear_ramp",    4, "z",    "linear", "ramp", 1e-4),
    ("4d_z_cubic_sine",     4, "z",    "cubic",  "sine", 1e-2),
    ("2d_xy_linear_ramp",   2, "xy",   "linear", "ramp", 1e-4),
    ("2d_xy_cubic_sine",    2, "xy",   "cubic",  "sine", 1e-2),
    ("3d_xy_linear_ramp",   3, "xy",   "linear", "ramp", 1e-4),
    ("3d_xy_cubic_sine",    3, "xy",   "cubic",  "sine", 1e-2),
    ("3d_xz_linear_ramp",   3, "xz",   "linear", "ramp", 1e-4),
    ("3d_xz_cubic_sine",    3, "xz",   "cubic",  "sine", 1e-2),
    ("3d_yz_linear_ramp",   3, "yz",   "linear", "ramp", 1e-4),
    ("3d_yz_cubic_sine",    3, "yz",   "cubic",  "sine", 1e-2),
    ("3d_xyz_linear_ramp",  3, "xyz",  "linear", "ramp", 1e-4),
    ("3d_xyz_cubic_sine",   3, "xyz",  "cubic",  "sine", 1e-2),
    ("4d_xt_linear_ramp",   4, "xt",   "linear", "ramp", 1e-4),
    ("4d_xt_cubic_sine",    4, "xt",   "cubic",  "sine", 1e-2),
    ("4d_xyzt_linear_ramp", 4, "xyzt", "linear", "ramp", 1e-4),
    ("4d_xyzt_cubic_sine",  4, "xyzt", "cubic",  "sine", 1e-2),
]


def _gradient_factor(coordinates: dict[str, Any], skip: str) -> Any:
    # multilinear dependence on every axis other than the scanned one
    factor = 1.0
    for a, c in coordinates.items():
        if a != skip:
            factor = factor * (1 + GRADIENT[a] * c)
    return factor


def _period(axis: str, scan: str) -> float:
    # the other axes only have a few map points so need a long period to be interpolated well
    return PERIOD if axis == scan else 4 * OTHER_RANGE[axis]


def _component(shape: Any, c: str, coordinates: dict[str, Any], reflected: str) -> Any:
    # a component along a reflected axis follows that axis, the others follow the scanned one
    scan = reflected[0]
    a = c if c in reflected else scan
    return (COMPONENT_SCALE[c] * shape(coordinates[a], _period(a, scan))
            * _gradient_factor(coordinates, a))


def _map_axes(dim: int, reflected: str) -> str:
    # a 1D map is along the reflected axis itself, higher dimensions are x, y, z, t in order
    return reflected if dim == 1 else "xyzt"[:dim]


def _build_map(dim: int, reflected: str, shape: Any) -> np.ndarray:
    # only the u >= 0 half of each reflected axis is written - the reflections supply the rest
    scan = reflected[0]
    axes = _map_axes(dim, reflected)
    ranges = []
    for a in axes:
        if a == scan:
            ranges.append(np.linspace(0, LENGTH, NPOINTS))
        elif a in reflected:
            ranges.append(np.linspace(0, OTHER_RANGE[a], NOTHER))
        else:
            ranges.append(np.linspace(-OTHER_RANGE[a], OTHER_RANGE[a], NOTHER))
    grids = np.meshgrid(*ranges, indexing="ij")
    coordinates = dict(zip(axes, grids))

    data = np.zeros(grids[0].shape + (dim + 3,))
    for i, a in enumerate(axes):
        data[..., i] = grids[i] * (1e-9 if a == "t" else 1)  # file: cm and seconds
    for i, c in enumerate("xyz"):
        data[..., dim + i] = _component(shape, c, coordinates, reflected)
    return data


def _write_map(path: Path, dim: int, reflected: str, data: np.ndarray) -> None:
    if dim == 1:
        FIELD_CLASS[1](data, column=reflected.upper()).Write(str(path))
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


def _query_axes(dim: int, reflected: str) -> list[dict[str, Any]]:
    # scan the first reflected axis, sit at +/- a point on the other reflected axes so every
    # sign combination is queried, and at a single point on all the rest
    scan = reflected[0]
    query = []
    for a in "xyzt" if dim == 4 else "xyz":
        offset = _query_offset(a)
        if a == scan:
            query.append({"name": a, "n": NQUERY, "lo": offset - QUERY_MAX, "hi": offset + QUERY_MAX})
        elif a in reflected:
            query.append({"name": a, "n": 2, "lo": offset - FIXED[a], "hi": offset + FIXED[a]})
        else:
            query.append({"name": a, "n": 1, "lo": offset + FIXED[a], "hi": 0})
    return query


def _run_query(directory: Path, name: str, dim: int, reflected: str, map_file: Path,
               interpolator: str, reflection: str,
               nprimary: int) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    gmad_name = directory / f"{name}.gmad"
    out_name  = directory / f"{name}.dat"
    query = _query_axes(dim, reflected)
    params = {
        'DIM': dim,
        'FIELD_FILE': str(map_file),
        'INTERPOLATOR': interpolator,
        'REFLECTION': reflection,
        'BOX_LENGTH': BOX_LENGTH,
        'BOX_WIDTH': BOX_WIDTH,
        'QUERY': query,
        'OUTFILE': str(out_name),
    }
    pybdsim.Run.RenderGmadJinjaTemplate("field_map_query.tpl", str(gmad_name), params)
    sproc, timing = pybdsim.Run.Bdsim(str(gmad_name), str(directory / name), nprimary,
                                      seed=1, returnTiming=True)
    assert sproc.returncode == 0
    result = _load_query(out_name)

    # the columns are the scanned coordinates (x, y, z, t order, cm and seconds) then Fx, Fy, Fz
    scanned = [q["name"] for q in query if q["n"] > 1]
    positions = {}
    for q in query:
        a = q["name"]
        if a in scanned:
            column = result[:, scanned.index(a)]
            positions[a] = column / 1e-9 if a == "t" else column - _query_offset(a)
        else:
            positions[a] = np.full(len(result), FIXED[a])
    return positions, result[:, len(scanned):], timing


@pytest.mark.parametrize("case, dim, reflected, interpolator, shape, rel_tol", FIELD_REFLECTION_CASES)
def test(geant4_version: str, bdsim_version: str,
         test_length: str, testlength_primaries: Any, testdata_store: Any, tmp_path: Path,
         case: str, dim: int, reflected: str, interpolator: str, shape: str, rel_tol: float) -> None:

    os.chdir(Path(__file__).resolve().parent)

    nprimary = testlength_primaries.get_nprimary(__file__, test_length)
    shape_function, amplitude = SHAPES[shape]
    scan = reflected[0]
    reflection = " ".join(f"reflect{a}" for a in reflected)

    map_file = tmp_path / f"map_{case}.dat"
    _write_map(map_file, dim, reflected, _build_map(dim, reflected, shape_function))

    position, reflect_field, t_reflected = _run_query(tmp_path, f"reflected_{case}", dim, reflected,
                                                      map_file, interpolator, reflection, nprimary)
    control_position, control_field, t_control = _run_query(tmp_path, f"control_{case}", dim, reflected,
                                                            map_file, interpolator, "", nprimary)
    for a in position:
        assert np.allclose(position[a], control_position[a])

    # the field at the mirrored point, and the sign each component takes there:
    # reflecting an axis flips the field component along that axis on the reflected side
    mirrored = {a: np.abs(position[a]) for a in _map_axes(dim, reflected)}
    expected = {c: _component(shape_function, c, mirrored, reflected) for c in "xyz"}
    sign = {c: np.sign(position[c]) if c in reflected else 1.0 for c in "xyz"}
    tol = rel_tol * amplitude * max(np.max(_gradient_factor(mirrored, a)) for a in mirrored)

    q = position[scan]
    inside  = np.abs(q) < 58
    outside = np.abs(q) > 61

    error = np.max([np.abs(reflect_field[:, i] - sign[c] * expected[c])
                    for i, c in enumerate("xyz")], axis=0)
    assert error[inside].max() < tol
    assert np.abs(reflect_field[outside]).max() < 1e-12

    # without the reflection the field is only the written quadrant (all reflected coordinates > 0),
    # and is zero beyond the scanned axis' origin
    mapped   = np.all([position[a] > 0 for a in reflected], axis=0)
    positive = mapped & inside & (q > 1)
    negative = q < -1
    assert positive.any() and negative.any()
    for i, c in enumerate("xyz"):
        assert np.abs(control_field[positive, i] - expected[c][positive]).max() < tol
    assert np.abs(control_field[negative]).max() < 1e-12

    te = testdata_store.new_test_entry("14_field/field_reflection_" + case, __file__, nprimary,
                                       t_reflected + t_control)
    te.add_input_parameter_dict({'dimension': dim, 'reflected': reflected, 'interpolator': interpolator,
                                 'shape': shape, 'reflection': reflection})
    te.add_output_parameter("max_abs_error", float(error[inside].max()))
