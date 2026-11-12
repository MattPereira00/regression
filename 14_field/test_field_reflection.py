import pybdsim
import os
import pytest
import numpy as np
from pathlib import Path
from typing import Any

SLOPE     = 0.01   # T/cm
AMPLITUDE = 0.5    # T
PERIOD    = 10.0   # cm
LENGTH    = 60.0   # cm, extent of the half map
NPOINTS   = 120
NQUERY    = 100
QUERY_MAX = 75.0   # cm
TANGENTIAL_SCALE = 0.5

# axis -> (box length, box width) in cm, so the box spans the whole query range
BOX = {"x": (10, 200), "y": (10, 200), "z": (160, 10)}


def _ramp(u: np.ndarray) -> np.ndarray:
    return SLOPE * u


def _sine(u: np.ndarray) -> np.ndarray:
    return AMPLITUDE * np.sin(2 * np.pi * u / PERIOD)


# shape -> (function of the axis coordinate, amplitude)
SHAPES = {"ramp": (_ramp, SLOPE * LENGTH), "sine": (_sine, AMPLITUDE)}

# case, axis, interpolator, shape, relative tolerance on the amplitude
FIELD_REFLECTION_CASES = [
    ("x_linear_ramp", "x", "linear", "ramp", 1e-4),
    ("x_cubic_sine",  "x", "cubic",  "sine", 1e-2),
    ("y_linear_ramp", "y", "linear", "ramp", 1e-4),
    ("y_cubic_sine",  "y", "cubic",  "sine", 1e-2),
    ("z_linear_ramp", "z", "linear", "ramp", 1e-4),
    ("z_cubic_sine",  "z", "cubic",  "sine", 1e-2),
]


def _load_query(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() == "" or line.startswith("!") or ">" in line:
            continue
        rows.append([float(v) for v in line.split()])
    return np.array(rows)


def _query_offset(axis: str) -> float:
    # z is measured from the start of the box, x and y from its centre
    return BOX[axis][0] / 2 if axis == "z" else 0.0


def _run_query(directory: Path, name: str, axis: str, map_file: Path, interpolator: str,
               reflection: str, nprimary: int) -> tuple[np.ndarray, float]:
    gmad_name = directory / f"{name}.gmad"
    out_name  = directory / f"{name}.dat"
    offset    = _query_offset(axis)
    params = {
        'FIELD_FILE': str(map_file),
        'INTERPOLATOR': interpolator,
        'REFLECTION': reflection,
        'AXIS': axis,
        'BOX_LENGTH': BOX[axis][0],
        'BOX_WIDTH': BOX[axis][1],
        'NQ': NQUERY,
        'QMIN': offset - QUERY_MAX,
        'QMAX': offset + QUERY_MAX,
        'OUTFILE': str(out_name),
    }
    pybdsim.Run.RenderGmadJinjaTemplate("field_map_query.tpl", str(gmad_name), params)
    sproc, timing = pybdsim.Run.Bdsim(str(gmad_name), str(directory / name), nprimary,
                                      seed=1, returnTiming=True)
    assert sproc.returncode == 0
    result = _load_query(out_name)
    result[:, 0] -= offset
    return result, timing


@pytest.mark.parametrize("case, axis, interpolator, shape, rel_tol", FIELD_REFLECTION_CASES)
def test(geant4_version: str, bdsim_version: str,
         test_length: str, testlength_primaries: Any, testdata_store: Any, tmp_path: Path,
         case: str, axis: str, interpolator: str, shape: str, rel_tol: float) -> None:

    os.chdir(Path(__file__).resolve().parent)

    nprimary = testlength_primaries.get_nprimary(__file__, test_length)
    oracle, amplitude = SHAPES[shape]
    tol = rel_tol * amplitude
    reflection = f"reflect{axis}"

    # field component along the reflected axis, one transverse component, and one left at zero
    normal     = "xyz".index(axis) + 1
    tangential = "xyz".index("y" if axis != "y" else "x") + 1
    unused     = 6 - normal - tangential

    # only the u >= 0 half of the map is written - the reflection supplies the rest
    u = np.linspace(0, LENGTH, NPOINTS)
    data = np.zeros((NPOINTS, 4))
    data[:, 0] = u
    data[:, normal] = oracle(u)
    data[:, tangential] = TANGENTIAL_SCALE * oracle(u)
    map_file = tmp_path / f"map_{case}.dat"
    pybdsim.Field.Field1D(data, column=axis.upper()).Write(str(map_file))

    reflected, t_reflected = _run_query(tmp_path, f"reflected_{case}", axis, map_file, interpolator, reflection, nprimary)
    control, t_control     = _run_query(tmp_path, f"control_{case}", axis, map_file, interpolator, "", nprimary)

    q = reflected[:, 0]
    assert np.allclose(q, control[:, 0])
    inside   = np.abs(q) < 58
    outside  = np.abs(q) > 61
    positive = (q > 1) & inside
    negative = q < -1

    # reflecting an axis flips the field component along that axis on the reflected side
    error = np.maximum(np.abs(reflected[:, normal] - np.sign(q) * oracle(np.abs(q))),
                       np.abs(reflected[:, tangential] - TANGENTIAL_SCALE * oracle(np.abs(q))))
    assert error[inside].max() < tol
    assert np.abs(reflected[:, unused]).max() < 1e-12
    assert np.abs(reflected[outside, 1:]).max() < 1e-12

    assert np.abs(control[positive, normal] - oracle(q[positive])).max() < tol
    assert np.abs(control[positive, tangential] - TANGENTIAL_SCALE * oracle(q[positive])).max() < tol
    assert np.abs(control[negative, 1:]).max() < 1e-12

    te = testdata_store.new_test_entry("14_field/field_reflection_" + case, __file__, nprimary,
                                       t_reflected + t_control)
    te.add_input_parameter_dict({'axis': axis, 'interpolator': interpolator,
                                 'shape': shape, 'reflection': reflection})
    te.add_output_parameter("max_abs_error", float(error[inside].max()))
