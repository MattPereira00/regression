import pybdsim
import os
import pytest
from pathlib import Path

# case, FIELD_TYPE, FIELD_FORMAT, FIELD_FILE, BEAM_ENERGY,
# expected Sigma_x, Sigma_xp, mean_x, Sigma_y, Sigma_yp, mean_y
FIELD_TRACK_CASES = [
    ("1d", "bmap1d", "bdsim1d", "1dexample.dat", "1.0",
     0.005376337095341065, 0.0020095770501316365, 0.009675747349709855,
     0.013080372518385144, 0.007200602146071492, -0.4976539144992828),
    ("2d", "bmap2d", "bdsim2d", "2dexample.dat", "1.0",
     0.018419807674184555, 0.012275748730702854, 0.00010576244183662311,
     0.0028903558879167703, 0.0032919311481443892, 1.0439962574434957e-05),
    ("3d", "bmap3d", "bdsim3d", "3dexample.dat", "10.0",
     0.014271013981376675, 0.007416345178360901, 0.5799057118058205,
     0.004409168245179408, 0.002207146467824476, 0.226743698656559),
    ("4d", "bmap4d", "bdsim4d", "4dexample.dat", "1.0",
     0.0017499456565023273, 0.0005163381596259943, -0.024854568827524782,
     0.0017271221410862713, 0.0004066147497225078, -0.012548425126634538),
]

# 4D map spans t = -2 to 1.5 ns and z = -35 to 29 cm: keep the beam inside both,
# with a small maximum step so the integration is converged
PARAMS_4D = {'B_SCALING': '0.01', 'T0': '-4.4', 'FIELD_LENGTH': '0.5', 'MAX_STEP': '0.25'}

# expected values on Geant4 < 11.2, where the 4D results differ
G4_PRE_11_2_OVERRIDES = {
    "4d": (0.0017549558409089073, 0.0005210023812053462, -0.024858626325055956,
           0.001727174026235025, 0.0004068548046396223, -0.01255045544905588),
}


def _g4_pre_11_2(geant4_version: str) -> bool:
    major, minor = (int(x) for x in geant4_version.split(".")[:2])
    return (major, minor) < (11, 2)


@pytest.mark.parametrize(
    "case, field_type, field_format, field_file, beam_energy,"
    " expected_Sigma_x, expected_Sigma_xp, expected_mean_x,"
    " expected_Sigma_y, expected_Sigma_yp, expected_mean_y",
    FIELD_TRACK_CASES
)
def test(geant4_version, bdsim_version,
         test_length, testlength_primaries, testdata_store,
         case, field_type, field_format, field_file, beam_energy,
         expected_Sigma_x, expected_Sigma_xp, expected_mean_x,
         expected_Sigma_y, expected_Sigma_yp, expected_mean_y):

    os.chdir(Path(__file__).resolve().parent)

    base_name     = f"field_tracking_{case}"
    template_name = "field_map_track.tpl"
    gmad_name     = base_name + ".gmad"
    root_name     = base_name + ".root"

    params = {
        'FIELD_TYPE': field_type,
        'FIELD_FORMAT': field_format,
        'FIELD_FILE': field_file,
        'FIELD_LENGTH': '1.0',
        'BEAM_ENERGY': beam_energy,
    }
    if field_type == "bmap4d":
        params.update(PARAMS_4D)

    nprimary = testlength_primaries.get_nprimary(__file__, test_length)

    pybdsim.Run.RenderGmadJinjaTemplate(template_name, gmad_name, params)
    pybdsim.Run.Bdsim(gmad_name, base_name, nprimary, 1)

    data = pybdsim.DataPandas.BDSIMOutput(root_name)
    sampler_data = data.get_sampler("d2.")

    sampler_Sigma_x  = sampler_data['x'].std()
    sampler_Sigma_xp = sampler_data['xp'].std()
    sampler_mean_x   = sampler_data['x'].mean()
    sampler_Sigma_y  = sampler_data['y'].std()
    sampler_Sigma_yp = sampler_data['yp'].std()
    sampler_mean_y   = sampler_data['y'].mean()

    if case in G4_PRE_11_2_OVERRIDES and _g4_pre_11_2(geant4_version):
        (expected_Sigma_x, expected_Sigma_xp, expected_mean_x,
         expected_Sigma_y, expected_Sigma_yp, expected_mean_y) = G4_PRE_11_2_OVERRIDES[case]

    assert pytest.approx(sampler_Sigma_x, rel=1e-3)  == expected_Sigma_x
    assert pytest.approx(sampler_Sigma_xp, rel=1e-3) == expected_Sigma_xp
    assert pytest.approx(sampler_mean_x, rel=1e-3)   == expected_mean_x
    assert pytest.approx(sampler_Sigma_y, rel=1e-3)  == expected_Sigma_y
    assert pytest.approx(sampler_Sigma_yp, rel=1e-3) == expected_Sigma_yp
    assert pytest.approx(sampler_mean_y, rel=1e-3)   == expected_mean_y

    te = testdata_store.new_test_entry("14_field/field_tracking_" + case, __file__, nprimary, 0)
    te.add_input_parameter_dict(params)
    te.add_output_file(os.path.dirname(__file__)+"/"+root_name, "root")
    te.add_output_parameter("Sigma_x", sampler_Sigma_x)
    te.add_output_parameter("Sigma_xp", sampler_Sigma_xp)
    te.add_output_parameter("mean_x", sampler_mean_x)
    te.add_output_parameter("Sigma_y", sampler_Sigma_y)
    te.add_output_parameter("Sigma_yp", sampler_Sigma_yp)
    te.add_output_parameter("mean_y", sampler_mean_y)
