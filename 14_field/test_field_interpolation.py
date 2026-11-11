import pybdsim
import os
import pytest
from pathlib import Path

# case, FIELD_TYPE, FIELD_FORMAT, FIELD_FILE, BEAM_ENERGY, FIELD_INTERPOLATOR,
# expected Sigma_x, Sigma_xp, mean_x, Sigma_y, Sigma_yp, mean_y
FIELD_INTERPOLATION_CASES = [
    ("1d_linear",    "bmap1d", "bdsim1d", "1dexample.dat", "1.0", "linear",
     0.005354202580646126, 0.001999076865966775, 0.009670783941715035,
     0.013084708676590241, 0.007202701938688855, -0.49687208319306375),
    ("1d_linearmag", "bmap1d", "bdsim1d", "1dexample.dat", "1.0", "linearmag",
     0.005360657975868763, 0.0020020719479500963, 0.009683479112656596,
     0.01307943167235869, 0.0072005010788836685, -0.49720062289237976),
    ("1d_nearest",   "bmap1d", "bdsim1d", "1dexample.dat", "1.0", "nearest",
     0.0018571098686965446, 0.0004469812469452971, 0.009439205919555388,
     0.01307691510397638, 0.007198931645154899, -0.49778312349915504),
    ("2d_linear",    "bmap2d", "bdsim2d", "2dexample.dat", "1.0", "linear",
     0.015804539022049573, 0.010316564235008652, 8.499772373153859e-05,
     0.0021579719729259575, 0.0023403134007770198, 1.8904866732009396e-05),
    ("2d_linearmag", "bmap2d", "bdsim2d", "2dexample.dat", "1.0", "linearmag",
     0.01969824900808272, 0.013199531233735034, 0.00010222111806351676,
     0.0038697662117538945, 0.004129952429779972, 4.843972614912673e-06),
    ("2d_nearest",   "bmap2d", "bdsim2d", "2dexample.dat", "1.0", "nearest",
     0.003369653184855525, 0.0017251188722198771, 3.561127408404445e-05,
     0.021222188911178278, 0.021194220357673686, -0.00018837782195287218),
    ("2d_cubic",     "bmap2d", "bdsim2d", "2dexample.dat", "1.0", "cubic",
     0.018419807674184555, 0.012275748730702854, 0.00010576244183662311,
     0.0028903558879167703, 0.0032919311481443892, 1.0439962574434957e-05),
    ("3d_linear",    "bmap3d", "bdsim3d", "3dexample.dat", "10.0", "linear",
     0.014324369577942097, 0.007515015279365734, 0.5660426369667053,
     0.0044860079849747825, 0.002270031791075037, 0.22305434429049492),
    ("3d_linearmag", "bmap3d", "bdsim3d", "3dexample.dat", "10.0", "linearmag",
     0.014327077885869297, 0.007516079445364859, 0.5661105772137642,
     0.004486432469495695, 0.002270183744965747, 0.22307491642832755),
    ("3d_nearest",   "bmap3d", "bdsim3d", "3dexample.dat", "10.0", "nearest",
     0.015180510433518032, 0.007908333993363166, 0.5767620910525322,
     0.00595327484532122, 0.003382907443600101, 0.22346383558809757),
    ("3d_cubic",     "bmap3d", "bdsim3d", "3dexample.dat", "10.0", "cubic",
     0.014271013981376675, 0.007416345178360901, 0.5799057118058205,
     0.004409168245179408, 0.002207146467824476, 0.226743698656559),
    ("4d_linear",    "bmap4d", "bdsim4d", "4dexample.dat", "1.0", "linear",
     0.0017470572654710952, 0.0005099889726282857, -0.0243233716648072,
     0.0017243080769067224, 0.00040422254644303105, -0.01227595617128536),
    ("4d_linearmag", "bmap4d", "bdsim4d", "4dexample.dat", "1.0", "linearmag",
     0.0017474172300197496, 0.0005109210746173078, -0.02438345778360963,
     0.001738130517491375, 0.0004082201986125555, -0.012322448405716568),
    ("4d_nearest",   "bmap4d", "bdsim4d", "4dexample.dat", "1.0", "nearest",
     0.0017595644708096358, 0.0005266989756302343, -0.024804426898434758,
     0.0019314756950125776, 0.0005176813500108591, -0.013160282177664339),
    ("4d_cubic",     "bmap4d", "bdsim4d", "4dexample.dat", "1.0", "cubic",
     0.0017499456565023273, 0.0005163381596259943, -0.024854568827524782,
     0.0017271221410862713, 0.0004066147497225078, -0.012548425126634538),
]

# 4D map spans t = -2 to 1.5 ns and z = -35 to 29 cm: keep the beam inside both,
# with a small maximum step so the integration is converged
PARAMS_4D = {'B_SCALING': '0.01', 'T0': '-4.4', 'FIELD_LENGTH': '0.5', 'MAX_STEP': '0.25'}

# nearest gives a piecewise-constant field, so the result still varies with the integration steps
REL_TOL = {"4d_nearest": 1e-2}

# expected values on Geant4 < 11.2, where the 4D results differ
G4_PRE_11_2_OVERRIDES = {
    "4d_linear": (0.001750676813281026, 0.0005132592594114615, -0.024323004391416908,
                  0.001724329786788213, 0.00040431646783347383, -0.012275770476087928),
    "4d_linearmag": (0.0017510501941477392, 0.0005141994224302569, -0.02438309539668262,
                     0.0017381535896425812, 0.00040831396658678337, -0.012322268273402005),
    "4d_cubic": (0.0017549558409089073, 0.0005210023812053462, -0.024858626325055956,
                 0.001727174026235025, 0.0004068548046396223, -0.01255045544905588),
}


def _g4_pre_11_2(geant4_version: str) -> bool:
    major, minor = (int(x) for x in geant4_version.split(".")[:2])
    return (major, minor) < (11, 2)


@pytest.mark.parametrize(
    "case, field_type, field_format, field_file, beam_energy, interpolator,"
    " expected_Sigma_x, expected_Sigma_xp, expected_mean_x,"
    " expected_Sigma_y, expected_Sigma_yp, expected_mean_y",
    FIELD_INTERPOLATION_CASES
)
def test(geant4_version, bdsim_version,
         test_length, testlength_primaries, testdata_store,
         case, field_type, field_format, field_file, beam_energy, interpolator,
         expected_Sigma_x, expected_Sigma_xp, expected_mean_x,
         expected_Sigma_y, expected_Sigma_yp, expected_mean_y):

    os.chdir(Path(__file__).resolve().parent)

    base_name     = f"field_interpolation_{case}"
    template_name = "field_map_interpolator.tpl"
    gmad_name     = base_name + ".gmad"
    root_name     = base_name + ".root"

    params = {
        'FIELD_TYPE': field_type,
        'FIELD_FORMAT': field_format,
        'FIELD_FILE': field_file,
        'FIELD_LENGTH': '1.0',
        'BEAM_ENERGY': beam_energy,
        'FIELD_INTERPOLATOR': interpolator,
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

    rel_tol = REL_TOL.get(case, 1e-3)

    assert pytest.approx(sampler_Sigma_x, rel=rel_tol)  == expected_Sigma_x
    assert pytest.approx(sampler_Sigma_xp, rel=rel_tol) == expected_Sigma_xp
    assert pytest.approx(sampler_mean_x, rel=rel_tol)   == expected_mean_x
    assert pytest.approx(sampler_Sigma_y, rel=rel_tol)  == expected_Sigma_y
    assert pytest.approx(sampler_Sigma_yp, rel=rel_tol) == expected_Sigma_yp
    assert pytest.approx(sampler_mean_y, rel=rel_tol)   == expected_mean_y

    te = testdata_store.new_test_entry("14_field/field_interpolation_" + case, __file__, nprimary, 0)
    te.add_input_parameter_dict(params)
    te.add_output_file(os.path.dirname(__file__)+"/"+root_name, "root")
    te.add_output_parameter("Sigma_x", sampler_Sigma_x)
    te.add_output_parameter("Sigma_xp", sampler_Sigma_xp)
    te.add_output_parameter("mean_x", sampler_mean_x)
    te.add_output_parameter("Sigma_y", sampler_Sigma_y)
    te.add_output_parameter("Sigma_yp", sampler_Sigma_yp)
    te.add_output_parameter("mean_y", sampler_mean_y)
