import pytest

import sys
import os
import subprocess
import inspect
import json
from pathlib import Path
import shutil
from bdsim_regression import regression_data as rd

###############################################################
# BDSIM and Geant4 options
###############################################################
bash_path = shutil.which("bash") # find the right shell as typically bdsim is setup in the shell setup
_uname = subprocess.run("uname -a",
                        shell=True,
                        stdout=subprocess.PIPE).stdout.decode("utf-8")
_bdsim_version = subprocess.run("bdsim --version",
                                shell=True,
                                executable=bash_path,
                                stdout=subprocess.PIPE).stdout.decode("utf-8").strip()
_geant4_version = subprocess.run("geant4-config --version",
                                 shell=True,
                                 executable=bash_path,
                                 stdout=subprocess.PIPE).stdout.decode("utf-8").strip()
print(f"BDSIM version {_bdsim_version}")
print(f"Geant4 version {_geant4_version}")

@pytest.fixture
def uname() :
    return _uname

@pytest.fixture
def bdsim_version():
    return _bdsim_version

@pytest.fixture
def geant4_version():
    return _geant4_version

###############################################################
# Input options
###############################################################
def pytest_addoption(parser):
    parser.addoption("--length", action="store", default="short")

@pytest.fixture
def test_length(request):
    return request.config.getoption("--length")

###############################################################
# Utility functions for test names and paths
###############################################################
def get_testname(testpath) :
    p = Path(testpath)
    stem = str(p.stem)
    base = str(p.parent).split('/')[-1]
    return base+"/"+stem

def get_testpath(testpath) :
    p = Path(testpath)
    return str(p.parent)

def get_testfile_size(filepath) :
    p = Path(filepath)
    return p.stat().st_size  /(1024 ** 2)

###############################################################
# ngenerate for each test
###############################################################
class test_nprimary :
    def __init__(self):
        self.nprimary = {"02_elements/test_drift":{"short":1000,"medium":100,"long":100},
                         "02_elements/test_quadrupole":{"short":1000,"medium":100,"long":1000},
                         "02_elements/test_rbend":{"short":1000,"medium":1000,"long":1000},
                         "02_elements/test_sbend":{"short":1000,"medium":1000,"long":1000},
                         "02_elements/test_sextupole":{"short":1000,"medium":1000,"long":1000},
                         "10_beam/test_reference":{"short":1000,"medium":1000,"long":1000},
                         "10_beam/test_gaussmatrix": {"short": 1000, "medium": 1000, "long": 1000},
                         "10_beam/test_gausstwiss": {"short": 1000, "medium": 1000, "long": 1000},
                         "14_field/test_field_tracking":{"short":5000,"medium":5000,"long":5000},
                         "14_field/test_field_offset":{"short":5000,"medium":5000,"long":5000},
                         "14_field/test_field_interpolation":{"short":5000,"medium":5000,"long":5000},
                         "14_field/test_field_em_tracking":{"short":5000,"medium":5000,"long":5000},
                         "14_field/test_field_reflection":{"short":1,"medium":1,"long":1},
                         "16_eloss/test_eloss_collimator":{"short":1000,"medium":10000,"long":10000},
                         "16_eloss/test_eloss_collimator_storeElossLinks": {"short": 1000, "medium": 10000, "long": 10000},
                         "20_scorer/test_scorer3d": {"short": 5000, "medium": 10000, "long": 50000},
                         "22_processes/test_laserwire_compton_cumulative":{"short":10000,"medium":10000,"long":10000},
                         "22_processes/test_laserwire_compton_multiStep": {"short": 10000, "medium": 10000,"long": 10000},
                         "22_processes/test_synch_rad": {"short": 10000, "medium": 100000,"long": 1000000},
                         "99_machines/test_atf2":{"short":5000,"medium":10000,"long":10000},
                         "99_machines/test_diamond":{"short":1,"medium":10000,"long":10000},
                         "99_machines/test_lhc":{"short":1,"medium":10000,"long":10000},
                         "99_machines/test_lhara_stage1":{"short":10000,"medium":10000,"long":10000}}

    def get_nprimary(self,testpath, length) :
        testname = get_testname(testpath)
        return self.nprimary[testname][length]

_test_nprimary = test_nprimary()

@pytest.fixture
def testlength_primaries():
    return _test_nprimary

###############################################################
# test output store
###############################################################

# Store that is persistent
_test_entry_store = rd.test_entry_store()

@pytest.fixture
def testdata_store() :
    return _test_entry_store

def pytest_sessionfinish(session, exitstatus):
    '''Write testdata_store to file'''
    _test_entry_store.write_json("./regression_data.dat")

###############################################################
# base code for rpc bdsim call
###############################################################
@pytest.fixture
def make_bdsim_test_code() :
    return make_bdsim_test_code_func

def make_bdsim_test_code_func(func, args = "", dir="", functions=[]) :
    func_name = func.__name__

    code_to_run  = "import bdsim\n"
    code_to_run += "import pytest\n"
    code_to_run += "import sys\n"
    code_to_run += "import os\n"
    if dir != "" :
        code_to_run += 'os.chdir("' + dir + '")\n'
    for f in functions :
        code_to_run += inspect.getsource(func)
    code_to_run += inspect.getsource(func)
    code_to_run += "ret ="+func_name+"("+args+")\n"
    code_to_run += "sys.exit(ret)"

    print(code_to_run)
    return code_to_run

@pytest.fixture
def run_bdsim_test_code_as_subprocess():
    return run_bdsim_test_code_as_subprocess_func

def run_bdsim_test_code_as_subprocess_func(code) :
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()

@pytest.fixture
def simple_bdslink() :
    return simple_bdslink_code

@pytest.fixture
def running_under_pytest() :
    return running_under_pytest_func

def running_under_pytest_func():
    return "PYTEST_CURRENT_TEST" in os.environ

def simple_bdslink_code(gmadFile = "./trackerInterface.gmad",
                        pdgID = 2212,
                        kineticEnergy = 5e6,
                        relativeEnergyCutIn = 0.01,
                        seed=1234,
                        referenceIonCharge=1,
                        batchMode=True) :

    import bdsim
    bds_ti = bdsim.BDSLinkTrackerInterface.GetInstance(gmadFile,
                                                       pdgID,
                                                       kineticEnergy,
                                                       relativeEnergyCutIn,
                                                       seed,
                                                       referenceIonCharge,
                                                       batchMode)

    # create element
    e = bdsim.Element()
    e.name = "drift1"
    e.type = bdsim.elementtype.ElementType.DRIFT
    e.l = 1.0

    # add element
    l = bds_ti.GetBDSIMLink()
    l.AddLinkElement(e)

    # select element
    l.SelectLinkElement("drift1")

    # reference particle
    rp = bds_ti.GetReferenceParticleDefinition()

    # add particle
    bds_ti.AddParticle(0, # x
                       0, # y
                       0, # px
                       0, # py
                       rp.Momentum(), # pz
                       0., # t
                       0., # s
                       0, # trackid
                       2212) # pdgid;

    return bds_ti
