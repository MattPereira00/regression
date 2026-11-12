f1: field, type="bmap1d", magneticFile="bdsim1d:{{ FIELD_FILE }}", magneticInterpolator="{{ INTERPOLATOR }}"{% if REFLECTION %}, magneticReflection="{{ REFLECTION }}"{% endif %};

box: rcol, fieldAll="f1", l={{ BOX_LENGTH }}*cm, material="air", horizontalWidth={{ BOX_WIDTH }}*cm;
l1: line=(box);
use, l1;

beam, particle="e-",
      energy=3*GeV,
      distrType="gauss",
      sigmaX=1*cm,
      sigmaY=1*cm,
      sigmaXp=1e-6,
      sigmaYp=1e-6;

option, ngenerate=1;
option, outputFormat="none";

q1: query, n{{ AXIS }}={{ NQ }}, {{ AXIS }}min={{ QMIN }}*cm, {{ AXIS }}max={{ QMAX }}*cm,
    outfileMagnetic="{{ OUTFILE }}",
    queryMagneticField=1,
    fieldObject="f1",
    checkParameters=0;
