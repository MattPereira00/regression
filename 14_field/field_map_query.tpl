f1: field, type="bmap{{ DIM }}d", magneticFile="bdsim{{ DIM }}d:{{ FIELD_FILE }}", magneticInterpolator="{{ INTERPOLATOR }}"{% if REFLECTION %}, magneticReflection="{{ REFLECTION }}"{% endif %};

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

{# lengths are in cm, but the query time is a plain number in ns #}
q1: query{% for a in QUERY %}{% set unit = '' if a.name == 't' else '*cm' %}, n{{ a.name }}={{ a.n }}, {{ a.name }}min={{ a.lo }}{{ unit }}{% if a.n > 1 %}, {{ a.name }}max={{ a.hi }}{{ unit }}{% endif %}{% endfor %},
    outfileMagnetic="{{ OUTFILE }}",
    queryMagneticField=1,
    fieldObject="f1",
    checkParameters=0;
