OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[3];
h q[1];
h q[0];
cu(0.6777056476788708,0.24933788345857033,0.08937134083608667,0.062487271005078115) q[0],q[1];
cx q[1],q[2];
cx q[0],q[2];
cx q[3],q[1];
