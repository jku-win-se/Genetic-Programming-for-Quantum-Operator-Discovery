OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(2.0196571048948924) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
h q[2];
ryy(2.0196571048948924) q[3],q[2];
cy q[3],q[1];
h q[0];
h q[3];
cz q[0],q[1];
h q[1];
