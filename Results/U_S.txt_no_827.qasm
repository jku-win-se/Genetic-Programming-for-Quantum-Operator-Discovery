OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[2];
ry(2.2143444503841354) q[0];
rxx(1.5388140459914932) q[0],q[3];
cy q[3],q[1];
cx q[0],q[3];
cx q[2],q[3];
