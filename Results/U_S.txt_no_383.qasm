OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[2];
h q[1];
y q[2];
cx q[3],q[0];
cx q[1],q[3];
h q[0];
z q[1];
z q[1];
swap q[2],q[1];
cz q[0],q[1];
h q[1];
x q[2];
