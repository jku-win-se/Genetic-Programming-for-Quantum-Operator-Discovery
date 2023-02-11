OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.9196148801143642) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
h q[3];
ryy(1.9196148801143642) q[1],q[0];
h q[2];
x q[0];
rz(0.9590763819254728) q[0];
cx q[2],q[1];
cx q[3],q[2];
x q[1];
