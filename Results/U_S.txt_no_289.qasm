OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.6689755552099942) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_46912878743904(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.63660087367265) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_46912878745296(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.63660087367265) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_46912878747120(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.63660087367265) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
rz(0.5337263120218625) q[1];
swap q[0],q[3];
h q[1];
rxx(0.9519096087266208) q[2],q[3];
ryy(0.6689755552099942) q[2],q[0];
cz q[3],q[2];
swap q[2],q[0];
cz q[0],q[2];
ryy_46912878743904(0.63660087367265) q[3],q[2];
ryy_46912878745296(0.63660087367265) q[3],q[2];
ryy_46912878747120(0.63660087367265) q[0],q[3];
rz(0.8786079761994404) q[3];
y q[3];
cx q[1],q[2];
x q[2];
rxx(2.245381491174159) q[1],q[2];
