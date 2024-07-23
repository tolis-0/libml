% testgrad1.m
output_precision(10)

learning_rate = 0.03;
k = 3;

w1 = [1.3, -1.1; -0.9, 0.4];
w2 = [-0.7, 0.3; -1.6, 2.0];
b1 = [-0.1; 0.1];
b2 = [0.25; 0.15];

x      = [1.6, -1.0, 0.5 ;-0.5, 1.0, 0.4];
t      = [1.0, 0.0, 1.0; 0.0, 1.0, 0.0];

nn1(w1, b1, w2, b2, x, t, learning_rate, k)
