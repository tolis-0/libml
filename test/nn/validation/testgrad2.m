% testgrad2.m
output_precision(10)

learning_rate = 0.03;
k = 3;

w1 = [-0.47, -0.59; -0.23, 0.41];
w2 = [0.33, 0.27; -0.60, 0.56];
b1 = [0.01; -0.01];
b2 = [-0.01; 0.01];

x      = [2.14, 1.76, -1.84; -1.36, 1.45, 2.94];
t      = [1.0, 1.0, 0.0; 0.0, 0.0, 1.0];

nn1(w1, b1, w2, b2, x, t, learning_rate, k)
