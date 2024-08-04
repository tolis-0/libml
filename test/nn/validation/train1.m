% train1.m
output_precision(10)

lr = 0.02;
e = 3;
k = 2;
s = 6;

num_batches = s / k;

w1 = [-0.47, -0.59; -0.23, 0.41];
w2 = [0.33, 0.27; -0.60, 0.56];
b1 = [0.01; -0.01];
b2 = [-0.01; 0.01];

x      = [2.14, 1.76, -1.84,  1.6, -1.0, 0.5;
         -1.36, 1.45,  2.94, -0.5,  1.0, 0.4];
t      = [1.0,  1.0,   0.0,   1.0,  0.0, 1.0;
          0.0,  0.0,   1.0,   0.0,  1.0, 0.0];

% Training the neural network
for i = 1:e
    for j = 1:num_batches
        batch_start = (j-1) * k + 1;
        batch_end   = j * k;

        x_batch = x(:, batch_start:batch_end);
        t_batch = t(:, batch_start:batch_end);

        x1      = transpose(w1) * x_batch + b1;
        x_lrelu = max(x1, 0.01 * x1);
        x2      = transpose(w2) * x_lrelu + b2;
        y       = 1.0 ./ (1.0 + exp(-x2));

        grad = y - t_batch;
        grad = (1 - y) .* y .* grad;

        b2 -= lr * sum(grad, 2) / k;
        w2 -= lr * x_lrelu * transpose(grad) / k;

        grad = (w2 * grad) .* ((x1 > 0) + 0.01 * (x1 <= 0));

        b1 -= lr * sum(grad, 2) / k;
        w1 -= lr * x_batch * transpose(grad) / k;
    end
end

ub = [transpose(b1), transpose(b2)]
uw = [transpose(w1(:)); transpose(w2(:))]
