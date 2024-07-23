% nn2.m

function nn2(_w1, _b1, _w2, _b2, x, t, lr, k)
    x1      = transpose(_w1) * x + _b1
    x_lrelu = max(x1, 0.01 * x1)
    x2      = transpose(_w2) * x_lrelu + _b2
    y       = tanh(x2)

    grad = y - t
    grad = (1 - y .* y) .* grad;

    b2 = _b2 - lr * sum(grad, 2) / k
    w2 = _w2 - lr * x_lrelu * transpose(grad) / k

    grad = (_w2 * grad) .* ((x1 > 0) + 0.01 * (x1 <= 0));

    b1 = _b1 - lr * sum(grad, 2) / k
    w1 = _w1 - lr * x * transpose(grad) / k
end
