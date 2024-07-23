% nn1.m 

function nn1(_w1, _b1, _w2, _b2, x, t, lr, k)
    x1     = transpose(_w1) * x + _b1
    x_relu = max(0, x1)
    x2     = transpose(_w2) * x_relu + _b2
    y      = 1.0 ./ (1.0 + exp(-x2))

    grad = y - t
    grad = (1 - y) .* y .* grad;

    b2 = _b2 - lr * sum(grad, 2) / k
    w2 = _w2 - lr * x_relu * transpose(grad) / k

    grad = (_w2 * grad) .* (x1 > 0);

    b1 = _b1 - lr * sum(grad, 2) / k
    w1 = _w1 - lr * x * transpose(grad) / k
end
