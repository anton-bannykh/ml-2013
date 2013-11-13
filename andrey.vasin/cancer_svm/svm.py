def get_lagrange_coef(x, y):
    solvers.options['show_progress'] = False
    identity = numpy.identity(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            identity[i][j] = y[i] * y[j] * numpy.inner(x[i], x[j])
    P = matrix(identity)
    tmp = transpose([y])
    A = matrix(tmp)
    b = matrix([0.0])
    q = matrix([-1.0] * len(x))
    G = numpy.zeros(shape=(len(x), len(x)))
    h = matrix([-1.0] * len(x))
    for i in range(len(x)):
            G[i][i] = -1
    G = matrix(transpose(G))
    sol = solvers.qp(P, q, G, h, A, b)['x']
    w = [sol[i] for i in range(len(x[0]))]
    return w