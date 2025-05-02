import numpy as np


def anderson_mixing(g, w0, m=5, max_iter=100, tol=1e-6):
    past_w = []
    past_residuals = []
    w = w0.copy()

    for _ in range(max_iter):
        g_w = g(w)
        f = g_w - w

        if np.linalg.norm(f) < tol:
            break

        past_w.append(g_w)
        past_residuals.append(f)

        # Keep only last p elements
        if len(past_w) > m:
            past_w.pop(0)
            past_residuals.pop(0)

        # Only start mixing after we have some history
        if len(past_residuals) > 1:
            # Build the matrix of residual differences
            F = np.array(past_residuals[1:]).T - np.array(past_residuals[:-1]).T

            # Solve the least squares problem
            gamma = np.linalg.lstsq(F, -past_residuals[-1], rcond=None)[0]

            # Compute weights (beta)
            beta = np.zeros(len(past_residuals))
            beta[-1] = 1.0
            beta[:-1] = gamma

            # Compute next w as weighted combination
            w = w = np.dot(beta, np.array(past_w))
        else:
            w = g_w

    return w
