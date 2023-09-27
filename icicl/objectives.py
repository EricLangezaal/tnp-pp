def iccnp_objective(model, x_c, y_c, x_t, y_t, d_c):
    pred_y_t = model(x_c, y_c, x_t, d_c)
    loglik = pred_y_t.log_prob(y_t).sum()

    return -loglik


def cnp_objective(model, x_c, y_c, x_t, y_t):
    pred_y_t = model(x_c, y_c, x_t)
    loglik = pred_y_t.log_prob(y_t).sum()

    return -loglik
