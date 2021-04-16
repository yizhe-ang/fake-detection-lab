def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def loss(beta, lambda):
    # Make sure values are in the range (0, 1)
    b = sigmoid(beta)
    b = b.reshape(342, 512)

    original = im + b * mask

    # Promote smoothness in image as the prior
    col_diff = original - jnp.roll(original, shift=-1, axis=0)
    row_diff = original - jnp.roll(original, shift=-1, axis=1)

    # Squared error
    reg = (col_diff**2 + row_diff**2).sum()
    # Total variation
    # reg = (col_diff.abs() + row_diff.abs()).sum()

    return lambda * reg

loss_grad = jax.jit(jax.grad(loss, argnums=0))

def loss_grad_wrapper(beta):
    return onp.array(loss_grad(beta))