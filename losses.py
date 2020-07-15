import tensorflow as tf

from utils import fp32, scale_loss, custom_unscale_grads_in_mixed_precision


@tf.function
def G_loss_fn(fake_scores, write_summary, step):
    loss = -tf.reduce_mean(fake_scores)
    if write_summary:
        with tf.name_scope('G_WGAN-GP'):
            tf.summary.scalar('Loss', loss, step=step)
    return loss


@tf.function
def D_loss_fn(D_model, optimizer, mixed_precision, real_scores, real_images, fake_scores, fake_images,
              write_summary, step,
              wgan_lambda=10.0,  # Weight for the gradient penalty term
              wgan_epsilon=0.001,  # Weight for the epsilon term, \epsilon_{drift}
              wgan_target=1.0  # Target value for gradient magnitudes
              ):
    fake_part_loss = tf.reduce_mean(fake_scores)
    real_part_loss = tf.reduce_mean(real_scores)
    loss = fake_part_loss - real_part_loss

    batch_size = real_scores.get_shape().as_list()[0]

    # Gradient penalty
    alpha = tf.random.uniform(
        shape=[batch_size, 1, 1, 1], minval=0., maxval=1.0, dtype=real_images.dtype
    )
    inter_samples = alpha * real_images + (1. - alpha) * fake_images
    with tf.GradientTape(watch_accessed_variables=False) as tape_gp:
        tape_gp.watch(inter_samples)
        inter_samples_loss = tf.reduce_sum(fp32(D_model(inter_samples)))
        inter_samples_loss = scale_loss(optimizer, inter_samples_loss, mixed_precision)
    gp_grads = tape_gp.gradient(inter_samples_loss, inter_samples)
    # Default grads unscaling doesn't work inside this function,
    # though it is ok to use it inside train steps
    if mixed_precision:
        gp_grads = custom_unscale_grads_in_mixed_precision(optimizer, gp_grads, inter_samples)
    gp_grads_norm = tf.sqrt(
        tf.reduce_sum(tf.square(gp_grads), axis=[1, 2, 3])
    )
    grads_penalty = tf.reduce_mean((gp_grads_norm - wgan_target) ** 2)
    loss += wgan_lambda * grads_penalty

    # Epsilon penalty
    epsilon_penalty = tf.reduce_mean(tf.square(real_scores))
    loss += wgan_epsilon * epsilon_penalty

    if write_summary:
        with tf.name_scope('D_WGAN-GP'):
            tf.summary.scalar('FakePartLoss', fake_part_loss, step=step)
            tf.summary.scalar('RealPartLoss', real_part_loss, step=step)
            tf.summary.scalar('GradsPenalty', grads_penalty, step=step)
            tf.summary.scalar('EpsilonPenalty', epsilon_penalty, step=step)
            tf.summary.scalar('TotalLoss', loss, step=step)

    return loss
