import tensorflow as tf

def huber_loss(x, delta=1.0):
  """Apply the function:
  ```
  0.5*x^2 if |x| < delta else delta*(|x| - 0.5*delta)
  ```
  """
  abs_x = tf.abs(x)
  return tf.where(
    abs_x < delta,
    tf.square(x) * 0.5,
    delta * (abs_x - 0.5 * delta)
  )
