import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs Available: ", [gpu.name for gpu in gpus])
else:
    print("No GPUs found")
