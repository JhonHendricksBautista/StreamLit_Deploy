from keras.models import load_model
import tensorflow as tf

model = load_model("bagongDahon.keras")


# Recreate model architecture
new_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=model.output
)

new_model.save("model.h5")