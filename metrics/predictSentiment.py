import tensorflow as tf
import numpy as np

MODEL_SAVE_ADDR = './model/amazon'
reloaded_model = tf.keras.models.load_model(MODEL_SAVE_ADDR)

prediction = reloaded_model.predict(
    [
     """ great, totally buy again """
    ]
)

print(prediction)
print('predicted class: {0}'.format(np.argmax(prediction)))
