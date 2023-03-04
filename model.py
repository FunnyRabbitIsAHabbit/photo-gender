"""
FastAPI server for image processing

Model evaluation file

@author: Stanislav Ermokhin
"""

import tensorflow as tf

from pathlib import Path


__version__ = "4.2.1"
IMAGE_SHAPE = (224, 224, 3)
CLASSES = ["This is a picture of a female", "This is a picture of a male"]
MODEL_BASE_DIR: Path = Path(__file__).parent.resolve(strict=True) / "static"
NEW_PICTURE_PATH: Path = MODEL_BASE_DIR / "artefacts"

directory_name_pattern: str = "myModel"
model_directory: str = f"{MODEL_BASE_DIR}/{directory_name_pattern}-{__version__}"

model = tf.keras.models.load_model(model_directory)


@tf.function
def load_image(filename):
    raw = tf.io.read_file(filename)
    image = tf.image.decode_image(raw,
                                  channels=IMAGE_SHAPE[2],
                                  expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image,
                            (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    image = tf.expand_dims(image, axis=0)

    return image


def predict_pipeline(filename, threshold=0.649):
    input_image = load_image(filename)
    prediction = model(input_image).numpy()[0][0]
    prediction = 0 if prediction - threshold < 0 else 1

    return CLASSES[prediction]