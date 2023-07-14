from enum import StrEnum
import tensorflow as tf
from tensorflow import keras
import keras_cv
from matplotlib import pyplot as plt
import click

from aiimageserving.util import set_mem_limit


class Precision(StrEnum):
    DEFAULT = ''
    HALF = 'float16'
    SINGLE = 'float32'
    MIXED = 'mixed_float16'


class StableDiffusionRunner:
    def __init__(self,
                 width=511,
                 height=512,
                 batch_size=3,
                 device="CPU:0",
                 precision: Precision = None
                 ):
        self._batch_size = batch_size
        self._device = device
        if precision and precision is not Precision.DEFAULT:
            keras.mixed_precision.set_global_policy(str(precision))  # TODO Does this have to be global?
        self._model = keras_cv.models.StableDiffusion(
            img_width=width, img_height=height)

    def run(self, prompt: str):
        with tf.device(self._device):
            return self._model.text_to_image(prompt, batch_size=self._batch_size)


def save_images(images, filename):
    plt.figure(figsize=(30, 10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.savefig(filename)


@click.command()
@click.argument("prompt")
@click.argument("filename")
@click.option("--width", type=int, default=512)
@click.option("--height", type=int, default=512)
@click.option("--batch-size", type=int, default=3)
@click.option("--device", type=str, default="CPU:0")
@click.option("--precision",
              type=click.Choice([e.name for e in Precision]),
              default=Precision.DEFAULT.name)
@click.option("--mem", type=int, default=None)
def main(prompt, filename, width, height, batch_size, device, precision, mem):
    if mem:
        set_mem_limit(device, mem)
    runner = StableDiffusionRunner(width, height, batch_size, device, Precision[precision])
    images = runner.run(prompt)
    save_images(images, filename)


if __name__ == "__main__":
    main()
