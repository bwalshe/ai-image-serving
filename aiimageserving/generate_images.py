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


def save_images(images, filename):
    plt.figure(figsize=(30, 10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.savefig(filename)


def run_stable_diffusion(prompt: str, width=512, height=512, batch_size=3,
                         device="CPU:0", precision: Precision = None):
    if precision and precision is not Precision.DEFAULT:
        keras.mixed_precision.set_global_policy(str(precision))

    model = keras_cv.models.StableDiffusion(
        img_width=width, img_height=height)

    with tf.device(device):
        return model.text_to_image(prompt, batch_size=batch_size)


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
    images = run_stable_diffusion(prompt, width, height, batch_size,
                                  device, Precision[precision])
    save_images(images, filename)


if __name__ == "__main__":
    main()
