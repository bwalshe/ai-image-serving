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
                 width=512,
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

    def text_to_image(self, prompt: str):
        with tf.device(self._device):
            return self._model.text_to_image(prompt, batch_size=self._batch_size)

    def export_model(self, model_dir: str) -> None:
        with tf.device(self._device):
            self._model.text_encoder.save(f"{model_dir}/text_encoder/1")
            self._model.image_encoder.save(f"{model_dir}/image_encoder/1")
            self._model.decoder.save(f"{model_dir}/decoder/1")
            self._model.diffusion_model.save(f"{model_dir}/diffusion_model/1")


def save_images(images, filename):
    plt.figure(figsize=(30, 10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.savefig(filename)


@click.group()
@click.option("--width", type=int, default=512, help="Width in pixels of the individual images in the batch.")
@click.option("--height", type=int, default=512, help="height in pixels of the individual images in the batch.")
@click.option("--batch-size", type=int, default=3, help="number of images to generate.")
@click.option("--device", type=str, default="CPU:0", help="physical device to use for tensor operations.")
@click.option("--precision",
              type=click.Choice([e.name for e in Precision]),
              default=Precision.DEFAULT.name)
@click.option("--mem", type=int, default=None, help="Manually specify how much memory to allocate on the device.")
@click.pass_context
def cli(ctx, width, height, batch_size, device, precision, mem):
    if mem:
        set_mem_limit(device, mem)
    ctx.ensure_object(dict)
    ctx.obj["runner"] = StableDiffusionRunner(width, height, batch_size, device, Precision[precision])


@cli.command
@click.pass_context
@click.argument("prompt")
@click.argument("filename")
def text_to_image(ctx, prompt, filename):
    """Generate a PNG format image from PROMPT and save it to FILENAME"""
    runner = ctx.obj["runner"]
    images = runner.text_to_image(prompt)
    save_images(images, filename)


@cli.command
@click.pass_context
@click.argument("base_dir")
def export(ctx, base_dir):
    """Export the components of the Stable Diffusion model to BASE_DIR"""
    ctx.obj["runner"].export_model(base_dir)


if __name__ == "__main__":
    cli(obj={})
