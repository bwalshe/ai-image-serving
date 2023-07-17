from enum import StrEnum
import tensorflow as tf
from tensorflow import keras
from tensorflow_serving.config import model_server_config_pb2
import keras_cv
from matplotlib import pyplot as plt
import click
from pathlib import Path
from google.protobuf import text_format

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

    def export_model(self, local_model_dir: str) -> None:
        model_dir = Path(local_model_dir)
        model_dir.mkdir(exist_ok=True, parents=True)
        sd_config = model_server_config_pb2.ModelServerConfig()

        def add_model(model_name, model):
            config = sd_config.model_config_list.config.add()
            config.name = model_name
            config.base_path = f"/models/{model_name}"
            config.model_platform = "tensorflow"
            model.save(model_dir / model_name / "1")

        with tf.device(self._device):
            add_model("text_encoder", self._model.text_encoder)
            add_model("image_encoder", self._model.image_encoder)
            add_model("decoder", self._model.decoder)
            add_model("diffusion_model", self._model.diffusion_model)

        with (model_dir / "models.config").open("w") as config_file:
            config_file.write(text_format.MessageToString(sd_config))


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
