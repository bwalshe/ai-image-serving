import tensorflow as tf
from aiimageserving.stable_diffusion_client import StableDiffusionClient


def test_check_status():
    client = StableDiffusionClient()
    status = client.check_status()
    assert len(status) == 4
    assert set(status.values()) == {"AVAILABLE"}


def test_text_encoding():
    client = StableDiffusionClient()
    encoding = client.encode_prompt("this text doesn't really matter")
    encoding_t = tf.make_ndarray(encoding)
    assert encoding_t.shape == (1, 77, 768)


def test_generate_image():
    client = StableDiffusionClient()
    encoding = client.encode_prompt("A cute puppy")
    encoding_t = tf.constant(tf.make_ndarray(encoding))
    result = client.generate_image(encoding_t, num_steps=5, batch_size=2)
    assert result.shape == (2, 256, 256, 3)
