from aiimageserving.stable_diffusion_client import StableDiffusionClient


def test_check_status():
    client = StableDiffusionClient()
    status = client.check_status()
    assert len(status) == 4
    assert set(status.values()) == {"AVAILABLE"}
