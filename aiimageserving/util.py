import tensorflow as tf


def set_mem_limit(device_name: str, limit: int) -> None:
    """Manually set the memory limit on a physical device. """
    """This is needed on my system, at least, as I have observed that """
    """Tensorflow can be over-conservative on how much memory it can """
    """take from my GPU"""

    devices = [d for d in tf.config.list_physical_devices()
               if d.name == f"/physical_device:{device_name}"]

    if len(devices) != 1:
        names = [d.name for d in tf.config.list_physical_devices()]
        raise RuntimeError(f"Unknown device: ${device_name}. "
                           f"Valid options are {names}")

    tf.config.set_logical_device_configuration(
        devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=limit)]
    )
