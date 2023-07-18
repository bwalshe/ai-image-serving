from tensorflow_serving.apis import (
    get_model_status_pb2,
    model_service_pb2_grpc
)
from google.protobuf import json_format
import grpc


class StableDiffusionClient:
    def __init__(self, endpoint="127.0.0.1:8500"):
        self._endpoint = endpoint
        self._channel = grpc.insecure_channel(endpoint)
        self._model_names = ["text_encoder", "image_encoder", "decoder", "diffusion_model"]

    def check_status(self):
        status_service = model_service_pb2_grpc.ModelServiceStub(self._channel)
        responses = {}
        for model in self._model_names:
            status_request = get_model_status_pb2.GetModelStatusRequest()
            status_request.model_spec.name = model
            response = status_service.GetModelStatus(status_request)
            if len(response.model_version_status) != 1:
                raise RuntimeError("The Stable Diffusion server should only ever have one version "
                                   "of each model available.")
            v1_status = response.model_version_status[0]
            responses[model] = json_format.MessageToDict(v1_status)['state']
        return responses
