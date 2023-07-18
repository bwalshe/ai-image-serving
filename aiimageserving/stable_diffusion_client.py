import tensorflow as tf
from tensorflow_serving.apis import (
    predict_pb2,
    prediction_service_pb2_grpc,
    get_model_status_pb2,
    model_service_pb2_grpc
)
from google.protobuf import json_format
import grpc
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer

MAX_PROMPT_LENGTH = 77


class StableDiffusionClient:
    def __init__(self, endpoint="127.0.0.1:8500"):
        self._endpoint = endpoint
        self._channel = grpc.insecure_channel(endpoint)
        self._model_names = ["text_encoder", "image_encoder", "decoder", "diffusion_model"]
        self._tokenizer = None
        self._prediction_service = \
            prediction_service_pb2_grpc.PredictionServiceStub(self._channel)

    def check_status(self):
        status_service = model_service_pb2_grpc.ModelServiceStub(self._channel)
        responses = {}
        for model in self._model_names:
            status_request = get_model_status_pb2.GetModelStatusRequest()
            status_request.model_spec.name = model
            response = status_service.GetModelStatus(status_request)
            if len(response.model_version_status) != 1:
                raise RuntimeError("The Stable Diffusion server should only ever have one versioneon "
                                   "of each model available.")
            v1_status = response.model_version_status[0]
            responses[model] = json_format.MessageToDict(v1_status)['state']
        return responses

    def _encode_prompt(self, prompt):
        if self._tokenizer is None:
            self._tokenizer = SimpleTokenizer()
        inputs = self._tokenizer.encode(prompt)

        if len(inputs) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt is too long (should be <= {MAX_PROMPT_LENGTH} tokens)"
            )
        phrase = inputs + [49407] * (MAX_PROMPT_LENGTH - len(inputs))
        phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)
        positions = tf.convert_to_tensor(
            [list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32
        )
        req = predict_pb2.PredictRequest()
        req.model_spec.name = "text_encoder"
        req.model_spec.signature_name = "serving_default"
        req.inputs["tokens"].CopyFrom(tf.make_tensor_proto(phrase))
        req.inputs["positions"].CopyFrom(tf.make_tensor_proto(positions))

        response = self._prediction_service.Predict(req)
        return response.outputs["layer_normalization_24"]
