import math
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import (
    predict_pb2,
    get_model_metadata_pb2,
    prediction_service_pb2_grpc,
    get_model_status_pb2,
    model_service_pb2_grpc
)
from google.protobuf import json_format
import grpc
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.constants import _ALPHAS_CUMPROD
from keras_cv.models.stable_diffusion.constants import _UNCONDITIONAL_TOKENS


class StableDiffusionClient:
    def __init__(self, endpoint="127.0.0.1:8500"):
        self._endpoint = endpoint
        self._channel = grpc.aio.insecure_channel(endpoint)
        self._model_names = ["text_encoder", "image_encoder", "decoder", "diffusion_model"]
        self._tokenizer = None
        self._prediction_service = \
            prediction_service_pb2_grpc.PredictionServiceStub(self._channel)
        self._init_diffusion_metadata()
        self._init_text_metadata()

    def _init_diffusion_metadata(self):
        sync_channel = grpc.insecure_channel(self._endpoint)
        service = prediction_service_pb2_grpc.PredictionServiceStub(sync_channel)
        req = get_model_metadata_pb2.GetModelMetadataRequest()
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        req.model_spec.name = "diffusion_model"
        req.metadata_field.append("signature_def")
        response = service.GetModelMetadata(req)
        signature_map.ParseFromString(response.metadata["signature_def"].value)
        model_inputs = signature_map.signature_def["serving_default"].inputs
        latent_inputs = model_inputs["input_5"]
        self._latent_height = latent_inputs.tensor_shape.dim[1].size
        self._latent_width = latent_inputs.tensor_shape.dim[2].size

    def _init_text_metadata(self):
        sync_channel = grpc.insecure_channel(self._endpoint)
        service = prediction_service_pb2_grpc.PredictionServiceStub(sync_channel)
        req = get_model_metadata_pb2.GetModelMetadataRequest()
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        req.model_spec.name = "text_encoder"
        req.metadata_field.append("signature_def")
        response = service.GetModelMetadata(req)
        signature_map.ParseFromString(response.metadata["signature_def"].value)
        model_inputs = signature_map.signature_def["serving_default"].inputs
        self._max_prompt_length = model_inputs["tokens"].tensor_shape.dim[1].size

    async def check_status(self):
        status_service = model_service_pb2_grpc.ModelServiceStub(self._channel)
        responses = {}
        for model in self._model_names:
            status_request = get_model_status_pb2.GetModelStatusRequest()
            status_request.model_spec.name = model
            response = await status_service.GetModelStatus(status_request)
            if len(response.model_version_status) != 1:
                raise RuntimeError("The Stable Diffusion server should only ever have one version "
                                   "of each model available.")
            v1_status = response.model_version_status[0]
            responses[model] = json_format.MessageToDict(v1_status)['state']
        return responses

    async def encode_prompt(self, prompt):
        if self._tokenizer is None:
            self._tokenizer = SimpleTokenizer()
        inputs = self._tokenizer.encode(prompt)
        if len(inputs) > self._max_prompt_length:
            raise ValueError(
                f"Prompt is too long (should be <= {self._max_prompt_length} tokens)"
            )
        phrase = inputs + [49407] * (self._max_prompt_length - len(inputs))
        phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)
        return await self._encode_tokens(phrase)

    async def _encode_tokens(self, phrase):

        positions = tf.convert_to_tensor(
            [list(range(self._max_prompt_length))], dtype=tf.int32
        )
        req = predict_pb2.PredictRequest()
        req.model_spec.name = "text_encoder"
        req.model_spec.signature_name = "serving_default"
        req.inputs["tokens"].CopyFrom(tf.make_tensor_proto(phrase))
        req.inputs["positions"].CopyFrom(tf.make_tensor_proto(positions))

        response = await self._prediction_service.Predict(req)
        return response.outputs["layer_normalization_24"]

    async def text_to_image(self, prompt, batch_size=1, num_steps=50, unconditional_guidance_scalar=7.5):
        encoded_text = await self.encode_prompt(prompt)
        return await self.generate_image(tf.make_ndarray(encoded_text), batch_size, num_steps, unconditional_guidance_scalar)

    async def generate_image(self, encoded_text, batch_size=1, num_steps=50, unconditional_guidance_scale=7.5):
        encoded_text = tf.cast(encoded_text, tf.float32)
        context = tf.make_tensor_proto(self._expand_tensor(encoded_text, batch_size))
        unconditional_context = tf.make_tensor_proto(tf.repeat(
            await self._get_unconditional_context(), batch_size, axis=0
        ))
        timesteps = tf.range(1, 1000, 1000 // num_steps)
        alphas, alphas_prev = self._get_initial_alphas(timesteps)
        latent = tf.random.normal(
            (batch_size, self._latent_height, self._latent_width, 4)
        )

        iteration = 0
        for index, timestep in list(enumerate(timesteps))[::-1]:
            latent_prev = latent  # Set aside the previous latent vector
            t_emb = tf.make_tensor_proto(self._get_timestep_embedding(timesteps[index], batch_size))
            req = predict_pb2.PredictRequest()
            req.model_spec.name = "diffusion_model"
            req.model_spec.signature_name = "serving_default"
            req.inputs["input_3"].CopyFrom(unconditional_context)
            req.inputs["input_5"].CopyFrom(tf.make_tensor_proto(latent))
            req.inputs["input_4"].CopyFrom(t_emb)
            response = await self._prediction_service.Predict(req)
            unconditional_latent = tf.make_ndarray(response.outputs["padded_conv2d_151"])
            req.inputs["input_3"].CopyFrom(context)
            response = await self._prediction_service.Predict(req)
            latent = tf.make_ndarray(response.outputs["padded_conv2d_151"])
            latent = unconditional_latent + unconditional_guidance_scale * (
                    latent - unconditional_latent
            )

            a_t, a_prev = alphas[index], alphas_prev[index]
            pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(
                a_t
            )
            latent = (
                    latent * math.sqrt(1.0 - a_prev) + math.sqrt(a_prev) * pred_x0
            )
            iteration += 1

        decode_req = predict_pb2.PredictRequest()
        decode_req.model_spec.name = "decoder"
        decode_req.model_spec.signature_name = "serving_default"
        decode_req.inputs["input_2"].CopyFrom(tf.make_tensor_proto(latent))
        response = await self._prediction_service.Predict(decode_req)
        decoded = tf.make_ndarray(response.outputs["padded_conv2d_67"])
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")

    @staticmethod
    def _get_timestep_embedding(
            timestep, batch_size, dim=320, max_period=10000
    ):
        half = dim // 2
        freqs = tf.math.exp(
            -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return tf.repeat(embedding, batch_size, axis=0)

    def _get_initial_alphas(self, timesteps):
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        return alphas, alphas_prev

    def _expand_tensor(self, text_embedding, batch_size):
        """Extends a tensor by repeating it to fit the shape of the given batch
        size."""
        text_embedding = tf.squeeze(text_embedding)
        if text_embedding.shape.rank == 2:
            text_embedding = tf.repeat(
                tf.expand_dims(text_embedding, axis=0), batch_size, axis=0
            )
        return text_embedding

    async def _get_unconditional_context(self):
        unconditional_tokens = tf.convert_to_tensor(
            [_UNCONDITIONAL_TOKENS], dtype=tf.int32
        )
        return tf.cast(tf.constant(
            tf.make_ndarray(await self._encode_tokens(unconditional_tokens))
        ), tf.float32)
