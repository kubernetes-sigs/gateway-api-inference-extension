# Serving Multimodal Models

This guide explains how to configure the Inference Gateway to serve multimodal models, which can process and generate content across different modalities like text, images, audio, and video.

!!! Note about optimizations
    The current implementation of multimodal support is not optimized in the IGW. Optimizations relating for multimodal models are planned in future releases, for example [multimodal prefix cache aware routing](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/2172).

Multimodal models are becoming increasingly prevalent in AI, enabling applications that can understand and interact with the world in more human-like ways. For example, a single model might be able to answer questions about an image, transcribe audio, or generate video descriptions. The Inference Gateway provides a flexible and efficient way to deploy and manage these complex models within your Kubernetes cluster.

## How

Presently, The Inference Gateway (IGW) supports multimodal models by the Open AI's ChatCompletions API. The Endpoint Picker (EPP) component of the IGW is designed to parse these structured requests, extract the multimodal content (e.g., image URLs, audio data, video URLs), and route them to the appropriate backend inference server capable of processing such inputs. This allows for a unified API endpoint for diverse multimodal workloads, simplifying client-side integration.

## Prerequisites & Setup

This guide assumes you have completed the Getting Started guide and have a functional Inference Gateway setup.

You will need to deploy a multimodal model server that is capable of handling the desired input modalities (image, video, audio). The specific deployment manifest for your multimodal model server will depend on the model and its serving framework.

### Example deployment

To be added

## Try the setup

Once your multimodal model server, `InferencePool`, and EPP are deployed and running, you can test the setup by sending requests with different multimodal content.

First, ensure you have the `IP` and `PORT` of your Inference Gateway:

```bash
export IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses.value}')
export PORT=80 # Or your gateway's port
```

### Image Input

Send a request with an image URL and a text prompt. The model `your-multimodal-image-model` should be the name of the model configured in your `InferencePool` to handle image inputs.

```bash
$ curl ${IP}:${PORT}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "your-multimodal-image-model",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image? Describe it briefly."
        },
        {
          "type": "image_url",
    }
  ]
}'
```
Expected output: The model should return a text description of the image.

### Video Input

Send a request with a video URL. The model `your-multimodal-video-model` should be configured to handle video inputs.

```bash
$ curl ${IP}:${PORT}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "your-multimodal-video-model",
  "messages": [
    {
          "video_url": {
            "url": "https://example.com/video.mp4"
          }
        },
        {
            "type": "text",
            "text": "Summarize the key steps shown in this video."
        }
      ]
    }
  ]
}'
```
Expected output: The model should return a text summary of the video content.

### Audio Input

Send a request with base64 encoded audio data. The model `your-multimodal-audio-model` should be configured to handle audio inputs.

```bash
$ curl ${IP}:${PORT}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "your-multimodal-audio-model",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_audio",
          "input_audio": {
            "data": "base64_encoded_audio_data_here",
            "format": "mp3"
          }
        },
        {
            "type": "text",
            "text": "Transcribe this audio and identify the speaker."
        }
      ]
    }
  ]
}'
```