# Serving Multimodal Models

This guide explains how to configure the Inference Gateway to serve multimodal models, which can process and generate content across different modalities like text, images, audio, and video.

!!! Note about optimizations
    The current implementation of multimodal support is not optimized in the IGW. Optimizations relating for multimodal models are planned in future releases, for example [multimodal prefix cache aware routing](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/2172). Presently, disabling prefix cache aware routing delivers better results.

Multimodal models are becoming increasingly prevalent in AI, enabling applications that can understand and interact with the world in more human-like ways. For example, a single model might be able to answer questions about an image, transcribe audio, or generate video descriptions.

## How

Presently, The Inference Gateway (IGW) supports multimodal models by the Open AI's ChatCompletions API.

## Prerequisites & Setup

This guide assumes you have completed the Getting Started guide and have a functional Inference Gateway setup. Follow the steps at [getting-started](https://gateway-api-inference-extension.sigs.k8s.io/guides/) to learn how to setup IGW.

You will need to deploy a multimodal model server that is capable of handling the desired input modalities (image, video, audio). The specific deployment manifest for your multimodal model server will depend on the model and its serving framework. A [sample deployment yaml is here](https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/heads/main/config/manifests/vllm/multimodal-gpu-deployment.yaml)


## Try the setup

Once your multimodal model server, `InferencePool`, and EPP are deployed and running, you can test the setup by sending requests with different multimodal content.

First, ensure you have the `IP` and `PORT` of your Inference Gateway:

```bash
export IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses.value}')
export PORT=80 # Or your gateway's port
```

### Image Input

Send a request with an image URL and a text prompt. The model `Qwen/Qwen2-VL-2B-Instruct` should be the name of the model configured in your `InferencePool` to handle image inputs.

```bash
curl ${IP}:${PORT}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2-VL-2B-Instruct",
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
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gnome-face-smile.svg/1200px-Gnome-face-smile.svg.png"
          }
        }
      ]
    }
  ]
}'
```
Expected output: The model should return a text description of the image.

### Video Input

Send a request with a video URL. The model `Qwen/Qwen2-VL-2B-Instruct` should be configured to handle video inputs.

```bash
curl ${IP}:${PORT}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2-VL-2B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "video_url",
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

More details with an audio model to be added once [#2525](https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/2525) is merged.



