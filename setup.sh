docker run -it --gpus all \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd):/app \
    -e PYTHONIOENCODING=utf8 \
    --net=host --rm --name tts_test tts_test /bin/bash;

tritonserver \
  --model-repository=/priyam-workspace/tts/tts-benchmarking/models-repository \
  --log-verbose 1 \
  --model-control-mode=explicit \
  --load-model=tts_tensorrt_decomposed;

docker run -it --gpus all \
    -v /datadrive/priyam-workspace/tts/:/priyam-workspace/tts \
    --net=host \
    nvcr.io/nvidia/tritonserver:22.12-py3-sdk;
