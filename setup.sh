docker run -it --gpus all \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd):/app \
    -e PYTHONIOENCODING=utf8 \
    --net=host --rm --name tts_test tts_test /bin/bash;

tritonserver \
  --model-repository=triton \
  --log-verbose 1 \
  --model-control-mode=explicit \
  --load-model=tts_hi_batched;

docker run -it --gpus all \
    -v $(pwd):/app \
    --net=host \
    nvcr.io/nvidia/tritonserver:22.12-py3-sdk;