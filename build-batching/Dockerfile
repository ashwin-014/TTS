FROM nvcr.io/nvidia/tritonserver:23.04-py3

RUN mkdir /app
WORKDIR /app
COPY . /app

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN apt-get update && apt-get install -y libsndfile1 build-essential libssl-dev swig python3-dev python-is-python3
RUN apt install -y libsndfile1-dev ffmpeg enchant

# RUN sh trt_install.sh

RUN pip3 install ai4bharat-transliteration ffmpeg translators indic-numtowords nemo-text-processing pyenchant pydantic pandas ./aksharamukha-python TTS
RUN pip3 install webrtcvad onnxconverter_common setuptools==59.5.0 "accelerate>=0.16.0,<1" "optimum[onnxruntime-gpu]" "transformers[torch]>=4.28.1,<5" "bitsandbytes>=0.31.5" "torch>=1.13.1,<2" torchaudio
RUN pip3 install -r TTS/requirements.txt

# RUN python -m pip install "git+https://github.com/huggingface/optimum.git#egg=optimum[onnxruntime-gpu]"
# RUN git clone https://github.com/libindic/indic-trans.git && cd indic-trans && pip install -r requirements.txt && pip install .
# RUN git clone https://github.com/AI4Bharat/indic-punct.git && cd indic-punct && git checkout light && pip install pynini && pip install -e .
# CMD tritonserver --model-repository=/models/model_repository

CMD [ "sleep", "1000m" ]
# megh@ai4bharat