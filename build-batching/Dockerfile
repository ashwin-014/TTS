FROM nvcr.io/nvidia/tritonserver:22.12-py3

RUN mkdir /app
WORKDIR /app
COPY . /app

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN apt-get update && apt-get install -y libsndfile1 build-essential libssl-dev swig python3-dev python-is-python3

# RUN sh trt_install.sh

RUN pip3 install webrtcvad setuptools==59.5.0 "accelerate>=0.16.0,<1" "optimum[onnxruntime-gpu]" "transformers[torch]>=4.28.1,<5" "bitsandbytes>=0.31.5" "torch>=1.13.1,<2" torchaudio
RUN pip3 install -r requirements.txt
# RUN cd /tmp && wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.tar.gz &&\
#     tar -zxvf cmake-3.25.1-linux-x86_64.tar.gz
# RUN git clone https://github.com/nvidia-riva/riva-asrlib-decoder &&\
#     cd riva-asrlib-decoder && git submodule update --init --recursive &&\
#     mkdir build && cd build &&\
#     /tmp/cmake-3.25.1-linux-x86_64/bin/cmake -DRIVA_ASRLIB_BUILD_PYTHON_BINDINGS=NO .. &&\
#     make -j 2
# COPY swig /home/swig
# RUN cd /home/swig && bash setup.sh
RUN python -m pip install "git+https://github.com/huggingface/optimum.git#egg=optimum[onnxruntime-gpu]"
# RUN git clone https://github.com/libindic/indic-trans.git && cd indic-trans && pip install -r requirements.txt && pip install .
# RUN git clone https://github.com/AI4Bharat/indic-punct.git && cd indic-punct && git checkout light && pip install pynini && pip install -e .
# CMD tritonserver --model-repository=/models/model_repository

CMD [ "sleep", "1000m" ]
# megh@ai4bharat