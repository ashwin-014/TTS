FROM nvcr.io/nvidia/tritonserver:22.12-py3

RUN mkdir /app
WORKDIR /app
COPY . /app

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN apt-get update && apt-get install -y libsndfile1 build-essential libssl-dev swig python3-dev python-is-python3

RUN pip3 install setuptools==59.5.0 "torch>=1.13.1,<2" torchaudio
RUN pip3 install -r requirements.txt

CMD [ "sleep", "1000m" ]