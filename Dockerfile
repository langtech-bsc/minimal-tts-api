# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Install required packages for building eSpeak and general utilities
RUN apt-get update && apt-get install -y \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        git \
        cmake \ 
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# eSpeak install
RUN git clone -b dev-ca https://github.com/projecte-aina/espeak-ng

RUN pip install --upgrade pip && \
 cd espeak-ng && \
 ./autogen.sh && \
 ./configure --prefix=/usr && \
 make && \
 make install

RUN mkdir -p cache && chmod 777 cache

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app
# Onnx install 

COPY --chown=user requirements.txt $HOME/app/

RUN pip install -r requirements.txt

# download from hf hub
RUN huggingface-cli download projecte-aina/matxa-tts-cat-multiaccent matxa_multiaccent_wavenext_e2e.onnx --local-dir  $HOME/app/
RUN huggingface-cli download projecte-aina/matxa-tts-cat-multiaccent config.yaml--local-dir  $HOME/app/

COPY --chown=user . $HOME/app/

# Fix ownership issues
USER root
RUN chown -R user:user $HOME/app
USER user

EXPOSE 8000

CMD ["python3", "-u", "app.py"]
