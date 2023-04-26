# Base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set working directory
WORKDIR /app

# Install git and python3-pip
RUN apt update && apt install -y git python3-pip

# Clone repository
# RUN git clone <your_repository_url> /app

# Install dependencies
RUN pip3 install sbx-rl
RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip3 install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip3 install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install dm-control
RUN pip3 install wandb
RUN pip3 install tensorboard
RUN pip3 install Box2d
RUN pip3 install earcut 
RUN pip3 install pygame
# RUN pip3 install

ENV WANDB_API_KEY="ea17412f95c94dfcc41410f554ef62a1aff388ab"

# Copy files from local machine to the image
COPY ./analysis_scripts /app/analysis_scripts
COPY ./gym_mxs /app/gym_mxs
COPY ./inertia /app/inertia
COPY ./models /app/models
COPY ./processing_scripts /app/processing_scripts
COPY ./pyaerso /app/pyaerso
COPY ./pymxs_sbx_run.py /app/pymxs_sbx_run.py
COPY ./pymxs_sbx_box.py /app/pymxs_sbx_box.py

RUN cd /app/gym_mxs && pip3 install -e . 

# Set the entrypoint
ENTRYPOINT ["python3", "/app/pymxs_sbx_box.py"]
CMD []
