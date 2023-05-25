# Set base image
FROM --platform=linux/amd64 nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables for user
ENV USER_NAME=sooh
ENV USER_ID=255692
ENV GROUP_NAME=DHLAB-unit
ENV GROUP_ID=11703

# Install sudo
#RUN apt-get update && apt-get install -y sudo

# Install build tools and libraries
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential \
    git curl vim unzip wget tmux screen ca-certificates apt-utils software-properties-common wget && \
    apt-get install -y sudo && \
    apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a group and user
RUN groupadd -g $GROUP_ID $GROUP_NAME
RUN useradd -ms /bin/bash -u $USER_ID -g $GROUP_ID $USER_NAME

# Add new user to sudoers
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Add Conda
ENV CONDA_PREFIX=/home/sooh/.conda
ENV CONDA=/home/sooh/.conda/condabin/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p ${CONDA_PREFIX} && \
    rm miniconda.sh && \
    ${CONDA} config --set auto_activate_base false && \
    ${CONDA} init bash && \
    ${CONDA} create --name myenv -y python=3.8

ENV PATH="/home/sooh/.conda/envs/myenv/bin:$PATH"

RUN /home/sooh/.conda/condabin/conda create -n myenv python=3.8 pip

RUN /home/sooh/.conda/condabin/conda run -n myenv pip install --upgrade pip setuptools

ENV PATH="/home/sooh/.conda/envs/myenv/bin:$PATH"

# Copy app directory
COPY ./historical-adapters /home/$USER_NAME/historical-adapters

# Set the working directory
WORKDIR /home/$USER_NAME/historical-adapters

RUN /home/sooh/.conda/condabin/conda run -n myenv pip install -r requirements.txt

# Login to wandb
ENV WANDB_API_KEY=e5b6f6e2c4975896380d866b44bb47be2943e717 
RUN /home/sooh/.conda/condabin/conda run -n myenv wandb login ${WANDB_API_KEY}

COPY ./entrypoint.sh .

# Change ownership of the copied files to the new user and group
RUN chown -R $USER_NAME:$GROUP_NAME /home/$USER_NAME/historical-adapters

# Switch to the new user
USER $USER_NAME

# Make sure your script is executable
RUN chmod +x entrypoint.sh

# Run run_parallel.sh when the container launches
CMD ["./entrypoint.sh"]
CMD ["sleep", "infinity"]