ARG UBUNTU_VER=20.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.9.7


FROM ubuntu:${UBUNTU_VER}

# System packages 
RUN apt-get update && apt-get install -yq curl wget jq vim
RUN apt-get install gcc g++ -y

# Use the above args during building https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

ARG PY_VER
# Install packages from conda and downgrade py (optional).
RUN conda install -c anaconda -y python=${PY_VER}

# Install conda environment
COPY environment.yml .
RUN conda env create -f ./environment.yml

# for predict
WORKDIR /app
COPY predict_example.csv .
COPY examples /app/examples
COPY /predict.py /app
COPY shared /app/shared

SHELL ["conda", "run", "-n", "hh_compete", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "hh_compete", "python", "predict.py"]
