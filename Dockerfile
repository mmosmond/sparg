# copying from https://github.com/binder-examples/minimal-dockerfile

FROM python:3.9-slim
RUN pip install --no-cache notebook jupyterlab
ENV HOME=/tmp

# create user with a home directory
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}

# install dependencies
pip install -r requirements.txt
