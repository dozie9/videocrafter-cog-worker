ARG BASE_IMAGE=runpod/pytorch:3.10-2.0.0-117
FROM ${BASE_IMAGE} as dependencies

# Install necessary packages and Python 3.10
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends software-properties-common curl git openssh-server && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends python3.10 python3.10-dev python3.10-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 &&\
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py

# Create a virtual environment
RUN python3 -m venv /opt/venv

COPY requirements.txt .

# Install runpod within the virtual environment
RUN /opt/venv/bin/pip install -r requirements.txt

ADD src/handler.py /rp_handler.py

CMD ["/opt/venv/bin/python3", "-u", "/rp_handler.py"]
