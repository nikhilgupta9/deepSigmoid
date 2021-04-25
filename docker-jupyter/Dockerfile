# install base image
FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libjpeg8-dev
    
RUN ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
RUN ln -s /usr/lib/x86_64-linux-gnu/libfreetype.so /usr/lib
RUN ln -s /usr/lib/x86_64-linux-gnu/libz.so /usr/lib

RUN pip install --no-cache-dir -I pillow

RUN conda install jupyter -y

WORKDIR /root

RUN mkdir /notebooks

WORKDIR /notebooks

# running and configuring jupyter
RUN jupyter notebook --generate-config

ENV CONFIG_PATH="/root/.jupyter/jupyter_notebook_config.py"

COPY "jupyter_notebook_config.py" ${CONFIG_PATH}

ENTRYPOINT ["sh", "-c", "jupyter notebook --allow-root -y --no-browser --ip=0.0.0.0 --config=${CONFIG_PATH}"]



