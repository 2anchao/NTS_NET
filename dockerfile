FROM nvcr.io/nvidia/pytorch:21.03-py3
LABEL maintainer = "AC"
# Install linux packages
RUN apt update && apt install -y zip htop psmisc

RUN pip install seaborn==0.11.1 \
                pytorch_ignite==0.4.4 \
                tensorboardX==2.1 \
                Python-Deprecated \
                torchcontrib \
                pandas \
                opencv_python \
                yacs matplotlib \
                numpy \
                Pillow \
                pretrainedmodels \
                protobuf \
                pycocotools \
                rfconv \
                scikit_learn -i https://pypi.tuna.tsinghua.edu.cn/simple


