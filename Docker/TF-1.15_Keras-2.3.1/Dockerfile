FROM tensorflow/tensorflow:1.15.5-gpu-py3

RUN useradd -rm -d /home/mcal -s /bin/bash -g root -G sudo -p mcal mcal
#USER mcal

RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN rm requirements.txt
#RUN cd /home/mcal
#RUN git clone https://github.com/hangqiu/ActiveLabeling.git

WORKDIR /ActiveLabeling
