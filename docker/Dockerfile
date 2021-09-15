FROM ubuntu:18.04

USER root

RUN apt-get update && \
    apt-get install -y python3.6 && \
	apt-get install python3-pip

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir matplotlib pillow 

WORKDIR /home/monet/research/opencv/build

COPY . .

RUN make install

WORKDIR /home/monet/research/YOWO/docker

COPY . .

CMD ["python", "-c", "import cv2; print(cv2.getBuildInformation())"]