FROM python:3

USER root

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir matplotlib pillow 

WORKDIR /home/monet/research/opencv/build

RUN make && make install

WORKDIR /home/monet/research/YOWO/docker

COPY . .

CMD ["python", "-c", "import cv2; print(cv2.getBuildInformation())"]