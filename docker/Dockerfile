FROM python:3.7

USER root

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir matplotlib pillow numpy

WORKDIR $HOME/research/opencv/build

COPY . .

RUN make install

CMD ["python", "-c", "import cv2; print(cv2.getBuildInformation())"]