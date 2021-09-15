FROM python:3.7

USER root

RUN apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
RUN apt install ubuntu-restricted-extras
RUN apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir matplotlib pillow numpy

WORKDIR $HOME/research/opencv/build

COPY . .

RUN make install

CMD ["python", "-c", "import cv2; print(cv2.getBuildInformation())"]