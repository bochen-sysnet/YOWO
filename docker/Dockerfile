FROM ubuntu:18.04

USER root

RUN apt-get update && apt-get -y --no-install-recommends install \
	libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
	gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 \
	gstreamer1.0-qt5 gstreamer1.0-pulseaudio \ 
	ubuntu-restricted-extras \
	libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

RUN apt-get install -y python3.7 python3-pip

RUN apt-get -y install git cmake

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir matplotlib pillow numpy

RUN git clone https://github.com/opencv/opencv.git

WORKDIR opencv

RUN git checkout 4.1.0

RUN mkdir build

WORKDIR build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON_EXECUTABLE=$(which python3) \
-D BUILD_opencv_python2=OFF \
-D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D WITH_GSTREAMER=ON \
-D BUILD_EXAMPLES=ON ..

RUN make && make install && ldconfig

CMD ["python3", "-c", "import cv2; print(cv2.getBuildInformation())"]