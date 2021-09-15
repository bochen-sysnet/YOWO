FROM python:3

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir nibabel pydicom matplotlib pillow && \
    pip install --no-cache-dir med2image

WORKDIR /app

COPY . .

CMD ["cat", "/etc/os-release"]

CMD ["python", "--version"]

CMD ["python", "-c", "import cv2; print(cv2.__version__)"]