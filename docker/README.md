# build the docker images
sudo docker build -t client -f docker/Dockerfile.client .
sudo docker build -t server -f docker/Dockerfile.server .

# start the docker container
1. On server
sudo docker run -it --rm --gpus all -p 8888:8888/udp server
2. On client
sudo docker run -it --rm -p 8888:8888/udp client