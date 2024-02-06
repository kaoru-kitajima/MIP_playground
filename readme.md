# python-mip

# on Linux, start X window system
xhost si:localuser:root

# docker build
cd path/to/docker
docker build  -t python-mip

# docker run
sudo docker run --name python-mip --gpus all --env=QT_X11_NO_MITSHM=1 -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /home/kaoru/Software/python-mip:/python-mip:rw --privileged -p 80:80 python-mip
# docker run with nvidia gpu
docker run --name python-mip --gpus all --env=QT_X11_NO_MITSHM=1 -ti --rm -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /home/kaoru/Software/python-mip:/python-mip:rw --privileged -p 80:80 python-mip
