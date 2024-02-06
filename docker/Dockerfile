FROM python:3.10

RUN apt-get update

RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y mesa-utils x11-apps xauth
RUN pip install --upgrade pip

RUN mkdir -p /root/src
COPY ./requirements.txt /root/src
WORKDIR /root/src

RUN pip install -r requirements.txt

ARG USERNAME=kaoru
ARG GROUPNAME=kaoru
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME

USER ${USERNAME}

CMD [ "/bin/bash" ]