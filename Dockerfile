FROM gcc:10.2
MAINTAINER Anton Betten "Anton.Betten@colostate.edu"

RUN apt-get update

RUN cd /opt && \
    git clone https://github.com/abetten/orbiter && \
    cd /opt/orbiter && \
    make -f makefile clean && \
    make -f makefile && \
    make install

ENV PATH="/opt/orbiter/bin:${PATH}"
