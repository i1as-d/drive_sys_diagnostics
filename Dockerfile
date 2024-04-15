FROM ubuntu:20.04

# LABEL about the custom image
LABEL maintainer="admin@sysadminjournal.com"
LABEL version="0.1"
LABEL description="This is custom Docker Image for \
the PHP-FPM and Nginx Services."

# Disable Prompt During Packages Installation
ENV DEBIAN_FRONTEND noninteractive
RUN export DEBIAN_FRONTEND="noninteractive"

# Update Ubuntu Software repository
RUN apt update && apt-get -y install atop

SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["tail", "-f", "/dev/null"]