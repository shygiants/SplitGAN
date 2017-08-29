FROM tensorflow/tensorflow:1.3.0-gpu

MAINTAINER Sanghoon Yoon <shygiants@gmail.com>

# Copy all source codes
COPY . /splitgan

# Set working directory
WORKDIR "/splitgan"

RUN mkdir /dataset
RUN mkdir /job-dir

ENTRYPOINT ["bash", "run.sh"]