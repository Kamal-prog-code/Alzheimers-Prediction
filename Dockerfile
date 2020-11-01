FROM python:3
ENV PYTHONUNBUFFERED 1
WORKDIR /APP
ADD . /APP
COPY ./requirements.txt /APP/requirements.txt
COPY ./passw.txt /APP/passw.txt
RUN pip3 install -r requirements.txt
COPY . /APP

