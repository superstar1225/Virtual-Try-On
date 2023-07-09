FROM python:3

WORKDIR /app

RUN pip install torch 
RUN pip install torchgeometry 
RUN pip install opencv-python

COPY . .

CMD [ "python", "test.py" ]