FROM nvcr.io/nvidia/tensorrt:20.12-py3

WORKDIR /app/
EXPOSE 8000

RUN apt-get update && apt-get install -y libcusolver10

COPY requirements.txt ./

RUN pip3 install --upgrade setuptools pip
RUN pip3 install --ignore-installed -r requirements.txt 

COPY . ./

CMD ["python3", "export.py"]
