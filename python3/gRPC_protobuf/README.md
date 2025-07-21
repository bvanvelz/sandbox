# About
Scaffolding for hello-world of gRPC/protobuf API service.


# Setup
```
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/service.proto
```

# Run
Start Server
```
source env/bin/activate
python service.py
```

Run client
```
source env/bin/activate
python client.py
```