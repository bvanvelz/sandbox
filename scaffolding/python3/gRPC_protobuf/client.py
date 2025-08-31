# client.py

import grpc
import proto.service_pb2_grpc as pb2_grpc
import proto.service_pb2 as pb2

channel = grpc.insecure_channel('localhost:50051')
stub = pb2_grpc.GreeterStub(channel)

request = pb2.HelloRequest(name='Alice')
response = stub.SayHello(request)

print(response.message)
