import concurrent.futures as futures
import grpc
import proto.service_pb2 as pb2
import proto.service_pb2_grpc as pb2_grpc

class GreeterServicer(pb2_grpc.GreeterServicer):

    def SayHello(self, request, context):
        print("request received for SayHello()")
        name = request.name
        message = "Hello, {}!".format(name)

        return pb2.HelloReply(message=message)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_GreeterServicer_to_server(GreeterServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
