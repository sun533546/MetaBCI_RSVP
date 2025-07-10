
from socket import *
import time
import binascii

class UDPBroadcaster():
    def __init__(self, port=30419):
        self.port = port
        self.address = ("<broadcast>", self.port)
        # 修正：直接调用 socket 模块的 socket 函数创建 socket 对象
        self.s = socket(AF_INET, SOCK_DGRAM)
        self.s.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)


    def send_data(self,hex_string = "545249473A33"):
        self.hex_string = hex_string 
        self.bytes_data = binascii.unhexlify(self.hex_string)    
        self.s.sendto(self.bytes_data, self.address)
        print(1)
        time.sleep(1)

    def close_connection(self):
        self.s.close()
# 创建UDPBroadcaster类的一个实例
broadcaster = UDPBroadcaster()

# 使用实例发送数据
broadcaster.send_data()

# 关闭连接
broadcaster.close_connection()