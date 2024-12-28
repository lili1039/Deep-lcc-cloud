import asyncio
import websockets
import pickle
import time

Port = 3394
# host_ip = '8.130.125.99'
host_ip = '172.17.45.246' # 服务器私网ip
 
async def server_recv(websocket):
    """ 接收从客户端发来的消息并处理，再返给客户端ok。
    """
    # while True:
    #     try:
    #         bytes_recv = await websocket.recv()
    #         msg_recv = pickle.loads(bytes_recv)
    #         print('the msg is ',msg_recv[0],msg_recv[1])

    #         msg_send = "ok!!!"
    #         bytes_msg_send = pickle.dumps(msg_send)
    #         await websocket.send(bytes_msg_send)

    #     except websockets.ConnectionClosed as e:
    #         # 客户端关闭连接，跳出对客户端的读取，结束函数
    #         print('connect closed', e.code)
    #         await asyncio.sleep(0.01)
    #         break

    bytes_recv = await websocket.recv()
    msg_recv = pickle.loads(bytes_recv)
    print('the msg is ',msg_recv[0],msg_recv[1])

    msg_send = "ok!!!"
    bytes_msg_send = pickle.dumps(msg_send)
    await websocket.send(bytes_msg_send)
 
async def server_run(websocket, path):
    await server_recv(websocket)  # 接收客户端消息并处理
    time.sleep(40)
    print('finish')
 
# main function
if __name__ == '__main__':
    print("======server main begin======")
    server = websockets.serve(server_run, host_ip, Port)  # 服务器端起server
    asyncio.get_event_loop().run_until_complete(server)  # 事件循环中调用
    asyncio.get_event_loop().run_forever()  # 一直运行