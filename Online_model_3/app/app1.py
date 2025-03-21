#-- coding:UTF-8 --
from util import SubsystemSolver,receive_data
import websockets
import asyncio
import threading
import queue
import functools


if __name__ == "__main__":
    Subsystem = SubsystemSolver()

    # 车辆参数
    Subsystem.cav_id = 1

    my_port = 5005
    stop_event = threading.Event()
    data_queue = queue.Queue()  # 🛠 用于存储接收到的数据

    # ✅ **启动后台线程，持续监听 my_port**
    # receiver_thread = threading.Thread(target=receive_data, args=(my_port, stop_event, data_queue))
    # receiver_thread.daemon = True  
    # receiver_thread.start()

    receiver_thread_0 = threading.Thread(target=receive_data, args=(my_port+0, stop_event, data_queue))
    receiver_thread_0.daemon = True  
    receiver_thread_0.start()

    receiver_thread_1 = threading.Thread(target=receive_data, args=(my_port+1, stop_event, data_queue))
    receiver_thread_1.daemon = True  
    receiver_thread_1.start()

    receiver_thread_2 = threading.Thread(target=receive_data, args=(my_port+2, stop_event, data_queue))
    receiver_thread_2.daemon = True  
    receiver_thread_2.start()

    receiver_thread_3 = threading.Thread(target=receive_data, args=(my_port+3, stop_event, data_queue))
    receiver_thread_3.daemon = True  
    receiver_thread_3.start()

    receiver_thread_4 = threading.Thread(target=receive_data, args=(my_port+4, stop_event, data_queue))
    receiver_thread_4.daemon = True  
    receiver_thread_4.start()

    # 创建事件循环
    loop = asyncio.new_event_loop()
    # 将该事件循环设置为当前线程的默认事件循环
    # asyncio.set_event_loop(loop)
    # 启动服务器
    # vehicle.solver 是一个回调函数，它会在客户端连接到服务器时被调用。你需要定义这个函数来处理客户端的请求和数据交换。
    # 监听本地主机的 6000 端口;host (第2个参数）是服务器监听的主机名或 IP 地址
    # start_server = websockets.serve(Subsystem.solver, f"veh-{Subsystem.cav_id}", 6000, loop=loop)

    async def server_start():
        server = await websockets.serve(
            functools.partial(Subsystem.solver, data_queue=data_queue),
            f"veh-{Subsystem.cav_id}", 6000
        )
        return server

    # 在事件循环中启动服务器
    loop.run_until_complete(server_start())
    # 一直运行服务器，直到手动停止
    loop.run_forever()