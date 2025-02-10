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
    Subsystem.cav_id = 0

    my_port = 5005 + Subsystem.cav_id
    stop_event = threading.Event()
    data_queue = queue.Queue()  # 🛠 用于存储接收到的数据

    # ✅ **启动后台线程，持续监听 my_port**
    receiver_thread = threading.Thread(target=receive_data, args=(my_port, stop_event, data_queue))
    receiver_thread.daemon = True  
    receiver_thread.start()

    # 创建事件循环
    loop = asyncio.new_event_loop()
    # 将该事件循环设置为当前线程的默认事件循环
    # asyncio.set_event_loop(loop)
    # 启动服务器
    # Subsystem.solver 是一个回调函数，它会在客户端连接到服务器时被调用。你需要定义这个函数来处理客户端的请求和数据交换。
    # 监听本地主机的 6000 端口;host (第2个参数）是服务器监听的主机名或 IP 地址
    # start_server = websockets.serve(Subsystem.solver, f"veh-{Subsystem.cav_id}", 6000, loop=loop)

    start_server = websockets.serve(
        functools.partial(Subsystem.solver, data_queue=data_queue),  
        f"veh-{Subsystem.cav_id}", 6000, loop=loop
    )
    
    # 在事件循环中启动服务器
    loop.run_until_complete(start_server)
    # 一直运行服务器，直到手动停止
    loop.run_forever()