#-- coding:UTF-8 --
import numpy as np
from util import VehicleSolver
import websockets
import asyncio
import csv


if __name__ == "__main__":
    vehicle = VehicleSolver()

    # 车辆参数
    vehicle.veh_id = 3
    
    data = []
    with open('/app/vehicle_param.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)

    header = data[0]  # 表头在第1行
    vehicle.Mass   = eval(data[vehicle.veh_id+1][header.index('Mass')])
    vehicle.Tao    = eval(data[vehicle.veh_id+1][header.index('Tao')])
    vehicle.Ca     = eval(data[vehicle.veh_id+1][header.index('Ca')])
    vehicle.Radius = eval(data[vehicle.veh_id+1][header.index('Radius')])
    vehicle.f      = eval(data[vehicle.veh_id+1][header.index('f')])
    vehicle.Eta    = eval(data[vehicle.veh_id+1][header.index('Eta')])

    # 权重矩阵
    vehicle.F = 10*np.identity(2*vehicle.Np)
    vehicle.G = 5*np.identity(2*vehicle.Np)
    vehicle.Q = 10*np.identity(2*vehicle.Np)
    vehicle.R = 0.01*np.identity(vehicle.Np)

    vehicle.update_param()

    print("start",flush=True)
    # 创建事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # 启动服务器
    start_server = websockets.serve(vehicle.solver, f"veh-{vehicle.veh_id}", 6000, loop=loop)
    # 在事件循环中启动服务器
    loop.run_until_complete(start_server)
    # 一直运行服务器
    loop.run_forever()