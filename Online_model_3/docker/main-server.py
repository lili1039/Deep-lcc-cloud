import pickle
import redis
import numpy as np
import asyncio
import sys
import os
# 获取当前文件所在目录（docker文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录（Online_model_3文件夹）的绝对路径
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from app.util import Is_Check,Com_way,iteration_num

total_time_step = 600
Tini = 20
N = 40
n_cav = 5

if __name__ =="__main__":
    '''初始化数据库连接:'''
    # 用终端查询键值的时候命令为 redis-cli -n 2(dataset的编号)
    rs = redis.StrictRedis(host='127.0.0.1',db=2, port=6379,password="chlpw1039")
    rs.flushdb()

    # 各自检查
    timestep_copy = 0
    k_copy = 0

    if Is_Check and Com_way == 2:
        
        while True:

            # 是否进入新一轮仿真
            if timestep_copy >= total_time_step-Tini-N: #600：总步数 60：收集数据阶段用的时长(Tini+N)
                while True:
                    value = rs.mget(f'timestep_copy')[0]
                    if value != None:
                        if pickle.loads(value) == 0:
                            break
                timestep_copy = 0
                k_copy = 0

            # 新一轮仿真开始
            while True:

                keys_error = []
                keys_tolerence = []
                error_sum = np.zeros(9)
                tolerence_sum = np.zeros(9)

                for i in range(n_cav):
                    keys_error.append(f'error_{i}_{timestep_copy}_{k_copy}')
                    keys_tolerence.append(f'tolerence_{i}_{timestep_copy}_{k_copy}')
                
                while True:
                    error_bytes = rs.mget(keys_error)
                    if None not in error_bytes:
                        for value in error_bytes:
                            error_sum = error_sum + (pickle.loads(value))
                        break

                while True:
                    tolerence_bytes = rs.mget(keys_tolerence)
                    if None not in tolerence_bytes:
                        for value in tolerence_bytes:
                            tolerence_sum = tolerence_sum + (pickle.loads(value))
                        break
                
                # print("here")

                if ~np.all(error_sum<tolerence_sum) and k_copy < iteration_num - 1: # 并不是所有error都小于阈值，继续迭代
                    rs.mset({f'rollout_flag_total_{timestep_copy}_{k_copy}':pickle.dumps(0)})

                    error_pri_sum = np.sum(error_sum[0:5])
                    error_dual_sum = np.sum(error_sum[5:])

                    if error_pri_sum>10*error_dual_sum:
                        rs.mset({f'rho_{timestep_copy}_{k_copy}':pickle.dumps(1)}) # 0:no change 1:double 2:half
                    elif error_dual_sum>10*error_pri_sum:
                        rs.mset({f'rho_{timestep_copy}_{k_copy}':pickle.dumps(2)})
                    else:
                        rs.mset({f'rho_{timestep_copy}_{k_copy}':pickle.dumps(0)})
                    k_copy = k_copy+1
                    break
                elif k_copy == iteration_num - 1:
                    print(f"Optimization quits in time step {timestep_copy+1} after maximum iterations.",flush=True)
                    k_copy = 0
                    timestep_copy = timestep_copy + 1
                    rs.mset({f'timestep_copy':pickle.dumps(timestep_copy)})
                    break
                elif np.all(error_sum<tolerence_sum):
                    rs.mset({f'rollout_flag_total_{timestep_copy}_{k_copy}':pickle.dumps(1)})
                    print(f"Optimization quits in time step {timestep_copy+1} after {k_copy+1} iterations.",flush=True)
                    k_copy = 0
                    timestep_copy = timestep_copy + 1
                    rs.mset({f'timestep_copy':pickle.dumps(timestep_copy)})
                    break



