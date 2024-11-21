import pickle
import redis
import numpy as np
import asyncio


if __name__ =="__main__":
    '''初始化数据库连接:'''
    # 用终端查询键值的时候命令为 redis-cli -n 2(dataset的编号)
    rs = redis.StrictRedis(host='127.0.0.1',db=2, port=6379,password="chlpw1039")
    rs.flushdb()
    
    # 存入目前混合交通流构成
    ID = np.array([1,0,0,1,0,0,1,0,0,1,0,0,1,0,0])  # ID of vehicle types 1: CAV  0: HDV
    pos_cav     = np.where(ID == 1)[0]         # position of CAVs: pos_cav = [ 0  3  6  9 12]
    n_cav       = len(pos_cav)                 # number of CAVs

    # 各自检查
    timestep_copy = 0
    k_copy = 0
    iteration_num = 30

    while True:
        if timestep_copy >= 5:
            while True:
                if rs.mget(f'timestep_copy')[0] != None:
                    if pickle.loads(rs.mget(f'timestep_copy')[0]) == 0:
                        break
            timestep_copy = 0
            k_copy = 0

        while True:
            keys = []
            flag_values = []

            for i in range(n_cav):
                keys.append(f'rollout_flag_{i}_{timestep_copy}_{k_copy}')
            
            flag_values_bytes = rs.mget(keys)

            for value in flag_values_bytes:
                if value != None:
                    flag_values.append(pickle.loads(value))

            if 0 in flag_values and k_copy < iteration_num - 1:
                rs.mset({f'rollout_flag_total_{timestep_copy}_{k_copy}':pickle.dumps(0)})
                k_copy = k_copy+1
                break
            elif k_copy == iteration_num - 1:
                print(f"Optimization quits in time step {timestep_copy+1} after maximum iterations.",flush=True)
                k_copy = 0
                timestep_copy = timestep_copy + 1
                rs.mset({f'timestep_copy':pickle.dumps(timestep_copy)})
                break
            elif all(var == 1 for var in flag_values) and len(flag_values)==n_cav:
                rs.mset({f'rollout_flag_total_{timestep_copy}_{k_copy}':pickle.dumps(1)})
                print(f"Optimization quits in time step {timestep_copy+1} after {k_copy+1} iterations.",flush=True)
                k_copy = 0
                timestep_copy = timestep_copy + 1
                rs.mset({f'timestep_copy':pickle.dumps(timestep_copy)})
                break
