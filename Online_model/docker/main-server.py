import pickle
import redis
import numpy as np
import asyncio
from StopCriteria import StopCriteria

if __name__ =="__main__":
    '''初始化数据库连接:'''
    # 用终端查询键值的时候命令为 redis-cli -n 2(dataset的编号)
    rs = redis.StrictRedis(host='127.0.0.1',db=2, port=6379,password="chlpw1039")
    rs.flushdb()
    
    # 存入目前混合交通流构成
    ID = np.array([1,0,0,1,0,0,1,0,0,1,0,0,1,0,0])  # ID of vehicle types 1: CAV  0: HDV
    pos_cav     = np.where(ID == 1)[0]         # position of CAVs: pos_cav = [ 0  3  6  9 12]
    n_cav       = len(pos_cav)                 # number of CAVs
    ds_num      = 0                            # number of dataset used to construct Hankel matrix
    
    for i in range(n_cav):
        rs.mset({f'ds_num_in_CAV_{i}':pickle.dumps(ds_num)})
    
    rs.mset({f'ID':pickle.dumps(ID)})
    rs.mset({f'n_cav':pickle.dumps(n_cav)})

    # 当返回误差容差计算结果的容器数占总容器的 %时，启动迭代停止检验
    check_rate = 0.6

    while True:
        if rs.mget(f'{i}_check_ready')[0] == None:
            continue
        
        # 检验是否所有容器都计算好了error和tolerance 
        while True:
            all_complete_flag = True
            
            for i in range(n_cav):
                if pickle.loads(rs.mget(f'{i}_check_ready')[0])[0] != 1:
                    all_complete_flag = False
                    break
            
            if all_complete_flag:
                break
        
        # 运行StopCriteria函数
        timestep = pickle.loads(rs.mget(f'{i}_check_ready')[0])[1]
        k = pickle.loads(rs.mget(f'{i}_check_ready')[0])[2]

        if asyncio.run(StopCriteria(k,rs,n_cav,timestep)):
            rs.mset({f'Check_Stop':pickle.dumps([1,timestep,k])})
        else:
            rs.mset({f'Check_Stop':pickle.dumps([0,timestep,k])})

        for i in range(n_cav):
            rs.mset({f'{i}_check_ready':pickle.dumps([0,timestep,k])})



        
            