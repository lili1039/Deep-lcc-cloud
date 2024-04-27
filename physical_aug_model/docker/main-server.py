import pickle
import redis
import numpy as np

if __name__ =="__main__":
    '''初始化数据库连接:'''
    # database = 2 
    # 用终端查询键值的时候命令为 redis-cli -n 2
    rs = redis.StrictRedis(host='127.0.0.1',db=2, port=6379,password="chlpw1039")
    rs.flushdb()
    
    # 存入目前混合交通流构成
    # Parameters in mixed traffic
    ID = np.array([1,0,0,1,0,0,1,0,0,1,0,0,1,0,0])  # ID of vehicle types 1: CAV  0: HDV
    pos_cav     = np.where(ID == 1)[0]         # position of CAVs: pos_cav = [ 0  3  6  9 12]
    n_vehicle   = len(ID)                      # number of vehicles
    n_cav       = len(pos_cav)                 # number of CAVs
    ds_num      = 0                            # number of dataset used to construct Hankel matrix
    
    for i in range(n_cav):
        rs.mset({f'ds_num_in_CAV_{i}':pickle.dumps(ds_num)})
    
    rs.mset({f'ID':pickle.dumps(ID)})
    rs.mset({f'n_cav':pickle.dumps(n_cav)})

                
