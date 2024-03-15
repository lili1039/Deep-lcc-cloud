import pickle
import redis
import numpy as np
from Initialization import InitParameter,InitVariable,InitAssumeState

if __name__ =="__main__":
    '''初始化数据库连接:'''
    rs = redis.StrictRedis(host='127.0.0.1',db=2, port=6379,password="chlpw1039")
    
    # 存入目前混合交通流构成
    # Parameters in mixed traffic
    ID = np.array([1,0,0,1,0,0,1,0,0,1,0,0,1,0,0])  # ID of vehicle types 1: CAV  0: HDV
    pos_cav     = np.where(ID == 1)[0]         # position of CAVs: pos_cav = [ 0  3  6  9 12]
    n_vehicle   = len(ID)                      # number of vehicles
    n_cav       = len(pos_cav)                 # number of CAVs
    rs.mset({f'ID':pickle.dumps(ID)})
    rs.mset({f'n_cav':pickle.dumps(n_cav)})

    # # 存入期望轨迹
    # rs.mset({f'wait_expect_state':pickle.dumps(0)})
    # for veh_id in range(param.Num_veh):
    #     rs.mset({f'curr_step-{veh_id}':pickle.dumps(-1)})
    #     rs.mset({f'veh{veh_id}_connect':pickle.dumps(0)})
    # while True:  # 等待前车写入数据
    #     value_bytes = rs.mget('wait_expect_state')[0]
    #     value = pickle.loads(value_bytes)
    #     if value == 0:
    #         continue
    #     elif value == 1:
    #         curr_step_bytes = rs.mget('curr_step')[0]
    #         curr_step = pickle.loads(curr_step_bytes)
    #         # 存入期望轨迹
    #         # [1:3]指的是数组的2、3个数
    #         x0_bytes = pickle.dumps(var.x0[curr_step:curr_step+param.Np+1].squeeze())
    #         v0_bytes = pickle.dumps(var.v0[curr_step:curr_step+param.Np+1].squeeze())
    #         expect_dict = {'x_expect':x0_bytes,'v_expect':v0_bytes}
    #         rs.mset(expect_dict)
    #         rs.mset({'wait_expect_state':pickle.dumps(2)})
            
    #     while True:
    #         if pickle.loads(rs.mget('wait_expect_state')[0]) == 1:
    #             rs.mset({'wait_expect_state':pickle.dumps(2)})
    #         all_complete = True  # 假设所有数据都已读完
    #         for veh_id in range(param.Num_veh):
    #             connect_veh_bytes = rs.mget(f'veh{veh_id}_connect')[0]
    #             connect_veh = pickle.loads(connect_veh_bytes)
    #             if connect_veh == 1:
    #                 all_complete = False
    #         if all_complete:
    #             rs.mset({'wait_expect_state':pickle.dumps(0)})
    #             break
                
