import asyncio
import numpy as np
import pickle
from scipy.optimize import minimize
import time
import redis
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as ConnectionTimeoutError
import websockets


# 车辆非线性模型
def NonlinearDynamic(X,u,Time_step,Veh):
    Position = X[0]
    Velocity = X[1]
    Torque = X[2]
    Mass = Veh[0]
    R = Veh[1]
    g = Veh[2]
    f = Veh[3]
    Eta = Veh[4]
    Ca = Veh[5]
    Tao = Veh[6]

    Xk = np.zeros([3,1])

    Xk[0,0] = Position + Velocity*Time_step
    Xk[1,0] = Velocity + 1/Mass *(Eta*Torque/R - Ca*Velocity**2 - Mass*g*f)*Time_step
    Xk[2,0] = Torque - 1/Tao*Torque*Time_step + 1/Tao*u*Time_step

    return Xk.squeeze()



def UpdateAssumeState(X0,U_best,Time_step,Vehicle_Type,Np,ua):
    U = U_best[0]
    Mass = Vehicle_Type[0]
    Radius = Vehicle_Type[1]
    g = Vehicle_Type[2]
    f = Vehicle_Type[3]
    Eta = Vehicle_Type[4]
    Ca = Vehicle_Type[5]

    Xk = NonlinearDynamic(X0,U,Time_step,Vehicle_Type)
    Position_star_0 = Xk[0]
    Velocity_star_0 = Xk[1]
    Torque_star_0   = Xk[2]

    Temp = np.zeros([3,Np+1])
    Temp[:,0] = [Position_star_0,Velocity_star_0,Torque_star_0]
    ua[0:Np-1] = U_best[1:Np]  # 构造假定控制序列，将前一刻最优控制序列后移一步
    for j in range(Np-1):
        Temp[:,j+1] = NonlinearDynamic(Temp[:,j].squeeze(),ua[j],Time_step,Vehicle_Type)
    ua[Np-1] = (Ca*Temp[1,Np-1]**2 + Mass*g*f)/Eta*Radius  # 控制序列最后一位为稳态值
    Temp[:,Np] = NonlinearDynamic(Temp[:,Np-1],ua[Np-1],Time_step,Vehicle_Type)
    Pa = Temp[0,:].reshape(Np+1,1)
    Va = Temp[1,:].reshape(Np+1,1)
    return Pa,Va,ua


class VehicleParam:
    def __init__(self):
        # 控制参数
        self.Time_step = 0.1
        self.Np = 20
        self.trigger_time = 8

        # 车辆参数
        self.Num_veh = 6
        self.veh_id = 0
        self.Mass   = np.array(1000)
        self.f      = 0.01
        self.Eta    = 0.96
        self.g      = 9.8
        self.Tao    = 0.5 + (self.Mass - 1000)/1000 * 0.3
        self.Ca     = (0.78 + (self.Mass - 1000)/1000 * 0.2)/2
        self.Radius = 0.3 + (self.Mass - 1000)/2000 * 0.1
        self.Vehicle_Type = []

        self.AccMax = 6
        self.AccMin = -6
        self.Ulimit = []

        self.Vmin = 0       # 行驶速度约束
        self.Vmax = 30

        self.d  = 20
        self.dsafe = 5             # 最小安全距离

        self.C = np.array([[1,0,0],
                           [0,1,0]])

        # 权重矩阵
        self.F = 10*np.identity(2*self.Np)
        self.G = 0
        self.Q = 10*np.identity(2*self.Np)
        self.R = 0.01*np.identity(self.Np)

    def update_param(self):
        Torquebound = np.zeros([1,2])
        Torquebound[0,0] = self.Mass * self.AccMin * self.Radius / self.Eta
        Torquebound[0,1] = self.Mass * self.AccMax * self.Radius / self.Eta
        for i in range(self.Np):
            self.Ulimit.append([Torquebound[0,0],Torquebound[0,1]])

        self.Vehicle_Type = [self.Mass,self.Radius,self.g,self.f,self.Eta,self.Ca,self.Tao]


class VehicleSolver(VehicleParam):
    def __init__(self):
        super().__init__()

    async def solver(self,websocket, path):
        start_time = time.perf_counter()
        executor = ThreadPoolExecutor()
        
        # 数据库连接
        rs = redis.StrictRedis(host='172.18.0.1',port=6379,db=2,password="SBpax200")
        rs.mset({f'veh{self.veh_id}_connect':pickle.dumps(1)})

        # 获取数据
        # 收到客户端的消息
        msg_bytes_recv = await websocket.recv()
        msg_recv = pickle.loads(msg_bytes_recv)
        client_send_time = msg_recv[0]
        state = msg_recv[1]
        Position = state[0]
        Velocity = state[1]
        Torque   = state[2]
        assume_state = msg_recv[2]
        Pa,Va,ua = assume_state
        curr_step = msg_recv[3]
        trigger_cloud = msg_recv[4]
        Ynfa = msg_recv[5]
        edge_solve_time = msg_recv[6]
        delay_up = time.time()-client_send_time
        
        #if self.veh_id == 1:
        #    self.trigger_time = self.Time_step + 0.1 - edge_solve_time - 0.05
        #else:
        #    self.trigger_time = self.Time_step - edge_solve_time - 0.05

        veh_redis_msg = RedisMessage(rs,self.veh_id,self.Np)
        veh_redis_msg.send_assume_msg(Pa,Va,curr_step)

        if not trigger_cloud:
            # 主动向客户端发送消息
            curr_time = time.time()
            msg_send = [delay_up,curr_time,-2*np.ones([self.Np,])]
            msg_bytes_send = pickle.dumps(msg_send)
            try:
                await websocket.send(msg_bytes_send)
            except websockets.exceptions.ConnectionClosedOK:
                print("Delay exceeded the specified range. Connection closed by the client",flush=True)
            rs.mset({f'veh{self.veh_id}_connect':pickle.dumps(0)})
        else:
            print("step: {}  |".format(curr_step),flush=True,end="")
            # 准备获取期望序列
            value = pickle.loads(rs.mget('wait_expect_state')[0])
            if value == 0:
                rs.mset({'curr_step':pickle.dumps(curr_step)})
                rs.mset({'wait_expect_state':pickle.dumps(1)})

            # 获取数据库数据
            start_read_redis = time.time()  # 记录读取数据库用时

            # 期望序列信息
            while True:
                value = pickle.loads(rs.mget('wait_expect_state')[0])
                if value == 2: # 期望序列存储完毕
                    break
            x_expect,v_expect = veh_redis_msg.receive_expect_msg()

            # 正常通信时数据源数量
            if self.veh_id == 0:
                msg_cnt = 1
            else:
                msg_cnt = 2

            # 获取前车数据
            Pa_ahead = Ynfa[0,:]

            end_read_redis = time.time()
            print("read redis: {:.3f}s  |".format(end_read_redis-start_read_redis),flush=True,end="")

            X0 = np.array([Position,Velocity,Torque]).T   # the vehicle variable in the last time step

            # 期望序列的信息
            Pd = x_expect - (self.veh_id+1) * self.d       # 期望位移序列
            Vd = v_expect                        # 期望速度序列
            Ydes = np.array([Pd,Vd]).squeeze()          # 输出量总期望序列
            Ya = np.array([Pa,Va]).squeeze()     # 假定输出序列
            # 构造终端约束
            # 终端约束
            Pnp = (Ynfa[0,Ynfa.shape[1]-1] + Ydes[0,Ydes.shape[1]-1])/msg_cnt
            Vnp = (Ynfa[1,Ynfa.shape[1]-1] + Ydes[1,Ydes.shape[1]-1])/msg_cnt
            Tnp = (self.Ca*Vnp**2 + self.Mass*self.g*self.f)/self.Eta*self.Radius
            Xnp = np.array([Pnp,Vnp,Tnp])

            # 记录求解时间
            start_solve_time = time.time()
            # 构造并求解优化问题
            u = ua[:]
            TerminalCons = {'type': 'eq', 'fun': self.EqConstraints, 'args': (X0,Xnp)} # 终端等式约束
            IneqCons = {'type': 'ineq', 'fun': self.IneqConstraints, 'args': (Pa_ahead,X0)}
            Cons = (TerminalCons,IneqCons)
            bounds = self.Ulimit  # 控制量约束
            future = executor.submit(minimize, self.CostFunction, x0=u, args=(X0,Ydes,Ya,Ynfa), method='SLSQP', bounds=bounds, constraints=Cons, options={'disp':False})
            try:
                # 等待线程返回结果
                res = await asyncio.wait_for(asyncio.wrap_future(future), timeout=self.trigger_time-(time.perf_counter()-start_time))
                if 'Iteration limit reached' in res.message:
                    print("Result discarded: Iteration limit exceeded.",flush=True,end="")
                    U_best = -1*np.ones([self.Np,])
                else:
                    print("compute success |",flush=True,end="")
                    U_best = res.x
                end_solve_time = time.time()
                # 主动向客户端发送消息
                curr_time = time.time()
                msg_send = [delay_up,curr_time,U_best]
                msg_bytes_send = pickle.dumps(msg_send)
                print("solve time: {:.3f}s".format(end_solve_time-start_solve_time),flush=True)
                try:
                    await websocket.send(msg_bytes_send)
                except websockets.exceptions.ConnectionClosedOK:
                    print("Delay exceeded the specified range. Connection closed by the client",flush=True)
                rs.mset({f'veh{self.veh_id}_connect':pickle.dumps(0)})
            except ConnectionTimeoutError:
                future.cancel()
                print("compute fail",flush=True)
                # 主动向客户端发送消息
                curr_time = time.time()
                msg_send = [delay_up,curr_time,-1*np.ones([self.Np,])]
                msg_bytes_send = pickle.dumps(msg_send)
                try:
                    await websocket.send(msg_bytes_send)
                except websockets.exceptions.ConnectionClosedOK:
                    print("Delay exceeded the specified range. Connection closed by the client",flush=True)
                rs.mset({f'veh{self.veh_id}_connect':pickle.dumps(0)})


    def CostFunction(self,u,X0,Ydes,Ya,Ynfa):
        Np = self.Np
        Time_step = self.Time_step
        Vehicle_Type = self.Vehicle_Type
        Mass = Vehicle_Type[0];Radius = Vehicle_Type[1]; g = Vehicle_Type[2]
        f = Vehicle_Type[3];Eta = Vehicle_Type[4];Ca = Vehicle_Type[5]

        Xp = np.zeros([3,Np+1])
        Xp[:,0] = X0

        for i in range(Np):
            Xp[:,i+1] = NonlinearDynamic(Xp[:,i],u[i],Time_step,Vehicle_Type)

        Yp = np.dot(self.C,Xp)      # Predictive State

        Udes = Radius/Eta*(Ca*Xp[1,:]**2 + Mass*g*f)

        Edes = Yp[:,0:Np] - Ydes[:,0:Np]
        Ea = Yp[:,0:Np] - Ya[:,0:Np]
        Enfa = Yp[:,0:Np] - Ynfa[:,0:Np]
        u = u.reshape(Np,1)
        Udes = Udes[0:Np].reshape(Np,1)

        Edes_flatten = np.ravel(Edes.T).reshape(-1,1)
        Ea_flatten = np.ravel(Ea.T).reshape(-1,1)
        Enfa_flatten = np.ravel(Enfa.T).reshape(-1,1)

        Cost = np.dot(np.dot(Edes_flatten.T,self.Q), Edes_flatten) + \
               np.dot(np.dot((u - Udes).T, self.R), u - Udes) + \
               np.dot(np.dot(Ea_flatten.T, self.F), Ea_flatten) + \
               np.dot(np.dot(Enfa_flatten.T, self.G), Enfa_flatten)

        return Cost.squeeze()


    # 等式约束
    def EqConstraints(self,u,X0,Xnp):
        Xp = np.zeros([3,self.Np+1])
        Xp[:,0] = X0
    
        for i in range(self.Np):
            Xp[:,i+1] = NonlinearDynamic(Xp[:,i],u[i],self.Time_step,self.Vehicle_Type)
    
        Ceq = Xp[:,Xp.shape[1]-1] - Xnp
    
        return Ceq

    
    # 不等式约束
    def IneqConstraints(self,u,Pa_ahead,X0):
        Np = self.Np
        Pa_ahead = Pa_ahead.reshape(Np+1,1)

        Xp = np.zeros([3,self.Np+1])
        Xp[:,0] = X0

        for i in range(self.Np):
            Xp[:,i+1] = NonlinearDynamic(Xp[:,i],u[i],self.Time_step,self.Vehicle_Type)

        speed_lower = Xp[1,0:Np].reshape(Np,1) - self.Vmin*np.ones([Np,1])
        speed_upper = self.Vmax*np.ones([Np,1]) - Xp[1,0:Np].reshape(Np,1)

        InCons = np.vstack((speed_lower,speed_upper))
        

        if self.veh_id > 0:
            C_dsafe = Pa_ahead[0:Np,0].reshape(Np,1) - Xp[0,0:Np].reshape(Np,1) + self.d - self.dsafe*np.ones([Np,1])
            InCons = np.vstack((InCons,C_dsafe))

        return InCons.squeeze()


class RedisMessage:
    def __init__(self,redis,veh_id,Np):
        self.veh_id = veh_id
        self.rs = redis
        self.Np = Np

    def receive_expect_msg(self):
        x_expect_bytes_from_redis = self.rs.mget('x_expect')[0]
        v_expect_bytes_from_redis = self.rs.mget('v_expect')[0]
        x_expect = pickle.loads(x_expect_bytes_from_redis)
        v_expect = pickle.loads(v_expect_bytes_from_redis)

        return x_expect,v_expect

    def send_assume_msg(self,Pa,Va,curr_step):
        # 序列化
        Pa_data = pickle.dumps(Pa.squeeze())
        Va_data = pickle.dumps(Va.squeeze())
        curr_step_data = pickle.dumps(curr_step)
        # 构建字典映射
        assume_dict = {f'Pa-{self.veh_id}':Pa_data,
                       f'Va-{self.veh_id}':Va_data,
                       f'curr_step-{self.veh_id}':curr_step_data
                       }
        # 存入数据
        self.rs.mset(assume_dict)
