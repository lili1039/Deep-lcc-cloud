import asyncio
import numpy as np
import pickle
import time
import redis
import math
# from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import TimeoutError as ConnectionTimeoutError
import websockets

async def dDeeP_LCC(cav_id,Uip,Yip,Uif,Yif,Eip,Eif,ui_ini,yi_ini,ei_ini,\
                lambda_yi,u_limit,si_limit,rho,KKT_vert,Hz_vert,rs):
    
    # stoping criterion
    iteration_num = 300
    error_absolute = 0.1
    error_relative = 1e-3

    # problem size
    m = ui_ini.ndim         # the size of control input of each subsystem
    p = yi_ini.shape[0]     # the size of output of each subsystem
    n_cav_bytes = rs.mget('n_cav')[0]
    n_cav = pickle.loads(n_cav_bytes)
    
    # time horizon
    Tini = int(Uip.shape[0]/m)
    N = int(Uif.shape[0]/m)
    T = int(Uip.shape[1]+Tini+N-1)

    # Initial dual variables
    g_initial = np.zeros(T-Tini-N+1)
    mu_initial = np.zeros(T-Tini-N+1)
    eta_initial = np.zeros(N)
    phi_initial = np.zeros(N)
    theta_initial = np.zeros(N)

    # parameters in ADMM
    K=np.kron(np.eye(N),np.append(np.zeros(p-2),[1,0]))
    P=np.kron(np.eye(N),np.append(np.zeros(p-1),[1]))

    ui_ini = ui_ini.reshape(m*Tini,order='F')       #一维数组
    yi_ini= yi_ini.reshape(p*Tini,order='F')        #一维数组
    ei_ini = ei_ini.reshape(Tini,order='F')         #一维数组

    if cav_id == 0:
        beqg = np.hstack((ui_ini,ei_ini,np.zeros(N)))  #一维数组
    else:
        beqg = np.hstack((ui_ini,ei_ini))              #一维数组

    # initial value for variables
    g = g_initial
    z = g_initial
    mu = mu_initial
    eta = eta_initial
    phi = phi_initial
    theta = theta_initial
    s=P@Yif@g       #一维数组
    u=Uif@g         #一维数组

    for k in range(iteration_num):
        # error for primal and dual problems
        error_pri1 = 0
        error_pri2 = 0
        error_pri3 = 0
        error_pri4 = 0
        error_dual1 = 0
        error_dual2 = 0
        error_dual3 = 0
        error_dual4 = 0

        tolerence_pri1 = 0
        tolerence_pri2 = 0
        tolerence_pri3 = 0
        tolerence_pri4 = 0
        tolerence_dual1 = 0
        tolerence_dual2 = 0
        tolerence_dual3 = 0
        tolerence_dual4 = 0

        # Step1: update g
        
            # updata eta_bar

            if cav_id != n_cav-1:
                eta_bar = eta - rho*K@Yif@z
                rs.mset({f'eta_bar_in_CAV_{cav_id}':pickle.dumps([eta_bar,k])})

                while True:
                    
            if cav_id == 0:
                # qg 的结果是一维数组
                qg = -lambda_yi*Yip.T@yi_ini + mu/2 - rho*z/2 - \
                    Yif.T@P.T@phi/2 - Uif.T@theta/2 - rho*Yif.T@P.T@s/2 - rho*Uif.T@u/2
            else:
                qg = -lambda_yi*Yip.T@yi_ini + mu/2 - rho*z/2 + Eif.T@eta[i-1]/2 - rho/2*Eif[i].T@K[i-1]@Yif[i-1]@z[i-1] - \
                    Yif[i].T@P[i].T@phi[i]/2 - Uif[i].T@theta[i]/2 - rho*Yif[i].T@P[i].T@s[i]/2 - rho*Uif[i].T@u[i]/2

        for i in range(n_cav):
            temp1 = np.hstack((-qg[i],beqg[i]))
            temp2 = KKT_vert[i]@temp1
            g_plus[i] = temp2[0:T-Tini-N+1]
        

        # Step2: parallel update z/s/u & mu/eta/phi/theta
        for i in range(n_cav-1):
            qz[i] = -mu[i]/2 - rho/2*g_plus[i] - \
                Yif[i].T@K[i].T@eta[i]/2 -rho/2*Yif[i].T@K[i].T@Eif[i+1]@g_plus[i+1]
            z_plus[i] = -Hz_vert[i]@qz[i]
            
            s_temp = P[i]@Yif[i]@g_plus[i] - phi[i]/rho
            s_plus[i] = np.minimum(np.maximum(s_temp,si_limit[i][0]*np.ones(N)),si_limit[i][1]*np.ones(N)) # projection in the box constraint

            u_temp = Uif[i]@g_plus[i]-theta[i]/rho
            u_plus[i] = np.minimum(np.maximum(u_temp,u_limit[0]*np.ones(N)),u_limit[1]*np.ones(N)) # projection in the box constraint

            mu_plus[i] = mu[i] + rho*(g_plus[i]-z_plus[i])
            eta_plus[i] = eta[i] + rho*(Eif[i+1]@g_plus[i+1]-K[i]@Yif[i]@z_plus[i])
            phi_plus[i] = phi[i] + rho*(s_plus[i] - P[i]@Yif[i]@g_plus[i])
            theta_plus[i] = theta[i] + rho*(u_plus[i] - Uif[i]@g_plus[i])

            error_pri1 = error_pri1 + np.linalg.norm(g_plus[i] - z_plus[i])
            tolerence_pri1 = tolerence_pri1 + math.sqrt(g_plus[i].shape[0])*error_absolute + \
                error_relative*max(np.linalg.norm(g_plus[i]),np.linalg.norm(z_plus[i]))
            error_dual1 = error_dual1 + rho*np.linalg.norm(z_plus[i]-z[i])
            tolerence_dual1 = tolerence_dual1 + math.sqrt(z_plus[i].shape[0])*error_absolute + error_relative*np.linalg.norm(mu_plus[i])
            
            error_pri2 = error_pri2 + np.linalg.norm(Eif[i+1]@g_plus[i+1] - K[i]@Yif[i]@z_plus[i])
            tolerence_pri2 = tolerence_pri2 + math.sqrt((Eif[i+1]@g_plus[i+1]).shape[0])*error_absolute + \
                error_relative*max(np.linalg.norm(Eif[i+1]@g_plus[i+1]),np.linalg.norm(K[i]@Yif[i]@z_plus[i]))
            error_dual2 = error_dual2 + rho*np.linalg.norm(Eif[i+1].T@K[i]@Yif[i]@(z_plus[i]-z[i]))
            tolerence_dual2 = tolerence_dual2 + math.sqrt((Eif[i+1].T@K[i]@Yif[i]@(z_plus[i]-z[i])).shape[0])*error_absolute + error_relative*np.linalg.norm(Eif[i+1].T@eta_plus[i])

            error_pri3 = error_pri3 + np.linalg.norm(s_plus[i] - P[i]@Yif[i]@g_plus[i])
            tolerence_pri3 = tolerence_pri3 + math.sqrt(s_plus[i].shape[0])*error_absolute + error_relative*max(np.linalg.norm(s_plus[i]),np.linalg.norm(P[i]@Yif[i]@g_plus[i]))
            error_dual3 = error_dual3 + rho*np.linalg.norm(Yif[i].T@P[i].T@(s_plus[i]-s[i]))
            tolerence_dual3 = tolerence_dual3 + math.sqrt(z_plus[i].shape[0])*error_absolute + error_relative*np.linalg.norm(Yif[i].T@P[i].T@phi_plus[i])


            error_pri4 =  error_pri4 + np.linalg.norm(u_plus[i]-Uif[i]@g_plus[i])
            tolerence_pri4 = tolerence_pri4 + math.sqrt(u_plus[i].shape[0])*error_absolute + error_relative*max(np.linalg.norm(Uif[i]@g_plus[i]),np.linalg.norm(u_plus[i]))
            error_dual4 = error_dual4 + rho*np.linalg.norm(Uif[i].T@(u_plus[i]-u[i]))
            tolerence_dual4 =tolerence_dual4 + math.sqrt(z_plus[i].shape[0])*error_absolute + error_relative*np.linalg.norm(Uif[i].T@theta_plus[i])

        i = i + 1
        qz[i] = -mu[i]/2 - rho/2*g_plus[i]
        z_plus[i] = -Hz_vert[i]@qz[i]
        
        s_temp = P[i]@Yif[i]@g_plus[i] - phi[i]/rho
        s_plus[i] = np.minimum(np.maximum(s_temp,si_limit[i][0]*np.ones(N)),si_limit[i][1]*np.ones(N)) # projection in the box constraint

        u_temp = Uif[i]@g_plus[i]-theta[i]/rho
        u_plus[i] = np.minimum(np.maximum(u_temp,u_limit[0]*np.ones(N)),u_limit[1]*np.ones(N)) # projection in the box constraint

        mu_plus[i] = mu[i] + rho*(g_plus[i]-z_plus[i])
        phi_plus[i] = phi[i] + rho*(s_plus[i] - P[i]@Yif[i]@g_plus[i])
        theta_plus[i] = theta[i] + rho*(u_plus[i] - Uif[i]@g_plus[i])

        error_pri1 = error_pri1 + np.linalg.norm(g_plus[i] - z_plus[i])
        tolerence_pri1 = tolerence_pri1 + math.sqrt(g_plus[i].shape[0])*error_absolute + \
            error_relative*max(np.linalg.norm(g_plus[i]),np.linalg.norm(z_plus[i]))
        error_dual1 = error_dual1 + rho*np.linalg.norm(z_plus[i]-z[i])
        tolerence_dual1 = tolerence_dual1 + math.sqrt(z_plus[i].shape[0])*error_absolute + error_relative*np.linalg.norm(mu_plus[i])
        
        error_pri3 = error_pri3 + np.linalg.norm(s_plus[i] - P[i]@Yif[i]@g_plus[i])
        tolerence_pri3 = tolerence_pri3 + math.sqrt(s_plus[i].shape[0])*error_absolute + error_relative*max(np.linalg.norm(s_plus[i]),np.linalg.norm(P[i]@Yif[i]@g_plus[i]))
        error_dual3 = error_dual3 + rho*np.linalg.norm(Yif[i].T@P[i].T@(s_plus[i]-s[i]))
        tolerence_dual3 = tolerence_dual3 + math.sqrt(z_plus[i].shape[0])*error_absolute + error_relative*np.linalg.norm(Yif[i].T@P[i].T@phi_plus[i])


        error_pri4 =  error_pri4 + np.linalg.norm(u_plus[i]-Uif[i]@g_plus[i])
        tolerence_pri4 = tolerence_pri4 + math.sqrt(u_plus[i].shape[0])*error_absolute + error_relative*max(np.linalg.norm(Uif[i]@g_plus[i]),np.linalg.norm(u_plus[i]))
        error_dual4 = error_dual4 + rho*np.linalg.norm(Uif[i].T@(u_plus[i]-u[i]))
        tolerence_dual4 =tolerence_dual4 + math.sqrt(z_plus[i].shape[0])*error_absolute + error_relative*np.linalg.norm(Uif[i].T@theta_plus[i])

        g = g_plus
        z = z_plus
        u = u_plus
        s = s_plus
        mu = mu_plus
        eta = eta_plus
        phi = phi_plus
        theta = theta_plus

        # Check stopping criterion
        if error_pri1 <= tolerence_pri1 and error_dual1 <= tolerence_dual1 \
            and error_pri2 <= tolerence_pri2 and error_dual2 <= tolerence_dual2 \
                and error_pri3 <= tolerence_pri3 and error_dual3 <= tolerence_dual3 \
                    and error_pri4 <= tolerence_pri4 and error_dual4 <= tolerence_dual4 :
            break


    # Record optimal value
    real_iter_num = k
    g_opt = g
    mu_opt = mu
    eta_opt = eta
    phi_opt = phi
    theta_opt = theta
    u_opt = np.zeros(n_cav*N)

    for i in range(n_cav):
        u_opt[i::n_cav] = u[i]
    
    return u_opt,g_opt,mu_opt,eta_opt,phi_opt,theta_opt,real_iter_num


class SubsystemParam:
    def __init__(self):
        # 仿真时间参数
        self.total_time = 40
        self.Tstep = 0.05
        self.total_time_step = int(self.total_time/self.Tstep)

        # DeeP-LCC 参数
        self.T = 300
        self.Tini = 20
        self.N = 50

        # 系统车辆参数
        self.cav_id = 0
        self.n_vehicle_sub   = 0
        self.Uip             = []
        self.Uif             = []
        self.Eip             = []
        self.Eif             = []
        self.Yip             = []
        self.Yif             = []
        self.uini            = []
        self.eini            = []
        self.yini            = []
        self.KKT_vert        = []
        self.Hz_vert         = []
        self.rho             = 0
        self.lambda_yi       = 0

        # 安全限制
        self.acel_max = 2       # max accelaration
        self.dcel_max = -5      # min accelaration
        self.spacing_max = 40
        self.spacing_min = 5
        self.s_star = 20
        self.u_limit = np.array([self.dcel_max,self.acel_max])
        self.s_limit = np.array([self.spacing_min-self.s_star,self.spacing_max-self.s_star])       

class SubsystemSolver(SubsystemParam):
    def __init__(self):
        super().__init__()

    async def solver(self,websocket, path):
        start_time = time.perf_counter()
        
        # 数据库连接
        rs = redis.StrictRedis(host='172.18.0.1',port=6379,db=2,password="chlpw1039")
        # 数据库标志该子系统已连接
        rs.mset({f'Subsystem{self.cav_id}_connect':pickle.dumps(1)})

        # 获取数据
        # 收到客户端的消息(初始参数和数据)
        msg_bytes_recv  = await websocket.recv()
        msg_recv        = pickle.loads(msg_bytes_recv)
        self.cav_id          = msg_recv[0]
        self.n_vehicle_sub   = msg_recv[1]
        self.Uip             = msg_recv[2]
        self.Uif             = msg_recv[3]
        self.Eip             = msg_recv[4]
        self.Eif             = msg_recv[5]
        self.Yip             = msg_recv[6]
        self.Yif             = msg_recv[7]
        self.uini            = msg_recv[8]
        self.eini            = msg_recv[9]
        self.yini            = msg_recv[10]
        self.KKT_vert        = msg_recv[11]
        self.Hz_vert         = msg_recv[12]
        self.rho             = msg_recv[13]
        self.lambda_yi       = msg_recv[14]
        print("Subsystem {}: data loaded".format(self.cav_id))
        msg_send = True
        msg_bytes_send = pickle.dumps(msg_send)
        await websocket.send(msg_bytes_send)

        # Construct distributed data for ADMM computation
        for k in range(self.Tini-1,self.total_time_step-1):
            
        

        # # 创建了一个 RedisMessage 对象，传入了 rs（Redis 连接）、self.veh_id 和 self.Np 作为参数：
        # subsystem_redis_msg = RedisMessage(rs,self.cav_id)
        # # 向redis传入消息
        # subsystem_redis_msg.send_assume_msg(Pa,Va,curr_step)

        # if not trigger_cloud:
        #     # 主动向客户端发送消息
        #     curr_time = time.time()
        #     msg_send = [delay_up,curr_time,-2*np.ones([self.Np,])]
        #     msg_bytes_send = pickle.dumps(msg_send)
        #     try:
        #         await websocket.send(msg_bytes_send)
        #     except websockets.exceptions.ConnectionClosedOK:
        #         # 延迟过大，客户端无法收到信息
        #         print("Delay exceeded the specified range. Connection closed by the client",flush=True)
        #     # 使用 pickle.dumps(0) 将数字 0 序列化为字节流，并将其存储到 Redis 中：
        #     # 这会在 Redis 中创建一个键名为 veh{self.veh_id}_connect 的条目，对应的值是序列化后的字节流。
        #     rs.mset({f'veh{self.veh_id}_connect':pickle.dumps(0)})
        # else:
        #     print("step: {}  |".format(curr_step),flush=True,end="")
        #     # 准备获取期望序列
        #     value = pickle.loads(rs.mget('wait_expect_state')[0])
        #     if value == 0:
        #         rs.mset({'curr_step':pickle.dumps(curr_step)})
        #         rs.mset({'wait_expect_state':pickle.dumps(1)})

        #     # 获取数据库数据
        #     start_read_redis = time.time()  # 记录读取数据库用时

        #     # 从redis中获取期望序列信息
        #     while True:
        #         value = pickle.loads(rs.mget('wait_expect_state')[0])
        #         if value == 2: # 期望序列存储完毕
        #             break
        #     x_expect,v_expect = veh_redis_msg.receive_expect_msg()

        #     # 正常通信时数据源数量
        #     if self.veh_id == 0:
        #         msg_cnt = 1
        #     else:
        #         msg_cnt = 2

        #     # 获取前车数据
        #     Pa_ahead = Ynfa[0,:]

        #     end_read_redis = time.time()
        #     print("read redis: {:.3f}s  |".format(end_read_redis-start_read_redis),flush=True,end="")

        #     X0 = np.array([Position,Velocity,Torque]).T   # the vehicle variable in the last time step

        #     # 期望序列的信息
        #     Pd = x_expect - (self.veh_id+1) * self.d       # 期望位移序列
        #     Vd = v_expect                        # 期望速度序列
        #     Ydes = np.array([Pd,Vd]).squeeze()          # 输出量总期望序列
        #     Ya = np.array([Pa,Va]).squeeze()     # 假定输出序列
        #     # 构造终端约束
        #     # 终端约束


        #     # 记录求解时间
        #     start_solve_time = time.time()
        #     # 构造并求解优化问题
        #     u = ua[:]
        #     TerminalCons = {'type': 'eq', 'fun': self.EqConstraints, 'args': (X0,Xnp)} # 终端等式约束
        #     IneqCons = {'type': 'ineq', 'fun': self.IneqConstraints, 'args': (Pa_ahead,X0)}
        #     Cons = (TerminalCons,IneqCons)
        #     bounds = self.Ulimit  # 控制量约束
        #     future = executor.submit(minimize, self.CostFunction, x0=u, args=(X0,Ydes,Ya,Ynfa), method='SLSQP', bounds=bounds, constraints=Cons, options={'disp':False})
        #     try:
        #         # 等待线程返回结果
        #         res = await asyncio.wait_for(asyncio.wrap_future(future), timeout=self.trigger_time-(time.perf_counter()-start_time))
        #         if 'Iteration limit reached' in res.message:
        #             print("Result discarded: Iteration limit exceeded.",flush=True,end="")
        #             U_best = -1*np.ones([self.Np,])
        #         else:
        #             print("compute success |",flush=True,end="")
        #             U_best = res.x
        #         end_solve_time = time.time()
        #         # 主动向客户端发送消息
        #         curr_time = time.time()
        #         msg_send = [delay_up,curr_time,U_best]
        #         msg_bytes_send = pickle.dumps(msg_send)
        #         print("solve time: {:.3f}s".format(end_solve_time-start_solve_time),flush=True)
        #         try:
        #             await websocket.send(msg_bytes_send)
        #         except websockets.exceptions.ConnectionClosedOK:
        #             print("Delay exceeded the specified range. Connection closed by the client",flush=True)
        #         rs.mset({f'veh{self.veh_id}_connect':pickle.dumps(0)})
        #     except ConnectionTimeoutError:
        #         future.cancel()
        #         print("compute fail",flush=True)
        #         # 主动向客户端发送消息
        #         curr_time = time.time()
        #         msg_send = [delay_up,curr_time,-1*np.ones([self.Np,])]
        #         msg_bytes_send = pickle.dumps(msg_send)
        #         try:
        #             await websocket.send(msg_bytes_send)
        #         except websockets.exceptions.ConnectionClosedOK:
        #             print("Delay exceeded the specified range. Connection closed by the client",flush=True)
        #         rs.mset({f'veh{self.veh_id}_connect':pickle.dumps(0)})



class RedisMessage:
    def __init__(self,redis,cav_id):
        self.veh_id = cav_id
        self.rs = redis

    # # 用于从Redis中接收预期的消息
    # def receive_expect_msg(self):
    #     x_expect_bytes_from_redis = self.rs.mget('x_expect')[0]
    #     v_expect_bytes_from_redis = self.rs.mget('v_expect')[0]
    #     # 使用pickle.loads反序列化为python对象
    #     x_expect = pickle.loads(x_expect_bytes_from_redis)
    #     v_expect = pickle.loads(v_expect_bytes_from_redis)

    #     return x_expect,v_expect

    # def send_assume_msg(self,Pa,Va,curr_step):
    #     # 序列化
    #     Pa_data = pickle.dumps(Pa.squeeze())
    #     Va_data = pickle.dumps(Va.squeeze())
    #     curr_step_data = pickle.dumps(curr_step)
    #     # 构建字典映射
    #     assume_dict = {f'Pa-{self.veh_id}':Pa_data,
    #                    f'Va-{self.veh_id}':Va_data,
    #                    f'curr_step-{self.veh_id}':curr_step_data
    #                    }
    #     # 将序列化后的数据存入Redis中
    #     self.rs.mset(assume_dict)
