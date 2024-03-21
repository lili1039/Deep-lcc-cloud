import asyncio
import numpy as np
import pickle
import time
import redis
import math
import websockets

def dDeeP_LCC(timestep,n_cav,cav_id,Uip,Yip,Uif,Yif,Eip,Eif,ui_ini,yi_ini,ei_ini,\
                    g_initial,mu_initial,eta_initial,phi_initial,theta_initial,\
                lambda_yi,u_limit,s_limit,rho,KKT_vert,Hz_vert,rs):
    
    # stoping criterion
    iteration_num = 250
    # 300
    error_absolute = 0.1
    error_relative = 1e-3

    # problem size
    m = ui_ini.ndim         # the size of control input of each subsystem
    p = yi_ini.shape[0]     # the size of output of each subsystem
    
    # time horizon
    Tini = int(Uip.shape[0]/m)
    N = int(Uif.shape[0]/m)
    T = int(Uip.shape[1]+Tini+N-1)

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
        print('In deep-lcc, interation_num = ',k,flush=True)
        # Step1: update g
            # updata eta_bar
            # 对于i=1~n-1的CAV计算eta_bar并发送给后一个CAVi+1
        if cav_id != n_cav-1:
            eta_bar = eta - rho*K@Yif@z
            rs.mset({f'eta_bar_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(eta_bar)})

        if cav_id != 0:
            read_redis_start = time.time()
            while True:
                if rs.mget(f'eta_bar_in_CAV_{cav_id-1}_Step_{timestep}_Iter_{k}')[0] != None:
                    eta_bar_former = pickle.loads(rs.mget(f'eta_bar_in_CAV_{cav_id-1}_Step_{timestep}_Iter_{k}')[0])
                    break
                if time.time()-read_redis_start > 3:
                    print('Read redis 1 time out!')
                    break
      
            
            # 计算qg
        if cav_id == 0:
            # qg 的结果是一维数组
            qg = -lambda_yi*Yip.T@yi_ini + mu/2 - rho*z/2 - \
                Yif.T@P.T@phi/2 - Uif.T@theta/2 - rho*Yif.T@P.T@s/2 - rho*Uif.T@u/2
        else:
            qg = -lambda_yi*Yip.T@yi_ini + mu/2 - rho*z/2 + Eif.T@eta_bar_former/2 - \
                Yif.T@P.T@phi/2 - Uif.T@theta/2 - rho*Yif.T@P.T@s/2 - rho*Uif.T@u/2

            # 计算g+
        temp1 = np.hstack((-qg,beqg))
        temp2 = KKT_vert@temp1
        g_plus = temp2[0:T-Tini-N+1]
        

        # Step2: parallel update z/s/u & mu/eta/phi/theta
            # update epsilon_bar
            # 对于i=2~n的CAV更新epsilon_bar后发送给CAVi-1
            # 对于i=2~n的CAV，同时发送Eif给CAVi-1，便于后面计算error_dual2和tolerence_dual2
        if cav_id != 0:
            epsilon_bar = Eif@g_plus
            rs.mset({f'epsilon_bar_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(epsilon_bar)})
            rs.mset({f'Eif_in_CAV_{cav_id}':pickle.dumps(Eif)})

        if cav_id != n_cav-1:
            read_redis_start = time.time()
            while True:
                if rs.mget(f'epsilon_bar_in_CAV_{cav_id+1}_Step_{timestep}_Iter_{k}')[0] != None:
                    epsilon_bar_latter = pickle.loads(rs.mget(f'epsilon_bar_in_CAV_{cav_id+1}_Step_{timestep}_Iter_{k}')[0])
                    break
                if time.time()-read_redis_start > 3:
                    print('Read redis 2 time out!')
                    break

            # update
        if cav_id != n_cav-1:
            qz = -mu/2 - rho/2*g_plus - \
                Yif.T@K.T@eta/2 -rho/2*Yif.T@K.T@epsilon_bar_latter
            z_plus = -Hz_vert@qz
            
            s_temp = P@Yif@g_plus - phi/rho
            s_plus = np.minimum(np.maximum(s_temp,s_limit[0]*np.ones(N)),s_limit[1]*np.ones(N)) # projection in the box constraint

            u_temp = Uif@g_plus-theta/rho
            u_plus = np.minimum(np.maximum(u_temp,u_limit[0]*np.ones(N)),u_limit[1]*np.ones(N)) # projection in the box constraint

            mu_plus = mu + rho*(g_plus-z_plus)
            eta_plus = eta + rho*(epsilon_bar_latter-K@Yif@z_plus)
            phi_plus = phi + rho*(s_plus - P@Yif@g_plus)
            theta_plus = theta + rho*(u_plus - Uif@g_plus)

            # 计算迭代停止条件并存入数据库
            error_pri1 = np.linalg.norm(g_plus - z_plus)
            rs.mset({f'error_pri1_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_pri1)})

            tolerence_pri1 = math.sqrt(g_plus.shape[0])*error_absolute + \
                error_relative*max(np.linalg.norm(g_plus),np.linalg.norm(z_plus))
            rs.mset({f'tolerence_pri1_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_pri1)})
            
            error_dual1 = rho*np.linalg.norm(z_plus-z)
            rs.mset({f'error_dual1_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_dual1)})

            tolerence_dual1 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(mu_plus)
            rs.mset({f'tolerence_dual1_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_dual1)})
  
            error_pri2 = np.linalg.norm(epsilon_bar_latter - K@Yif@z_plus)
            rs.mset({f'error_pri2_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_pri2)})
        
            tolerence_pri2 = math.sqrt((epsilon_bar_latter).shape[0])*error_absolute + \
                error_relative*max(np.linalg.norm(epsilon_bar_latter),np.linalg.norm(K@Yif@z_plus))
            rs.mset({f'tolerence_pri2_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_pri2)})

            Eif_latter = pickle.loads(rs.mget(f'Eif_in_CAV_{cav_id+1}')[0])
            error_dual2 = rho*np.linalg.norm(Eif_latter.T@K@Yif@(z_plus-z))
            rs.mset({f'error_dual2_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_dual2)})

            tolerence_dual2 = math.sqrt((Eif_latter.T@K@Yif@(z_plus-z)).shape[0])*error_absolute + error_relative*np.linalg.norm(Eif_latter.T@eta_plus)  
            rs.mset({f'tolerence_dual2_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_dual2)})
            
            error_pri3 = np.linalg.norm(s_plus - P@Yif@g_plus)
            rs.mset({f'error_pri3_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_pri3)})
        
            tolerence_pri3 = math.sqrt(s_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(s_plus),np.linalg.norm(P@Yif@g_plus))
            rs.mset({f'tolerence_pri3_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_pri3)})
        
            error_dual3 = rho*np.linalg.norm(Yif.T@P.T@(s_plus-s))
            rs.mset({f'error_dual3_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_dual3)})
        
            tolerence_dual3 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Yif.T@P.T@phi_plus)
            rs.mset({f'tolerence_dual3_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_dual3)})

            error_pri4 =  np.linalg.norm(u_plus-Uif@g_plus)
            rs.mset({f'error_pri4_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_pri4)})
        
            tolerence_pri4 = math.sqrt(u_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(Uif@g_plus),np.linalg.norm(u_plus))
            rs.mset({f'tolerence_pri4_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_pri4)})
        
            error_dual4 = rho*np.linalg.norm(Uif.T@(u_plus-u))
            rs.mset({f'error_dual4_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_dual4)})
        
            tolerence_dual4 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Uif.T@theta_plus)
            rs.mset({f'tolerence_dual4_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_dual4)})

            eta = eta_plus
        
        else:
            qz = -mu/2 - rho/2*g_plus
            z_plus = -Hz_vert@qz
            
            s_temp = P@Yif@g_plus - phi/rho
            s_plus = np.minimum(np.maximum(s_temp,s_limit[0]*np.ones(N)),s_limit[1]*np.ones(N)) # projection in the box constraint

            u_temp = Uif@g_plus-theta/rho
            u_plus = np.minimum(np.maximum(u_temp,u_limit[0]*np.ones(N)),u_limit[1]*np.ones(N)) # projection in the box constraint

            mu_plus = mu + rho*(g_plus-z_plus)
            phi_plus = phi + rho*(s_plus - P@Yif@g_plus)
            theta_plus = theta + rho*(u_plus - Uif@g_plus)

            # 计算迭代停止条件并存入数据库
            error_pri1 = np.linalg.norm(g_plus - z_plus)
            rs.mset({f'error_pri1_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_pri1)})

            tolerence_pri1 = math.sqrt(g_plus.shape[0])*error_absolute + \
                error_relative*max(np.linalg.norm(g_plus),np.linalg.norm(z_plus))
            rs.mset({f'tolerence_pri1_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_pri1)})
            
            error_dual1 = rho*np.linalg.norm(z_plus-z)
            rs.mset({f'error_dual1_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_dual1)})

            tolerence_dual1 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(mu_plus)
            rs.mset({f'tolerence_dual1_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_dual1)})
            
            error_pri3 = np.linalg.norm(s_plus - P@Yif@g_plus)
            rs.mset({f'error_pri3_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_pri3)})
        
            tolerence_pri3 = math.sqrt(s_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(s_plus),np.linalg.norm(P@Yif@g_plus))
            rs.mset({f'tolerence_pri3_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_pri3)})
        
            error_dual3 = rho*np.linalg.norm(Yif.T@P.T@(s_plus-s))
            rs.mset({f'error_dual3_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_dual3)})
        
            tolerence_dual3 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Yif.T@P.T@phi_plus)
            rs.mset({f'tolerence_dual3_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_dual3)})

            error_pri4 =  np.linalg.norm(u_plus-Uif@g_plus)
            rs.mset({f'error_pri4_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_pri4)})
        
            tolerence_pri4 = math.sqrt(u_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(Uif@g_plus),np.linalg.norm(u_plus))
            rs.mset({f'tolerence_pri4_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_pri4)})
        
            error_dual4 = rho*np.linalg.norm(Uif.T@(u_plus-u))
            rs.mset({f'error_dual4_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(error_dual4)})
        
            tolerence_dual4 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Uif.T@theta_plus)
            rs.mset({f'tolerence_dual4_in_CAV_{cav_id}_Step_{timestep}_Iter_{k}':pickle.dumps(tolerence_dual4)})

        g = g_plus
        z = z_plus
        u = u_plus
        s = s_plus
        mu = mu_plus
        phi = phi_plus
        theta = theta_plus

        # Check stopping criterion
        if StopCriteria(k,rs,n_cav,timestep):
            print(f'Iteration stop in step {k}')
            break

    # Record optimal value
    real_iter_num = k
    g_opt = g
    mu_opt = mu
    eta_opt = eta
    phi_opt = phi
    theta_opt = theta
    u_opt = u
    
    return u_opt,g_opt,mu_opt,eta_opt,phi_opt,theta_opt,real_iter_num

def StopCriteria(k,rs,n_cav,timestep):
    error_pri1 = 0
    tolerence_pri1 = 0

    error_dual1 = 0
    tolerence_dual1 = 0

    error_pri2 = 0
    tolerence_pri2 = 0

    error_dual2 = 0
    tolerence_dual2 = 0

    error_pri3 = 0
    tolerence_pri3 = 0

    error_dual3 = 0
    tolerence_dual3 = 0

    error_pri4 = 0
    tolerence_pri4 = 0

    error_dual4 = 0
    tolerence_dual4 = 0    

    for i in range(n_cav):
        read_redis_start = time.time()
        while True:
            if rs.mget(f'error_pri1_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None and rs.mget(f'tolerence_pri1_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None \
            and rs.mget(f'error_dual1_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None and rs.mget(f'tolerence_dual1_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None :
                break

            if time.time()-read_redis_start > 3:
                print('Read redis 3 time out!')
                break
        
        error_pri1 = error_pri1 + pickle.loads(rs.mget(f'error_pri1_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
        tolerence_pri1 = tolerence_pri1 + pickle.loads(rs.mget(f'tolerence_pri1_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
        error_dual1 = error_dual1 + pickle.loads(rs.mget(f'error_dual1_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
        tolerence_dual1 = tolerence_dual1 + pickle.loads(rs.mget(f'tolerence_dual1_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])

    if k == 249:
        print(error_pri1,tolerence_pri1,error_dual1,tolerence_dual1,flush=True)

    
    if error_pri1 > tolerence_pri1 or error_dual1 > tolerence_dual1:
        return False
    
    for i in range(n_cav):
        if i == n_cav-1:
            break

        read_redis_start = time.time()
        while True:
            if rs.mget(f'error_pri2_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None and rs.mget(f'tolerence_pri2_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None \
            and rs.mget(f'error_dual2_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None and rs.mget(f'tolerence_dual2_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None :
                break
            if time.time()-read_redis_start > 3:
                print('Read redis 4 time out!')
                break

        error_pri2 = error_pri2 + pickle.loads(rs.mget(f'error_pri2_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
        tolerence_pri2 = tolerence_pri2 + pickle.loads(rs.mget(f'tolerence_pri2_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
        error_dual2 = error_dual2 + pickle.loads(rs.mget(f'error_dual2_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
        tolerence_dual2 = tolerence_dual2 + pickle.loads(rs.mget(f'tolerence_dual2_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
    if k == 249:
        print(error_pri2,tolerence_pri2,error_dual2,tolerence_dual2,flush=True)
    if error_pri2 > tolerence_pri2 or error_dual2 > tolerence_dual2:
        return False
    
    for i in range(n_cav):
        read_redis_start = time.time()
        while True:
            if rs.mget(f'error_pri3_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None and rs.mget(f'tolerence_pri3_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None \
            and rs.mget(f'error_dual3_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None and rs.mget(f'tolerence_dual3_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None :
                break
            if time.time()-read_redis_start > 3:
                print('Read redis 5 time out!')
                break
        
        error_pri3 = error_pri3 + pickle.loads(rs.mget(f'error_pri3_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
        tolerence_pri3 = tolerence_pri3 + pickle.loads(rs.mget(f'tolerence_pri3_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
        error_dual3 = error_dual3 + pickle.loads(rs.mget(f'error_dual3_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
        tolerence_dual3 = tolerence_dual3 + pickle.loads(rs.mget(f'tolerence_dual3_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
    if k == 249:
        print(error_pri3,tolerence_pri3,error_dual3,tolerence_dual3,flush=True)
    if error_pri3 > tolerence_pri3 or error_dual3 > tolerence_dual3:
        return False   

    
    for i in range(n_cav):
        read_redis_start = time.time()
        while True:
            if rs.mget(f'error_pri4_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None and rs.mget(f'tolerence_pri4_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None \
            and rs.mget(f'error_dual4_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None and rs.mget(f'tolerence_dual4_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0] != None :
                break
            if time.time()-read_redis_start > 3:
                print('Read redis 6 time out!')
                break

        error_pri4 = error_pri4 + pickle.loads(rs.mget(f'error_pri4_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
        tolerence_pri4 = tolerence_pri4 + pickle.loads(rs.mget(f'tolerence_pri4_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
        error_dual4 = error_dual4 + pickle.loads(rs.mget(f'error_dual4_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])
        tolerence_dual4 = tolerence_dual4 + pickle.loads(rs.mget(f'tolerence_dual4_in_CAV_{i}_Step_{timestep}_Iter_{k}')[0])

    if k == 249:
        print(error_pri4,tolerence_pri4,error_dual4,tolerence_dual4,flush=True)
    if error_pri4 > tolerence_pri4 or error_dual4 > tolerence_dual4:
        return False
    
    return True

class SubsystemParam:
    def __init__(self):
        # 仿真时间参数
        self.total_time = 40
        self.Tstep = 0.05
        self.total_time_step = int(self.total_time/self.Tstep)
        
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
        self.lambda_gi       = 0

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
        # 数据库连接
        rs = redis.StrictRedis(host='172.18.0.1',port=6379,db=2,password="chlpw1039") 
        # 数据库标志该子系统已连接
        rs.mset({f'Subsystem{self.cav_id}_connect':pickle.dumps(1)})

        # 获取数据
        # 收到客户端的消息(初始参数和数据)
        try:
            msg_bytes_recv  = await websocket.recv()
            msg_recv        = pickle.loads(msg_bytes_recv)
            print("Subsystem {}: data loaded".format(self.cav_id),flush=True)
            msg_send = True
            msg_bytes_send = pickle.dumps(msg_send)
            await websocket.send(msg_bytes_send)
        except websockets.ConnectionClosed as e:
            print('Connect closed by the client.',e.code,flush=True)
            await asyncio.sleep(0.01)
        
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
        self.rho             = msg_recv[11]
        self.lambda_yi       = msg_recv[12]
        self.lambda_gi       = msg_recv[13]

        n_cav_bytes = rs.mget('n_cav')[0]
        n_cav = pickle.loads(n_cav_bytes)

        # problem size
        m = self.uini.ndim         # the size of control input of each subsystem
        p = self.yini.shape[0]     # the size of output of each subsystem
        # time horizon
        Tini = int(self.Uip.shape[0]/m)
        N = int(self.Uif.shape[0]/m)
        T = int(self.Uip.shape[1]+Tini+N-1)

        # other useful matrix
        weight_v        = 1     # weight coefficient for velocity error
        weight_s        = 0.5   # weight coefficient for spacing error   
        weight_u        = 0.1   # weight coefficient for control input


        K = np.kron(np.eye(N),np.append(np.zeros(int(self.n_vehicle_sub-1)),[1,0]))
        P = np.kron(np.eye(N),np.append(np.zeros(int(self.n_vehicle_sub)),[1]))

        Qi = np.diagflat(np.append(weight_v*np.ones(int(self.n_vehicle_sub)),[weight_s]))
        Qi_stack = np.kron(np.eye(N),Qi)
        Ri = weight_u
        Ri_stack = np.kron(np.eye(N),Ri)

            # KKT_vert
        if self.cav_id == 0: # the first subsystem has different Hgi
            Hg = self.Yif.T@Qi_stack@self.Yif+self.Uif.T@Ri_stack@self.Uif+\
                self.lambda_gi*np.eye(T-Tini-N+1)+self.lambda_yi*self.Yip.T@self.Yip+\
                self.rho/2*(np.eye(T-Tini-N+1)+self.Yif.T@P.T@P@self.Yif+self.Uif.T@self.Uif)
            Aeqg = np.vstack((self.Uip,self.Eip,self.Eif))
        else:
            Hg = self.Yif.T@Qi_stack@self.Yif+self.Uif.T@Ri_stack@self.Uif+\
                self.lambda_gi*np.eye(T-Tini-N+1)+self.lambda_yi*self.Yip.T@self.Yip+\
                self.rho/2*(np.eye(T-Tini-N+1)+self.Eif.T@self.Eif+self.Yif.T@P.T@P@self.Yif+self.Uif.T@self.Uif)
            Aeqg = np.vstack((self.Uip,self.Eip))
        
        Phi = np.vstack((np.hstack((Hg,Aeqg.T)),np.hstack((Aeqg,np.zeros([Aeqg.shape[0],Aeqg.shape[0]])))))
        self.KKT_vert = np.linalg.inv(Phi)

        # Hz_vert
        if self.cav_id != n_cav-1:
            Hz = self.rho/2*np.eye(T-Tini-N+1)+self.rho/2*self.Yif.T@K.T@K@self.Yif
        else:
            Hz = self.rho/2*np.eye(T-Tini-N+1)
        self.Hz_vert = np.linalg.inv(Hz)

        # Initial dual variables
        g_initial = np.zeros(T-Tini-N+1)
        mu_initial = np.zeros(T-Tini-N+1)
        eta_initial = np.zeros(N)
        phi_initial = np.zeros(N)
        theta_initial = np.zeros(N)

        # record
        computation_time = np.zeros(self.total_time_step - Tini)
        iteration_num = np.zeros(self.total_time_step - Tini)


        # # Construct distributed data for ADMM computation
        # for k in range(self.Tini-1,self.total_time_step-1):
        for k in range(Tini-1,Tini):
            start_time = time.time()
            u_opt,g_opt,mu_opt,eta_opt,phi_opt,theta_opt,real_iter_num = dDeeP_LCC(k,n_cav,self.cav_id,self.Uip,self.Yip,self.Uif,\
                self.Yif,self.Eip,self.Eif,self.uini,self.yini,self.eini,\
                g_initial,mu_initial,eta_initial,phi_initial,theta_initial,\
                self.lambda_yi,self.u_limit,self.s_limit,self.rho,self.KKT_vert,self.Hz_vert,rs)
            end_time = time.time()

            # # give input signal to client
            print('I am ready to send u!',flush = True)
            print('u=',u_opt[0],flush = True)
            # msg_send = u_opt[0]
            # msg_bytes_send = pickle.dumps(msg_send)
            # try:
            #     await websocket.send(msg_bytes_send)
            # except websockets.ConnectionClosedOK:
            #     print('Connection closed by the client',flush = True)

            # g_initial = g_opt
            # mu_initial = mu_opt
            # eta_initial = eta_opt
            # phi_initial = phi_opt
            # theta_initial = theta_opt

            # computation_time[k-Tini+1] = end_time-start_time
            # iteration_num[k-Tini+1] = real_iter_num
            # print('Step:%d'%(k+1))
            # print(f'Average computation time of CAV{self.cav_id}: ','%.4f s'%computation_time[k-Tini+1])

            # # receive output msg(einput and y) from client
            # msg_bytes_recv = await websocket.recv()
            # msg_recv = pickle.loads(msg_bytes_recv)
            # e_input = msg_recv[0]
            # y = msg_recv[1]

            # # construct eini
            # self.eini = np.append(self.eini[1:],e_input)
            # # construct uini
            # self.uini = np.append(self.uini[1:],u_opt[0])
            # # construct yini
            # self.yini = np.hstack((self.yini[:,1:],y.reshape([p,1])))