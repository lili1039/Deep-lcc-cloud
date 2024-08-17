import asyncio
import numpy as np
import pickle
import time
import redis
import math
import websockets


# Generate a Hankel matrix of order L
def Hankel_matrix(u,L):
    m = u.ndim      # the dimension of signal
    T = u.shape[0]      # the length of data
    U = np.zeros([m*L,T-L+1])

    for i in range(L):
        U[i*m:(i+1)*m,:] = u[i:(T-L+1+i)]

    return U

def whetherPE(U):
    rows,cols = U.shape
    if np.linalg.matrix_rank(U) == rows:
        return True
    else:
        return False

def dDeeP_LCC(timestep,n_cav,cav_id,Uip,Yip,Uif,Yif,Eip,Eif,ui_ini,yi_ini,ei_ini,\
                    g_initial,mu_initial,eta_initial,phi_initial,theta_initial,\
                lambda_yi,u_limit,s_limit,rho,KKT_vert,Hz_vert,rs):
    
    # stoping criterion
    iteration_num = 300
    error_absolute = 0.1
    error_relative = 1e-3

    # problem size
    m = ui_ini.ndim         # the size of control input of each subsystem
    p = yi_ini.shape[0]     # the size of output of each subsystem
    
    # time horizon
    Tini = int(Uip.shape[0]/m)
    N = int(Uif.shape[0]/m)
    T = int(Uip.shape[1]+Tini+N-1)
    kappa = int(Uip.shape[1])

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
        # print('In deep-lcc, interation_num = ',k,flush=True)
        # Step1: update g
            # updata eta_bar
            # 对于i=1~n-1的CAV计算eta_bar并发送给后一个CAVi+1
        if cav_id != n_cav-1:
            eta_bar = eta - rho*K@Yif@z
            rs.mset({f'eta_bar_{cav_id}_{timestep}_{k}':pickle.dumps(eta_bar)})

        if cav_id != 0:
            while True:
                if rs.mget(f'eta_bar_{cav_id-1}_{timestep}_{k}')[0] != None:
                    eta_bar_former = pickle.loads(rs.mget(f'eta_bar_{cav_id-1}_{timestep}_{k}')[0])
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
        g_plus = temp2[0:kappa]
        

        # Step2: parallel update z/s/u & mu/eta/phi/theta
            # update epsilon_bar
            # 对于i=2~n的CAV更新epsilon_bar后发送给CAVi-1
            # 对于i=2~n的CAV，同时发送Eif给CAVi-1，便于后面计算error_dual2和tolerence_dual2
        if cav_id != 0:
            epsilon_bar = Eif@g_plus
            rs.mset({f'epsilon_bar_{cav_id}_{timestep}_{k}':pickle.dumps(epsilon_bar)})

        if cav_id != n_cav-1:
            while True:
                if rs.mget(f'epsilon_bar_{cav_id+1}_{timestep}_{k}')[0] != None:
                    epsilon_bar_latter = pickle.loads(rs.mget(f'epsilon_bar_{cav_id+1}_{timestep}_{k}')[0])
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
            rs.mset({f'error_pri1_{cav_id}_{timestep}_{k}':pickle.dumps(error_pri1)})

            tolerence_pri1 = math.sqrt(g_plus.shape[0])*error_absolute + \
                error_relative*max(np.linalg.norm(g_plus),np.linalg.norm(z_plus))
            rs.mset({f'tolerence_pri1_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_pri1)})
            
            error_dual1 = rho*np.linalg.norm(z_plus-z)
            rs.mset({f'error_dual1_{cav_id}_{timestep}_{k}':pickle.dumps(error_dual1)})

            tolerence_dual1 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(mu_plus)
            rs.mset({f'tolerence_dual1_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_dual1)})
  
            error_pri2 = np.linalg.norm(epsilon_bar_latter - K@Yif@z_plus)
            rs.mset({f'error_pri2_{cav_id}_{timestep}_{k}':pickle.dumps(error_pri2)})
        
            tolerence_pri2 = math.sqrt((epsilon_bar_latter).shape[0])*error_absolute + \
                error_relative*max(np.linalg.norm(epsilon_bar_latter),np.linalg.norm(K@Yif@z_plus))
            rs.mset({f'tolerence_pri2_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_pri2)})

            Eif_latter = pickle.loads(rs.mget(f'Eif_in_CAV_{cav_id+1}')[0])
            error_dual2 = rho*np.linalg.norm(Eif_latter.T@K@Yif@(z_plus-z))
            rs.mset({f'error_dual2_{cav_id}_{timestep}_{k}':pickle.dumps(error_dual2)})

            tolerence_dual2 = math.sqrt((Eif_latter.T@K@Yif@(z_plus-z)).shape[0])*error_absolute + error_relative*np.linalg.norm(Eif_latter.T@eta_plus)  
            rs.mset({f'tolerence_dual2_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_dual2)})
            
            error_pri3 = np.linalg.norm(s_plus - P@Yif@g_plus)
            rs.mset({f'error_pri3_{cav_id}_{timestep}_{k}':pickle.dumps(error_pri3)})
        
            tolerence_pri3 = math.sqrt(s_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(s_plus),np.linalg.norm(P@Yif@g_plus))
            rs.mset({f'tolerence_pri3_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_pri3)})
        
            error_dual3 = rho*np.linalg.norm(Yif.T@P.T@(s_plus-s))
            rs.mset({f'error_dual3_{cav_id}_{timestep}_{k}':pickle.dumps(error_dual3)})
        
            tolerence_dual3 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Yif.T@P.T@phi_plus)
            rs.mset({f'tolerence_dual3_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_dual3)})

            error_pri4 =  np.linalg.norm(u_plus-Uif@g_plus)
            rs.mset({f'error_pri4_{cav_id}_{timestep}_{k}':pickle.dumps(error_pri4)})
        
            tolerence_pri4 = math.sqrt(u_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(Uif@g_plus),np.linalg.norm(u_plus))
            rs.mset({f'tolerence_pri4_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_pri4)})
        
            error_dual4 = rho*np.linalg.norm(Uif.T@(u_plus-u))
            rs.mset({f'error_dual4_{cav_id}_{timestep}_{k}':pickle.dumps(error_dual4)})
        
            tolerence_dual4 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Uif.T@theta_plus)
            rs.mset({f'tolerence_dual4_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_dual4)})

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
            rs.mset({f'error_pri1_{cav_id}_{timestep}_{k}':pickle.dumps(error_pri1)})

            tolerence_pri1 = math.sqrt(g_plus.shape[0])*error_absolute + \
                error_relative*max(np.linalg.norm(g_plus),np.linalg.norm(z_plus))
            rs.mset({f'tolerence_pri1_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_pri1)})
            
            error_dual1 = rho*np.linalg.norm(z_plus-z)
            rs.mset({f'error_dual1_{cav_id}_{timestep}_{k}':pickle.dumps(error_dual1)})

            tolerence_dual1 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(mu_plus)
            rs.mset({f'tolerence_dual1_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_dual1)})
            
            error_pri3 = np.linalg.norm(s_plus - P@Yif@g_plus)
            rs.mset({f'error_pri3_{cav_id}_{timestep}_{k}':pickle.dumps(error_pri3)})
        
            tolerence_pri3 = math.sqrt(s_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(s_plus),np.linalg.norm(P@Yif@g_plus))
            rs.mset({f'tolerence_pri3_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_pri3)})
        
            error_dual3 = rho*np.linalg.norm(Yif.T@P.T@(s_plus-s))
            rs.mset({f'error_dual3_{cav_id}_{timestep}_{k}':pickle.dumps(error_dual3)})
        
            tolerence_dual3 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Yif.T@P.T@phi_plus)
            rs.mset({f'tolerence_dual3_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_dual3)})

            error_pri4 =  np.linalg.norm(u_plus-Uif@g_plus)
            rs.mset({f'error_pri4_{cav_id}_{timestep}_{k}':pickle.dumps(error_pri4)})
        
            tolerence_pri4 = math.sqrt(u_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(Uif@g_plus),np.linalg.norm(u_plus))
            rs.mset({f'tolerence_pri4_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_pri4)})
        
            error_dual4 = rho*np.linalg.norm(Uif.T@(u_plus-u))
            rs.mset({f'error_dual4_{cav_id}_{timestep}_{k}':pickle.dumps(error_dual4)})
        
            tolerence_dual4 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Uif.T@theta_plus)
            rs.mset({f'tolerence_dual4_{cav_id}_{timestep}_{k}':pickle.dumps(tolerence_dual4)})

        g = g_plus
        z = z_plus
        u = u_plus
        s = s_plus
        mu = mu_plus
        phi = phi_plus
        theta = theta_plus

        # Check stopping criterion
        if StopCriteria(k,rs,n_cav,timestep):
            break

    # Record optimal value
    real_iter_num = k+1
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
        while True:
            if rs.mget(f'error_pri1_{i}_{timestep}_{k}')[0] != None and rs.mget(f'tolerence_pri1_{i}_{timestep}_{k}')[0] != None \
            and rs.mget(f'error_dual1_{i}_{timestep}_{k}')[0] != None and rs.mget(f'tolerence_dual1_{i}_{timestep}_{k}')[0] != None :
                break
        
        error_pri1 = error_pri1 + pickle.loads(rs.mget(f'error_pri1_{i}_{timestep}_{k}')[0])
        tolerence_pri1 = tolerence_pri1 + pickle.loads(rs.mget(f'tolerence_pri1_{i}_{timestep}_{k}')[0])
        error_dual1 = error_dual1 + pickle.loads(rs.mget(f'error_dual1_{i}_{timestep}_{k}')[0])
        tolerence_dual1 = tolerence_dual1 + pickle.loads(rs.mget(f'tolerence_dual1_{i}_{timestep}_{k}')[0])

    if error_pri1 > tolerence_pri1 or error_dual1 > tolerence_dual1:
        return False
    
    for i in range(n_cav):
        if i == n_cav-1:
            break
        while True:
            if rs.mget(f'error_pri2_{i}_{timestep}_{k}')[0] != None and rs.mget(f'tolerence_pri2_{i}_{timestep}_{k}')[0] != None \
            and rs.mget(f'error_dual2_{i}_{timestep}_{k}')[0] != None and rs.mget(f'tolerence_dual2_{i}_{timestep}_{k}')[0] != None :
                break

        error_pri2 = error_pri2 + pickle.loads(rs.mget(f'error_pri2_{i}_{timestep}_{k}')[0])
        tolerence_pri2 = tolerence_pri2 + pickle.loads(rs.mget(f'tolerence_pri2_{i}_{timestep}_{k}')[0])
        error_dual2 = error_dual2 + pickle.loads(rs.mget(f'error_dual2_{i}_{timestep}_{k}')[0])
        tolerence_dual2 = tolerence_dual2 + pickle.loads(rs.mget(f'tolerence_dual2_{i}_{timestep}_{k}')[0])

    if error_pri2 > tolerence_pri2 or error_dual2 > tolerence_dual2:
        return False
    
    for i in range(n_cav):
        # read_redis_start = time.time()
        while True:
            if rs.mget(f'error_pri3_{i}_{timestep}_{k}')[0] != None and rs.mget(f'tolerence_pri3_{i}_{timestep}_{k}')[0] != None \
            and rs.mget(f'error_dual3_{i}_{timestep}_{k}')[0] != None and rs.mget(f'tolerence_dual3_{i}_{timestep}_{k}')[0] != None :
                break
        
        error_pri3 = error_pri3 + pickle.loads(rs.mget(f'error_pri3_{i}_{timestep}_{k}')[0])
        tolerence_pri3 = tolerence_pri3 + pickle.loads(rs.mget(f'tolerence_pri3_{i}_{timestep}_{k}')[0])
        error_dual3 = error_dual3 + pickle.loads(rs.mget(f'error_dual3_{i}_{timestep}_{k}')[0])
        tolerence_dual3 = tolerence_dual3 + pickle.loads(rs.mget(f'tolerence_dual3_{i}_{timestep}_{k}')[0])

    if error_pri3 > tolerence_pri3 or error_dual3 > tolerence_dual3:
        return False   

    
    for i in range(n_cav):
        # read_redis_start = time.time()
        while True:
            if rs.mget(f'error_pri4_{i}_{timestep}_{k}')[0] != None and rs.mget(f'tolerence_pri4_{i}_{timestep}_{k}')[0] != None \
            and rs.mget(f'error_dual4_{i}_{timestep}_{k}')[0] != None and rs.mget(f'tolerence_dual4_{i}_{timestep}_{k}')[0] != None :
                break

        error_pri4 = error_pri4 + pickle.loads(rs.mget(f'error_pri4_{i}_{timestep}_{k}')[0])
        tolerence_pri4 = tolerence_pri4 + pickle.loads(rs.mget(f'tolerence_pri4_{i}_{timestep}_{k}')[0])
        error_dual4 = error_dual4 + pickle.loads(rs.mget(f'error_dual4_{i}_{timestep}_{k}')[0])
        tolerence_dual4 = tolerence_dual4 + pickle.loads(rs.mget(f'tolerence_dual4_{i}_{timestep}_{k}')[0])

    if error_pri4 > tolerence_pri4 or error_dual4 > tolerence_dual4:
        return False
    
    return True

class SubsystemParam:
    def __init__(self):
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
        start_time = time.time()
        # 数据库连接
        rs = redis.StrictRedis(host='172.18.0.1',port=6379,db=2,password="chlpw1039") 
        # 数据库标志该子系统已连接
        rs.mset({f'Subsystem{self.cav_id}_connect':pickle.dumps(1)})

        # 从websocket获取初始参数和数据
        msg_bytes_recv  = await websocket.recv()
        msg_recv        = pickle.loads(msg_bytes_recv)
        msg_send = True
        msg_bytes_send = pickle.dumps(msg_send)
        await websocket.send(msg_bytes_send)

        self.cav_id          = msg_recv[0]
        self.n_vehicle_sub   = msg_recv[1]
        self.uini            = msg_recv[2]
        self.eini            = msg_recv[3]
        self.yini            = msg_recv[4]
        curr_step            = msg_recv[5]
        Hankel_update_flag   = msg_recv[6]
        print(f"Subsystem {self.cav_id}: Step {curr_step} data loaded",flush=True)

        # 从数据库获取初始参数和数据
        self.Uip             = pickle.loads(rs.mget(f'Uip_in_CAV_{self.cav_id}')[0])
        self.Uif             = pickle.loads(rs.mget(f'Uif_in_CAV_{self.cav_id}')[0])
        self.Eip             = pickle.loads(rs.mget(f'Eip_in_CAV_{self.cav_id}')[0])
        self.Eif             = pickle.loads(rs.mget(f'Eif_in_CAV_{self.cav_id}')[0])
        self.Yip             = pickle.loads(rs.mget(f'Yip_in_CAV_{self.cav_id}')[0])
        self.Yif             = pickle.loads(rs.mget(f'Yif_in_CAV_{self.cav_id}')[0])
        self.rho             = pickle.loads(rs.mget(f'rho_in_CAV_{self.cav_id}')[0])
        self.lambda_yi       = pickle.loads(rs.mget(f'lambda_yi_in_CAV_{self.cav_id}')[0])
        self.lambda_gi       = pickle.loads(rs.mget(f'lambda_gi_in_CAV_{self.cav_id}')[0])
        self.KKT_vert        = pickle.loads(rs.mget(f'KKT_vert_in_CAV_{self.cav_id}')[0])
        self.Hz_vert         = pickle.loads(rs.mget(f'Hz_vert_in_CAV_{self.cav_id}')[0])
        g_initial = pickle.loads(rs.mget(f'g_initial_in_CAV_{self.cav_id}')[0])
        mu_initial = pickle.loads(rs.mget(f'mu_initial_in_CAV_{self.cav_id}')[0])
        eta_initial = pickle.loads(rs.mget(f'eta_initial_in_CAV_{self.cav_id}')[0])
        phi_initial = pickle.loads(rs.mget(f'phi_initial_in_CAV_{self.cav_id}')[0])
        theta_initial = pickle.loads(rs.mget(f'theta_initial_in_CAV_{self.cav_id}')[0])

        n_cav_bytes = rs.mget('n_cav')[0]
        n_cav = pickle.loads(n_cav_bytes)

        Su = pickle.loads(rs.mget(f'Su_in_CAV_{self.cav_id}')[0])
        Se = pickle.loads(rs.mget(f'Se_in_CAV_{self.cav_id}')[0])
        Sy = pickle.loads(rs.mget(f'Sy_in_CAV_{self.cav_id}')[0])

        # problem size
        m = self.uini.ndim         # the size of control input of each subsystem
        p = self.yini.shape[0]     # the size of output of each subsystem
        # time horizon
        Tini = int(self.Uip.shape[0]/m)
        N = int(self.Uif.shape[0]/m)
        Tstep = pickle.loads(rs.mget(f'Tstep')[0])
        max_col = 400

        # update Hankel matrix
        if Hankel_update_flag and curr_step>=N:
            kappa = self.Uip.shape[1]
            print('kappa = ',kappa)
            
            # ds_num = pickle.loads(rs.mget(f'ds_num_in_CAV_{self.cav_id}')[0])
            # ds_flag = pickle.loads(rs.mget(f'ds_flag_in_CAV_{self.cav_id}')[0])
            # ds_timeflag = pickle.loads(rs.mget(f'ds_timeflag_in_CAV_{self.cav_id}')[0])

            # PE的Hankel矩阵的阶数要达到(Tini+N+2*self.n_vehicle_sub)，为了能构建PE的Hankel，u的长度要至少>=阶数
            # L = Tini+N+2*self.n_vehicle_sub
            # if (kappa-ds_flag[ds_num][1])>=L:
            #     # new_u = Su[0,0:curr_step+Tini+1]
            #     new_u = Su[0,ds_timeflag[ds_num][1]:curr_step+Tini]
            #     new_e = Se[0,ds_timeflag[ds_num][1]:curr_step+Tini]
            #     if whetherPE(Hankel_matrix(new_u,L)) and whetherPE(Hankel_matrix(new_e,L)):
            #         # flag = 0
            #         # 上一个数据集满足持续激励后，开始下一个数据集
            #         ds_num = ds_num+1
            #         ds_flag.append([ds_flag[ds_num-1][1],kappa])
            #         ds_timeflag.append([ds_timeflag[ds_num-1][1],curr_step+Tini])
            #         rs.mset({f'ds_num_in_CAV_{self.cav_id}':pickle.dumps(ds_num)})
            #         rs.mset({f'ds_flag_in_CAV_{self.cav_id}':pickle.dumps(ds_flag)})
            #         rs.mset({f'ds_timeflag_in_CAV_{self.cav_id}':pickle.dumps(ds_timeflag)})
            #         print('Satisfy P.E.',flush=True)

            if  kappa< max_col:
                flag = 1
            else:
                # 如果列数已经到达最大容许列数，则减掉前一列，再加上新一列
                # 同时需要修改ds_flag
                # for i in range(ds_num+1):
                #     ds_flag[i] = [j-1 for j in ds_flag[i]]
                # rs.mset({f'ds_flag_in_CAV_{self.cav_id}':pickle.dumps(ds_flag)})
                flag = 0
            
            if flag:
                print('Change Hankel matrix by adding a new column.', flush = True)
                kappa = kappa + 1

                new_up = np.zeros(Tini)
                for j in range(Tini):
                    new_up[j] = Su[0,curr_step-N+j]
                Up_temp = np.hstack((self.Uip,new_up.reshape([int(Tini),1])))

                new_uf = np.zeros(N)
                for j in range(N):
                    new_uf[j] = Su[0,curr_step-N+Tini+j]
                Uf_temp = np.hstack((self.Uif,new_uf.reshape([int(N),1])))

                new_yp = np.zeros(p*Tini)
                for j in range(Tini):
                    new_yp[j*p:(j+1)*p] = Sy[:,curr_step-N+j]
                Yp_temp = np.hstack((self.Yip,new_yp.reshape([int(p*Tini),1])))

                new_yf = np.zeros(p*N)
                for j in range(N):
                    new_yf[j*p:(j+1)*p] = Sy[:,curr_step-N+Tini+j]
                Yf_temp = np.hstack((self.Yif,new_yf.reshape([int(p*N),1])))

                new_ep = np.zeros(Tini)
                for j in range(Tini):
                    new_ep[j] = Se[0,curr_step-N+j]
                Ep_temp = np.hstack((self.Eip,new_ep.reshape([int(Tini),1])))

                new_ef = np.zeros(N)
                for j in range(N):
                    new_ef[j] = Se[0,curr_step-N+Tini+j]
                Ef_temp = np.hstack((self.Eif,new_ef.reshape([int(N),1])))

                # g_initial / mu_initial
                g_initial = np.hstack((g_initial,0))
                mu_initial = np.hstack((mu_initial,0))

            else:
                print('Change Hankel matrix by delete one column and add a new one.', flush = True)

                ds0_end = 231
                new_up = np.zeros(Tini)
                for j in range(Tini):
                    new_up[j] = Su[0,curr_step-N+j]
                # Up_temp = np.hstack((self.Uip[:,1:],new_up.reshape([int(Tini),1])))
                Up_temp = np.hstack((self.Uip[:,0:231],self.Uip[:,232:],new_up.reshape([int(Tini),1])))

                new_uf = np.zeros(N)
                for j in range(N):
                    new_uf[j] = Su[0,curr_step-N+Tini+j]
                # Uf_temp = np.hstack((self.Uif[:,1:],new_uf.reshape([int(N),1])))
                Uf_temp = np.hstack((self.Uif[:,0:231],self.Uif[:,232:],new_uf.reshape([int(N),1])))

                new_yp = np.zeros(p*Tini)
                for j in range(Tini):
                    new_yp[j*p:(j+1)*p] = Sy[:,curr_step-N+j]
                # Yp_temp = np.hstack((self.Yip[:,1:],new_yp.reshape([int(p*Tini),1])))
                Yp_temp = np.hstack((self.Yip[:,0:231],self.Yip[:,232:],new_yp.reshape([int(p*Tini),1])))

                new_yf = np.zeros(p*N)
                for j in range(N):
                    new_yf[j*p:(j+1)*p] = Sy[:,curr_step-N+Tini+j]
                # Yf_temp = np.hstack((self.Yif[:,1:],new_yf.reshape([int(p*N),1])))
                Yf_temp = np.hstack((self.Yif[:,0:231],self.Yif[:,232:],new_yf.reshape([int(p*N),1])))

                new_ep = np.zeros(Tini)
                for j in range(Tini):
                    new_ep[j] = Se[0,curr_step-N+j]
                # Ep_temp = np.hstack((self.Eip[:,1:],new_ep.reshape([int(Tini),1])))
                Ep_temp = np.hstack((self.Eip[:,0:231],self.Eip[:,232:],new_ep.reshape([int(Tini),1])))

                new_ef = np.zeros(N)
                for j in range(N):
                    new_ef[j] = Se[0,curr_step-N+Tini+j]
                # Ef_temp = np.hstack((self.Eif[:,1:],new_ef.reshape([int(N),1])))
                Ef_temp = np.hstack((self.Eif[:,0:231],self.Eif[:,232:],new_ef.reshape([int(N),1])))

                # g_initial / mu_initial
                # g_initial = np.hstack((g_initial[1:],0))
                g_initial = np.hstack((g_initial[0:231],g_initial[232:],0))
                # mu_initial = np.hstack((mu_initial[1:],0))
                mu_initial = np.hstack((mu_initial[0:231],mu_initial[232:],0))

            self.Uip = Up_temp
            rs.mset({f'Uip_in_CAV_{self.cav_id}':pickle.dumps(self.Uip)})
            self.Uif = Uf_temp
            rs.mset({f'Uif_in_CAV_{self.cav_id}':pickle.dumps(self.Uif)})
            self.Yip = Yp_temp
            rs.mset({f'Yip_in_CAV_{self.cav_id}':pickle.dumps(self.Yip)})
            self.Yif = Yf_temp
            rs.mset({f'Yif_in_CAV_{self.cav_id}':pickle.dumps(self.Yif)})
            self.Eip = Ep_temp
            rs.mset({f'Eip_in_CAV_{self.cav_id}':pickle.dumps(self.Eip)})
            self.Eif = Ef_temp
            rs.mset({f'Eif_in_CAV_{self.cav_id}':pickle.dumps(self.Eif)})
            
            rs.mset({f'g_initial_in_CAV_{self.cav_id}':pickle.dumps(g_initial)})
            rs.mset({f'mu_initial_in_CAV_{self.cav_id}':pickle.dumps(mu_initial)})

            # KKT_vert
                # define physicis-augment elements here
                # matrix for physics-augmented calculation
            # Bf = np.hstack((np.eye(N-1),np.zeros([N-1,1])))  # Bf is a N-1*N matrix, Bf*v=v[0:-1]
            # Af = np.hstack((np.zeros([N-1,1]),np.eye(N-1)))  # Af is a N-1*N matrix, Af*v=v[1:]
            # DM = Af - Bf                                     # Diff_matrix DM*v = v[1:] - v[0:-1]
            # K = np.kron(np.eye(N),np.append(np.zeros(int(self.n_vehicle_sub-1)),[1,0]))
            # P = np.kron(np.eye(N),np.append(np.zeros(int(self.n_vehicle_sub)),[1]))
            # F = np.kron(np.eye(N),np.append([1],np.zeros(int(self.n_vehicle_sub)))) 
            # Q_delta1 = DM@P@self.Yif - Tstep*Bf@(self.Eif-F@self.Yif)
            # Q_delta2 = DM@F@self.Yif - Tstep*Bf@self.Uif

            # weight_delta1   = pickle.loads(rs.mget(f'weight_delta1')[0])
            # weight_delta2   = pickle.loads(rs.mget(f'weight_delta2')[0])   # weight coefficient for physics laws

            # M1 = weight_delta1 * np.eye(N-1)
            # M2 = weight_delta2 * np.eye(N-1)

            # cost coefficient
            weight_v        = 1     # weight coefficient for velocity error
            weight_s        = 0.5   # weight coefficient for spacing error   
            weight_u        = 0.1   # weight coefficient for control input

            Qi = np.diagflat(np.append(weight_v*np.ones(int(self.n_vehicle_sub)),[weight_s]))
            Qi_stack = np.kron(np.eye(N),Qi)
            Ri = weight_u
            Ri_stack = np.kron(np.eye(N),Ri)
                
            if self.cav_id == 0: # the first subsystem has different Hgi
                Hg = self.Yif.T@Qi_stack@self.Yif+self.Uif.T@Ri_stack@self.Uif+\
                    self.lambda_gi*np.eye(kappa)+self.lambda_yi*self.Yip.T@self.Yip+\
                    self.rho/2*(np.eye(kappa)+self.Yif.T@P.T@P@self.Yif+self.Uif.T@self.Uif)\
                    +Q_delta2.T@M2@Q_delta2
                print(Hg.max(),flush=True)
                Aeqg = np.vstack((self.Uip,self.Eip,self.Eif))
            else:
                Hg = self.Yif.T@Qi_stack@self.Yif+self.Uif.T@Ri_stack@self.Uif+\
                    self.lambda_gi*np.eye(kappa)+self.lambda_yi*self.Yip.T@self.Yip+\
                    self.rho/2*(np.eye(kappa)+self.Eif.T@self.Eif+self.Yif.T@P.T@P@self.Yif+self.Uif.T@self.Uif)+\
                    Q_delta1.T@M1@Q_delta1 + Q_delta2.T@M2@Q_delta2
                Aeqg = np.vstack((self.Uip,self.Eip))

            Phi = np.vstack((np.hstack((Hg,Aeqg.T)),np.hstack((Aeqg,np.zeros([Aeqg.shape[0],Aeqg.shape[0]])))))
            self.KKT_vert = np.linalg.inv(Phi)
            rs.mset({f'KKT_vert_in_CAV_{self.cav_id}':pickle.dumps(self.KKT_vert)})
            
            # Hz_vert
            if self.cav_id != n_cav-1:
                Hz = self.rho/2*np.eye(kappa)+self.rho/2*self.Yif.T@K.T@K@self.Yif
            else:
                Hz = self.rho/2*np.eye(kappa)
            
            self.Hz_vert = np.linalg.inv(Hz)
            rs.mset({f'Hz_vert_in_CAV_{self.cav_id}':pickle.dumps(self.Hz_vert)})

        u_opt,g_opt,mu_opt,eta_opt,phi_opt,theta_opt,real_iter_num = dDeeP_LCC(curr_step,n_cav,self.cav_id,self.Uip,self.Yip,self.Uif,\
            self.Yif,self.Eip,self.Eif,self.uini,self.yini,self.eini,g_initial,mu_initial,eta_initial,phi_initial,theta_initial,\
            self.lambda_yi,self.u_limit,self.s_limit,self.rho,self.KKT_vert,self.Hz_vert,rs)

        # save initial value of dual variables for next timestep
        rs.mset({f'g_initial_in_CAV_{self.cav_id}':pickle.dumps(g_opt)})
        rs.mset({f'mu_initial_in_CAV_{self.cav_id}':pickle.dumps(mu_opt)})
        rs.mset({f'eta_initial_in_CAV_{self.cav_id}':pickle.dumps(eta_opt)})
        rs.mset({f'phi_initial_in_CAV_{self.cav_id}':pickle.dumps(phi_opt)})
        rs.mset({f'theta_initial_in_CAV_{self.cav_id}':pickle.dumps(theta_opt)})

        use_time = time.time() - start_time
        y_prediction = self.Yif @ g_opt

        Qi = pickle.loads(rs.mget(f'Qi_in_CAV_{self.cav_id}')[0])
        Ri = pickle.loads(rs.mget(f'Ri_in_CAV_{self.cav_id}')[0])
        cost = y_prediction.T@Qi@y_prediction + u_opt.T@Ri@u_opt

        # give messages to clients
        msg_send = [u_opt[0],real_iter_num,use_time,y_prediction[0:p],cost]
        msg_bytes_send = pickle.dumps(msg_send)
        try:
            await websocket.send(msg_bytes_send)
        except websockets.ConnectionClosedOK:
            print('Connection closed by the client',flush = True)\