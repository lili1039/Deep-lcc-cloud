import asyncio
import numpy as np
import pickle
import time
import os
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
    
    rs.mset({f'{cav_id}_check_ready':pickle.dumps([0,timestep,0])})

    # stoping criterion
    iteration_num = 30
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
            # 对于i=2~n的CAV，同时发送Eif给CAVi-1，便于后面计算error_dual2和tolerance_dual2
        
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
            tolerance_pri1 = math.sqrt(g_plus.shape[0])*error_absolute + \
                error_relative*max(np.linalg.norm(g_plus),np.linalg.norm(z_plus))
            error_dual1 = rho*np.linalg.norm(z_plus-z)
            tolerance_dual1 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(mu_plus)

            error_pri2 = np.linalg.norm(epsilon_bar_latter - K@Yif@z_plus)
            tolerance_pri2 = math.sqrt((epsilon_bar_latter).shape[0])*error_absolute + \
                error_relative*max(np.linalg.norm(epsilon_bar_latter),np.linalg.norm(K@Yif@z_plus))
            Eif_latter = pickle.loads(rs.mget(f'Eif_in_CAV_{cav_id+1}')[0])
            error_dual2 = rho*np.linalg.norm(Eif_latter.T@K@Yif@(z_plus-z))    
            tolerance_dual2 = math.sqrt((Eif_latter.T@K@Yif@(z_plus-z)).shape[0])*error_absolute + error_relative*np.linalg.norm(Eif_latter.T@eta_plus)  
            error_pri3 = np.linalg.norm(s_plus - P@Yif@g_plus)
            tolerance_pri3 = math.sqrt(s_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(s_plus),np.linalg.norm(P@Yif@g_plus))
            error_dual3 = rho*np.linalg.norm(Yif.T@P.T@(s_plus-s))
            tolerance_dual3 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Yif.T@P.T@phi_plus)
            error_pri4 =  np.linalg.norm(u_plus-Uif@g_plus)
            tolerance_pri4 = math.sqrt(u_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(Uif@g_plus),np.linalg.norm(u_plus))
            error_dual4 = rho*np.linalg.norm(Uif.T@(u_plus-u))
            tolerance_dual4 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Uif.T@theta_plus)
            
            rs.mset({f'error_tolerance_{cav_id}_{timestep}_{k}':pickle.dumps([[error_pri1,tolerance_pri1,error_dual1,tolerance_dual1],[error_pri2,tolerance_pri2,error_dual2,tolerance_dual2],[error_pri3,tolerance_pri3,error_dual3,tolerance_dual3],[error_pri4,tolerance_pri4,error_dual4,tolerance_dual4]])})  

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
            tolerance_pri1 = math.sqrt(g_plus.shape[0])*error_absolute + \
                error_relative*max(np.linalg.norm(g_plus),np.linalg.norm(z_plus))
            error_dual1 = rho*np.linalg.norm(z_plus-z)            
            tolerance_dual1 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(mu_plus)
            error_pri3 = np.linalg.norm(s_plus - P@Yif@g_plus)
            tolerance_pri3 = math.sqrt(s_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(s_plus),np.linalg.norm(P@Yif@g_plus))
            error_dual3 = rho*np.linalg.norm(Yif.T@P.T@(s_plus-s))
            tolerance_dual3 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Yif.T@P.T@phi_plus)
            error_pri4 =  np.linalg.norm(u_plus-Uif@g_plus)
            tolerance_pri4 = math.sqrt(u_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(Uif@g_plus),np.linalg.norm(u_plus))
            error_dual4 = rho*np.linalg.norm(Uif.T@(u_plus-u))
            tolerance_dual4 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Uif.T@theta_plus)

            rs.mset({f'error_tolerance_{cav_id}_{timestep}_{k}':pickle.dumps([[error_pri1,tolerance_pri1,error_dual1,tolerance_dual1],[],[error_pri3,tolerance_pri3,error_dual3,tolerance_dual3],[error_pri4,tolerance_pri4,error_dual4,tolerance_dual4]])})  


        # Step3: error计算完毕，可以由main-server检查stop criteria是否满足
        # error和tolerance计算完成后，将check_ready标为1，表示该容器已经上传好信息，等待停止迭代信号
        g = g_plus
        z = z_plus
        u = u_plus
        s = s_plus
        mu = mu_plus
        phi = phi_plus
        theta = theta_plus

        # 检查check_stop信号是否产生
        Check_flag_bytes = rs.mget(f'Check_Stop_{timestep}_{k}')[0]
        # 已经产生
        if Check_flag_bytes != None:
            Check_flag = pickle.loads(Check_flag_bytes)
            if Check_flag == 0: # 继续迭代
                rs.mset({f'{cav_id}_check_ready':pickle.dumps([0,timestep,k])})
                continue
            elif Check_flag == 1: # 停止迭代
                rs.mset({f'{cav_id}_check_ready':pickle.dumps([0,timestep,k])})
                break
        
        # check_stop信号未产生，标记check_ready
        rs.mset({f'{cav_id}_check_ready':pickle.dumps([1,timestep,k])})

        start_time = time.time()
        while True:
            duration = time.time()-start_time
            if duration > 10:
                print("trap",flush=True)
                input("Press Enter to continue...")

            Check_flag_bytes = rs.mget(f'Check_Stop_{timestep}_{k}')[0]
            if Check_flag_bytes == None:
                continue
            else:
                Check_flag = pickle.loads(Check_flag_bytes)
                break
        
        rs.mset({f'{cav_id}_check_ready':pickle.dumps([0,timestep,k])})
        if Check_flag == 1:
            break
        elif Check_flag == 0:
            continue
    
    # Record optimal value
    real_iter_num = k+1
    g_opt = g
    mu_opt = mu
    eta_opt = eta
    phi_opt = phi
    theta_opt = theta
    u_opt = u
    
    return u_opt,g_opt,mu_opt,eta_opt,phi_opt,theta_opt,real_iter_num

def Hankel_add_new_col(Tini,N,Su,Sy,Se,Uip,Uif,Yip,Yif,Eip,Eif,k,p,g_initial,mu_initial):
    print('Change Hankel matrix by adding a new column.', flush = True)

    new_up = np.zeros(Tini)
    for j in range(Tini):
        new_up[j] = Su[0,k-N+j]
    Up_temp = np.hstack((Uip,new_up.reshape([int(Tini),1])))

    new_uf = np.zeros(N)
    for j in range(N):
        new_uf[j] = Su[0,k-N+Tini+j]
    Uf_temp = np.hstack((Uif,new_uf.reshape([int(N),1])))

    new_yp = np.zeros(p*Tini)
    for j in range(Tini):
        new_yp[j*p:(j+1)*p] = Sy[:,k-N+j]
    Yp_temp = np.hstack((Yip,new_yp.reshape([int(p*Tini),1])))

    new_yf = np.zeros(p*N)
    for j in range(N):
        new_yf[j*p:(j+1)*p] = Sy[:,k-N+Tini+j]
    Yf_temp = np.hstack((Yif,new_yf.reshape([int(p*N),1])))

    new_ep = np.zeros(Tini)
    for j in range(Tini):
        new_ep[j] = Se[0,k-N+j]
    Ep_temp = np.hstack((Eip,new_ep.reshape([int(Tini),1])))

    new_ef = np.zeros(N)
    for j in range(N):
        new_ef[j] = Se[0,k-N+Tini+j]
    Ef_temp = np.hstack((Eif,new_ef.reshape([int(N),1])))

    # g_initial / mu_initial
    g_initial = np.hstack((g_initial,0))
    mu_initial = np.hstack((mu_initial,0))
    
    return [Up_temp,Uf_temp,Yp_temp,Yf_temp,Ep_temp,Ef_temp,g_initial,mu_initial]

def Hankel_substitute_col(Hankel_col,Tini,N,Su,Sy,Se,Uip,Uif,Yip,Yif,Eip,Eif,k,p,g_initial,mu_initial):
    print('Change Hankel matrix by delete one column and add a new one.', flush = True)

    base_dataset_col = Hankel_col
    new_up = np.zeros(Tini)
    for j in range(Tini):
        new_up[j] = Su[0,k-N+j]
    # Up_temp = np.hstack((self.Uip[:,1:],new_up.reshape([int(Tini),1])))
    Up_temp = np.hstack((Uip[:,0:base_dataset_col],Uip[:,base_dataset_col+1:],new_up.reshape([int(Tini),1])))

    new_uf = np.zeros(N)
    for j in range(N):
        new_uf[j] = Su[0,k-N+Tini+j]
    # Uf_temp = np.hstack((self.Uif[:,1:],new_uf.reshape([int(N),1])))
    Uf_temp = np.hstack((Uif[:,0:base_dataset_col],Uif[:,base_dataset_col+1:],new_uf.reshape([int(N),1])))

    new_yp = np.zeros(p*Tini)
    for j in range(Tini):
        new_yp[j*p:(j+1)*p] = Sy[:,k-N+j]
    # Yp_temp = np.hstack((self.Yip[:,1:],new_yp.reshape([int(p*Tini),1])))
    Yp_temp = np.hstack((Yip[:,0:base_dataset_col],Yip[:,base_dataset_col+1:],new_yp.reshape([int(p*Tini),1])))

    new_yf = np.zeros(p*N)
    for j in range(N):
        new_yf[j*p:(j+1)*p] = Sy[:,k-N+Tini+j]
    # Yf_temp = np.hstack((self.Yif[:,1:],new_yf.reshape([int(p*N),1])))
    Yf_temp = np.hstack((Yif[:,0:base_dataset_col],Yif[:,base_dataset_col+1:],new_yf.reshape([int(p*N),1])))

    new_ep = np.zeros(Tini)
    for j in range(Tini):
        new_ep[j] = Se[0,k-N+j]
    # Ep_temp = np.hstack((self.Eip[:,1:],new_ep.reshape([int(Tini),1])))
    Ep_temp = np.hstack((Eip[:,0:base_dataset_col],Eip[:,base_dataset_col+1:],new_ep.reshape([int(Tini),1])))

    new_ef = np.zeros(N)
    for j in range(N):
        new_ef[j] = Se[0,k-N+Tini+j]
    # Ef_temp = np.hstack((self.Eif[:,1:],new_ef.reshape([int(N),1])))
    Ef_temp = np.hstack((Eif[:,0:base_dataset_col],Eif[:,base_dataset_col+1:],new_ef.reshape([int(N),1])))

    # g_initial / mu_initial
    # g_initial = np.hstack((g_initial[1:],0))
    g_initial = np.hstack((g_initial[0:base_dataset_col],g_initial[base_dataset_col+1:],0))
    # mu_initial = np.hstack((mu_initial[1:],0))
    mu_initial = np.hstack((mu_initial[0:base_dataset_col],mu_initial[base_dataset_col+1:],0))

    return [Up_temp,Uf_temp,Yp_temp,Yf_temp,Ep_temp,Ef_temp,g_initial,mu_initial]


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
        rs.mset({f'Curr_step':pickle.dumps(curr_step)})

        # 从数据库获取初始参数和数据
        keys = [
            f'Uip_in_CAV_{self.cav_id}',
            f'Uif_in_CAV_{self.cav_id}',
            f'Eip_in_CAV_{self.cav_id}',
            f'Eif_in_CAV_{self.cav_id}',
            f'Yip_in_CAV_{self.cav_id}',
            f'Yif_in_CAV_{self.cav_id}',
            f'rho_in_CAV_{self.cav_id}',
            f'lambda_yi_in_CAV_{self.cav_id}',
            f'KKT_vert_in_CAV_{self.cav_id}',
            f'Hz_vert_in_CAV_{self.cav_id}',
            f'g_initial_in_CAV_{self.cav_id}',
            f'mu_initial_in_CAV_{self.cav_id}',
            f'eta_initial_in_CAV_{self.cav_id}',
            f'phi_initial_in_CAV_{self.cav_id}',
            f'theta_initial_in_CAV_{self.cav_id}',
            f'Su_in_CAV_{self.cav_id}',
            f'Sy_in_CAV_{self.cav_id}',
            f'Se_in_CAV_{self.cav_id}',
            'n_cav',
        ]

        # 一次性获取所有键值
        values = rs.mget(keys)

        # 使用循环简化赋值操作
        (
            self.Uip, self.Uif, self.Eip, self.Eif, self.Yip, self.Yif,
            self.rho, self.lambda_yi, self.KKT_vert, 
            self.Hz_vert, g_initial, mu_initial, eta_initial, phi_initial, theta_initial, 
            Su, Sy, Se, n_cav
        ) = [pickle.loads(value) for value in values]

        # problem size
        m = self.uini.ndim         # the size of control input of each subsystem
        p = self.yini.shape[0]     # the size of output of each subsystem
        # time horizon
        Tini = int(self.Uip.shape[0]/m)
        N = int(self.Uif.shape[0]/m)
        Hankel_col = 231
        max_col = 400

        if Hankel_update_flag and curr_step >= N:
            kappa = self.Uip.shape[1]
            print('kappa = ',kappa)
            if  kappa < max_col:
                kappa = kappa + 1
                result = Hankel_add_new_col(Tini,N,Su,Sy,Se,self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,curr_step,p,g_initial,mu_initial)
            else:
                # 如果列数已经到达最大容许列数，则减掉前一列，再加上新一列
                result = Hankel_substitute_col(Hankel_col,Tini,N,Su,Sy,Se,self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,curr_step,p,g_initial,mu_initial)

            [self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,g_initial,mu_initial] = result

        # 调用deepc
        u_opt,g_opt,mu_opt,eta_opt,phi_opt,theta_opt,real_iter_num = dDeeP_LCC(curr_step,n_cav,self.cav_id,self.Uip,self.Yip,self.Uif,\
            self.Yif,self.Eip,self.Eif,self.uini,self.yini,self.eini,g_initial,mu_initial,eta_initial,phi_initial,theta_initial,\
            self.lambda_yi,self.u_limit,self.s_limit,self.rho,self.KKT_vert,self.Hz_vert,rs)

        # save initial value of dual variables for next timestep
        rs.mset({
            f'Uip_in_CAV_{self.cav_id}': pickle.dumps(self.Uip),
            f'Uif_in_CAV_{self.cav_id}': pickle.dumps(self.Uif),
            f'Eip_in_CAV_{self.cav_id}': pickle.dumps(self.Eip),
            f'Eif_in_CAV_{self.cav_id}': pickle.dumps(self.Eif),
            f'Yip_in_CAV_{self.cav_id}': pickle.dumps(self.Yip),
            f'Yif_in_CAV_{self.cav_id}': pickle.dumps(self.Yif),
            f'g_initial_in_CAV_{self.cav_id}': pickle.dumps(g_opt),
            f'mu_initial_in_CAV_{self.cav_id}': pickle.dumps(mu_opt),
            f'eta_initial_in_CAV_{self.cav_id}': pickle.dumps(eta_opt),
            f'phi_initial_in_CAV_{self.cav_id}': pickle.dumps(phi_opt),
            f'theta_initial_in_CAV_{self.cav_id}': pickle.dumps(theta_opt)
        })

        use_time = time.time() - start_time
        print(f"total computation time: {use_time}",flush=True)
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
            print('Connection closed by the client',flush = True)