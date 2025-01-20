import numpy as np
import pickle
import time
import redis
import math
import websockets
from scipy.linalg import block_diag 

# update method
Hankel_update_method = 1
# 1: with base dataset 2: without base dataset
Inv_method = 2
# 1: Inv by Recursion 2: Inv by numpy

# check stopping criteria
Is_Check = False

iteration_num = 20

def dDeeP_LCC(Tini,N,kappa,timestep,n_cav,cav_id,Uip,Yip,Uif,Yif,Eip,Eif,ui_ini,yi_ini,ei_ini,g_initial,mu_initial,eta_initial,phi_initial,theta_initial,delta_initial,lambda_yi,u_limit,s_limit,rho,Hgi_vert,Hz_vert,rs):

    # stoping criterion
    error_absolute = 0.1
    error_relative = 1e-3 # 1e-3

    # problem size
    m = ui_ini.ndim         # the size of control input of each subsystem
    p = yi_ini.shape[0]     # the size of output of each subsystem

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
    delta = delta_initial
    s=P@Yif@g       #一维数组
    u=Uif@g         #一维数组

    for k in range(iteration_num):
        # time_step12_start = time.time()
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
            Agi = np.vstack((Uip,Eip,Eif))
            qg = -lambda_yi*Yip.T@yi_ini + mu/2 - rho*z/2 - \
                Yif.T@P.T@phi/2 - Uif.T@theta/2 - rho*Yif.T@P.T@s/2 - rho*Uif.T@u/2 + 1/2*Agi.T@delta - rho/2*Agi.T@beqg
        else:
            Agi = np.vstack((Uip,Eip))
            qg = -lambda_yi*Yip.T@yi_ini + mu/2 - rho*z/2 + Eif.T@eta_bar_former/2 - \
                Yif.T@P.T@phi/2 - Uif.T@theta/2 - rho*Yif.T@P.T@s/2 - rho*Uif.T@u/2 + 1/2*Agi.T@delta - rho/2*Agi.T@beqg
            
        # 计算g+
        g_plus = -Hgi_vert@qg

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
            delta_plus = delta + rho*(Agi@g_plus-beqg)

            if Is_Check:
                # 计算迭代停止条件并存入数据库
                temp_pri0 = Agi@g_plus
                error_pri0 = np.linalg.norm(temp_pri0 - beqg)
                tolerance_pri0 = math.sqrt(beqg.shape[0])*error_absolute + \
                    error_relative*max(np.linalg.norm(temp_pri0),np.linalg.norm(beqg))
                
                error_pri1 = np.linalg.norm(g_plus - z_plus)
                tolerance_pri1 = math.sqrt(kappa)*error_absolute + \
                    error_relative*max(np.linalg.norm(g_plus),np.linalg.norm(z_plus))

                error_dual1 = rho*np.linalg.norm(z_plus-z)
                tolerance_dual1 = math.sqrt(kappa)*error_absolute + error_relative*np.linalg.norm(mu_plus)

                error_pri2 = np.linalg.norm(epsilon_bar_latter - K@Yif@z_plus)
                tolerance_pri2 = math.sqrt(N)*error_absolute + \
                    error_relative*max(np.linalg.norm(epsilon_bar_latter),np.linalg.norm(K@Yif@z_plus))
                
                Eif_latter = pickle.loads(rs.mget(f'Eif_in_CAV_{cav_id+1}')[0])
                error_dual2 = rho*np.linalg.norm(Eif_latter.T@K@Yif@(z_plus-z))    
                tolerance_dual2 = math.sqrt(N)*error_absolute + error_relative*np.linalg.norm(Eif_latter.T@eta_plus)  
                
                error_pri3 = np.linalg.norm(s_plus - P@Yif@g_plus)
                tolerance_pri3 = math.sqrt(s_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(s_plus),np.linalg.norm(P@Yif@g_plus))
                
                error_dual3 = rho*np.linalg.norm(Yif.T@P.T@(s_plus-s))
                tolerance_dual3 = math.sqrt(kappa)*error_absolute + error_relative*np.linalg.norm(Yif.T@P.T@phi_plus)
                
                error_pri4 =  np.linalg.norm(u_plus-Uif@g_plus)
                tolerance_pri4 = math.sqrt(N)*error_absolute + error_relative*max(np.linalg.norm(Uif@g_plus),np.linalg.norm(u_plus))
                
                error_dual4 = rho*np.linalg.norm(Uif.T@(u_plus-u))
                tolerance_dual4 = math.sqrt(kappa)*error_absolute + error_relative*np.linalg.norm(Uif.T@theta_plus)

                if error_pri0 <= tolerance_pri0 and error_pri1 <= tolerance_pri1 and error_dual1 <= tolerance_dual1 and error_pri2 <= tolerance_pri2 and error_dual2 <= tolerance_dual2 and error_pri3 <= tolerance_pri3 and error_dual3 <= tolerance_dual3 and error_pri4 <= tolerance_pri4 and error_dual4 <= tolerance_dual4:
                    rs.mset({f'rollout_flag_{cav_id}_{timestep}_{k}':pickle.dumps(1)})
                else:
                    rs.mset({f'rollout_flag_{cav_id}_{timestep}_{k}':pickle.dumps(0)})

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
            delta_plus = delta + rho*(Agi@g_plus-beqg)

            if Is_Check:
                temp_pri0 =  Agi@g_plus
                error_pri0 = np.linalg.norm(temp_pri0 - beqg)
                tolerance_pri0 = math.sqrt(beqg.shape[0])*error_absolute + \
                    error_relative*max(np.linalg.norm(temp_pri0),np.linalg.norm(beqg))
                
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

                if error_pri0 <= tolerance_pri0 and error_pri1 <= tolerance_pri1 and error_dual1 <= tolerance_dual1 and error_pri3 <= tolerance_pri3 and error_dual3 <= tolerance_dual3 and error_pri4 <= tolerance_pri4 and error_dual4 <= tolerance_dual4:
                    rs.mset({f'rollout_flag_{cav_id}_{timestep}_{k}':pickle.dumps(1)})
                else:
                    rs.mset({f'rollout_flag_{cav_id}_{timestep}_{k}':pickle.dumps(0)})

        # Step3: error计算完毕，可以由main-server检查stop criteria是否满足
        # error和tolerance计算完成后，将check_ready标为1，表示该容器已经上传好信息，等待停止迭代信号
        g = g_plus
        z = z_plus
        u = u_plus
        s = s_plus
        mu = mu_plus
        phi = phi_plus
        theta = theta_plus
        delta = delta_plus

        if k == iteration_num-1:
            break
        else:
            if Is_Check:
                while True:
                    check_flag_bytes=rs.mget(f'rollout_flag_total_{timestep}_{k}')[0]
                    if check_flag_bytes==None:
                            continue
                    else:
                        check_flag = pickle.loads(check_flag_bytes)
                        break

                if check_flag == 0:
                    continue
                elif check_flag==1:
                    break
            else:
                continue

    # Record optimal value
    real_iter_num = k+1
    g_opt = g
    mu_opt = mu
    eta_opt = eta
    phi_opt = phi
    theta_opt = theta
    u_opt = u
    delta_opt = delta
    
    return u_opt,g_opt,mu_opt,eta_opt,phi_opt,theta_opt,delta_opt,real_iter_num

# delete the first column and add a new column at the last
def Hankel_substitute_col_1(Tini,N,Su,Sy,Se,Uip,Uif,Yip,Yif,Eip,Eif,k,p,g_initial,mu_initial):
    new_up = np.zeros(Tini)
    for j in range(Tini):
        new_up[j] = Su[0,k+j]
    Up_temp = np.hstack((Uip[:,1:],new_up.reshape([int(Tini),1])))

    new_uf = np.zeros(N)
    for j in range(N):
        new_uf[j] = Su[0,k+Tini+j]
    Uf_temp = np.hstack((Uif[:,1:],new_uf.reshape([int(N),1])))

    new_yp = np.zeros(p*Tini)
    for j in range(Tini):
        new_yp[j*p:(j+1)*p] = Sy[:,k+j]
    Yp_temp = np.hstack((Yip[:,1:],new_yp.reshape([int(p*Tini),1])))

    new_yf = np.zeros(p*N)
    for j in range(N):
        new_yf[j*p:(j+1)*p] = Sy[:,k+Tini+j]
    Yf_temp = np.hstack((Yif[:,1:],new_yf.reshape([int(p*N),1])))

    new_ep = np.zeros(Tini)
    for j in range(Tini):
        new_ep[j] = Se[0,k+j]
    Ep_temp = np.hstack((Eip[:,1:],new_ep.reshape([int(Tini),1])))

    new_ef = np.zeros(N)
    for j in range(N):
        new_ef[j] = Se[0,k+Tini+j]
    Ef_temp = np.hstack((Eif[:,1:],new_ef.reshape([int(N),1])))

    g_initial = np.hstack((g_initial[1:],0))
    mu_initial = np.hstack((mu_initial[1:],0))

    c = np.hstack((new_up,new_ep,new_yp,new_uf,new_ef,new_yf))

    return [Up_temp,Uf_temp,Yp_temp,Yf_temp,Ep_temp,Ef_temp,g_initial,mu_initial,c]

# delete the first column of Online part and add a new column at the last
def Hankel_substitute_col_2(Tini,N,k_on,Su,Sy,Se,Uip,Uif,Yip,Yif,Eip,Eif,k,p,g_initial,mu_initial):
    # k_on is the index of the beginning of Online part, and also the minimal number of column that ensure offline data satisfying P.E. condition.

    new_up = np.zeros(Tini)
    for j in range(Tini):
        new_up[j] = Su[0,k+j]
    Up_temp = np.hstack((Uip[:,0:k_on],Uip[:,k_on+1:],new_up.reshape([int(Tini),1])))

    new_uf = np.zeros(N)
    for j in range(N):
        new_uf[j] = Su[0,k+Tini+j]
    Uf_temp = np.hstack((Uif[:,0:k_on],Uif[:,k_on+1:],new_uf.reshape([int(N),1])))

    new_yp = np.zeros(p*Tini)
    for j in range(Tini):
        new_yp[j*p:(j+1)*p] = Sy[:,k+j]
    Yp_temp = np.hstack((Yip[:,0:k_on],Yip[:,k_on+1:],new_yp.reshape([int(p*Tini),1])))

    new_yf = np.zeros(p*N)
    for j in range(N):
        new_yf[j*p:(j+1)*p] = Sy[:,k+Tini+j]
    Yf_temp = np.hstack((Yif[:,0:k_on],Yif[:,k_on+1:],new_yf.reshape([int(p*N),1])))

    new_ep = np.zeros(Tini)
    for j in range(Tini):
        new_ep[j] = Se[0,k+j]
    Ep_temp = np.hstack((Eip[:,0:k_on],Eip[:,k_on+1:],new_ep.reshape([int(Tini),1])))

    new_ef = np.zeros(N)
    for j in range(N):
        new_ef[j] = Se[0,k+Tini+j]
    Ef_temp = np.hstack((Eif[:,0:k_on],Eif[:,k_on+1:],new_ef.reshape([int(N),1])))

    g_initial = np.hstack((g_initial[0:k_on],g_initial[k_on+1:],0))
    mu_initial = np.hstack((mu_initial[0:k_on],mu_initial[k_on+1:],0))

    c = np.hstack((new_up,new_ep,new_yp,new_uf,new_ef,new_yf))

    return [Up_temp,Uf_temp,Yp_temp,Yf_temp,Ep_temp,Ef_temp,g_initial,mu_initial,c]

# Calculate inv(Hgi+) based on known inv(Hgi)
def Matrix_inv_add1newcol(c,Mi,Hi,rho,lambda_gi,Hgi_vert_last,kappa):
    # 尾加一列 
    c = c.reshape(-1,1)
    m_vec = c.reshape(1,-1)@Mi@Hi
    m = c.reshape(1,-1)@Mi@c.reshape(-1,1)+(lambda_gi+rho/2)
    ms = m-m_vec.reshape(1,-1)@Hgi_vert_last@m_vec.reshape(-1,1)

    # Hgi_plus = np.hstack((Hi,c.reshape(-1,1))).T@Mi@np.hstack((Hi,c.reshape(-1,1))) + (rho/2+lambda_gi)*np.eye(kappa+1)
    # Hgi_last = Hi.T@Mi@Hi + (rho/2+lambda_gi)*np.eye(kappa) 
    # Hgi_last_vert_2 = np.linalg.inv(Hgi_last)
    # print("last Hgi vert Results close?", np.allclose(Hgi_last_vert_2, Hgi_vert_last)) # True
    # Hgi_2 = np.vstack((np.hstack((Hgi_last,m_vec.reshape(-1,1))),np.hstack((m_vec.reshape(1,-1),m))))
    # print("Hgi construct Results close?", np.allclose(Hgi_plus, Hgi_2)) # True

    # print("Condition number of A11:", np.linalg.cond(Hgi_last)) # 条件数很大 病态严重

    Hgi_vert = (1/ms)*np.vstack((np.hstack((ms*Hgi_vert_last+Hgi_vert_last@m_vec.reshape(-1,1)@m_vec.reshape(1,-1)@Hgi_vert_last,-Hgi_vert_last@m_vec.reshape(-1,1))),np.hstack((-m_vec.reshape(1,-1)@Hgi_vert_last,np.eye(1)))))
    # Hgi_vert_true = np.linalg.inv(Hgi_plus)
    # print("Hgi vert Results close?", np.allclose(Hgi_vert_true[-1,-1], Hgi_vert[-1,-1])) # False
    # print('Recursion last element:',Hgi_vert[-1,-1])
    # print('True last element:',Hgi_vert_true[-1,-1])
                                  
    return Hgi_vert

# Calculate inv(Hgi-) based on known inv(Hgi)
def Matrix_inv_delete1stcol(Hgi_vert_last):
    a = Hgi_vert_last[0,0]
    cT = Hgi_vert_last[1:,0]
    b = Hgi_vert_last[0,1:]
    D = Hgi_vert_last[1:,1:]
    
    Hgi_vert = D-((1/a)*cT.reshape(-1,1)@b.reshape(1,-1))

    return Hgi_vert

def Matrix_inv_phase1(c,Mi,Hi,rho,lambda_gi,Hgi_vert_last,kappa):
    Hgi_vert_plus = Matrix_inv_add1newcol(c,Mi,Hi,rho,lambda_gi,Hgi_vert_last,kappa)
    # Hgi = np.hstack((Hi,c.reshape(-1,1))).T@Mi@np.hstack((Hi,c.reshape(-1,1))) + (rho/2+lambda_gi)*np.eye(kappa+1)
    # Hgi_vert_1 = np.linalg.inv(Hgi)
    # print("PLUS Results close?", np.allclose(Hgi_vert_1, Hgi_vert_plus))

    Hgi_vert = Matrix_inv_delete1stcol(Hgi_vert_plus)

    return Hgi_vert

def create_Pc(kon, kappa):
    # Identity matrices
    I_kon = np.eye(kon)  # kon x kon identity matrix
    I_1 = np.eye(1)      # 1 x 1 identity matrix
    I_kappa_kon = np.eye(kappa - kon)  # (kappa-kon) x (kappa-kon) identity matrix

    # Zero matrices
    Z_11 = np.zeros((kon, 1))
    Z_13 = np.zeros((kon, kappa - kon))
    Z_22 = np.zeros((1, kon))
    Z_23 = np.zeros((1, kappa - kon))
    Z_31 = np.zeros((kappa - kon, 1))
    Z_32 = np.zeros((kappa - kon, kon))

    # Assemble the block matrix
    matrix = np.block([
        [Z_11, I_kon, Z_13],
        [I_1, Z_22, Z_23],
        [Z_31, Z_32, I_kappa_kon]
    ])

    return matrix

def create_Pr(kon, kappa):
    # Identity matrices
    I_kon = np.eye(kon)  # kon x kon identity matrix
    I_1 = np.eye(1)      # 1 x 1 identity matrix
    I_kappa_kon = np.eye(kappa - kon)  # (kappa-kon-1) x (kappa-kon-1) identity matrix

    # Zero matrices
    Z_11 = np.zeros((1 , kon))
    Z_13 = np.zeros((1, kappa - kon))
    Z_22 = np.zeros((kon , 1))
    Z_23 = np.zeros((kon , kappa - kon))
    Z_31 = np.zeros((kappa - kon, kon))
    Z_32 = np.zeros((kappa - kon, 1))

    # Assemble the block matrix
    matrix = np.block([
        [Z_11, I_1, Z_13],
        [I_kon, Z_22, Z_23],
        [Z_31, Z_32, I_kappa_kon]
    ])

    return matrix

def Matrix_inv_phase2(c,Mi,Hi,rho,lambda_gi,Hgi_vert_last,kappa,k_on):
    Hgi_vert_plus = Matrix_inv_add1newcol(c,Mi,Hi,rho,lambda_gi,Hgi_vert_last,kappa)
    # Hgi = np.hstack((Hi,c.reshape(-1,1))).T@Mi@np.hstack((Hi,c.reshape(-1,1))) + (rho/2+lambda_gi)*np.eye(kappa+1)
    # Hgi_vert_1 = np.linalg.inv(Hgi)
    # print("PLUS Results close?", np.allclose(Hgi_vert_1, Hgi_vert_plus))
    # Hgi_vert_plus = Hgi_vert_1

    Pc = create_Pc(k_on,kappa)
    Pc_vert = np.linalg.inv(Pc)
    Pr = create_Pr(k_on,kappa)
    Pr_vert = np.linalg.inv(Pr)

    Hgi_vert = Matrix_inv_delete1stcol(Pc_vert @ Hgi_vert_plus @Pr_vert)

    return Hgi_vert

class SubsystemParam:
    def __init__(self):
        # 系统车辆参数s
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
        self.rho             = 0
        self.lambda_yi       = 0

        # 安全限制
        self.acel_max = 0       # max accelaration
        self.dcel_max = 0      # min accelaration
        self.spacing_max = 0
        self.spacing_min = 0
        self.s_star = 0
        self.u_limit = []
        self.s_limit = []
        self.u_limit = []
        self.s_limit = []       

class SubsystemSolver(SubsystemParam):
    def __init__(self):
        super().__init__()

    async def solver(self,websocket, path):
        # 数据库连接
        rs = redis.StrictRedis(host='172.18.0.1',port=6379,db=2,password="chlpw1039") 

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

        start_time = time.time()
        # 从数据库获取初始参数和数据
        keys = [
            f'Uip_in_CAV_{self.cav_id}',
            f'Uif_in_CAV_{self.cav_id}',
            f'Eip_in_CAV_{self.cav_id}',
            f'Eif_in_CAV_{self.cav_id}',
            f'Yip_in_CAV_{self.cav_id}',
            f'Yif_in_CAV_{self.cav_id}',
            f'lambda_yi',
            f'lambda_gi',
            f'rho',
            f'g_initial_in_CAV_{self.cav_id}',
            f'mu_initial_in_CAV_{self.cav_id}',
            f'eta_initial_in_CAV_{self.cav_id}',
            f'phi_initial_in_CAV_{self.cav_id}',
            f'theta_initial_in_CAV_{self.cav_id}',
            f'delta_initial_in_CAV_{self.cav_id}',
            f'Su_in_CAV_{self.cav_id}',
            f'Sy_in_CAV_{self.cav_id}',
            f'Se_in_CAV_{self.cav_id}',
            f'n_cav',
            f'k_on',
            f'Qi_in_CAV_{self.cav_id}',
            f'Ri_in_CAV_{self.cav_id}',
            f'K_in_CAV_{self.cav_id}',
            f'P_in_CAV_{self.cav_id}',
            f'acel_max',
            f'dcel_max',
            f's_max',
            f's_st',
            f's_star',
            f'Tstep',
        ]

        # 一次性获取所有键值
        values = rs.mget(keys)

        # 使用循环简化赋值操作
        (
            self.Uip, self.Uif, self.Eip, self.Eif, self.Yip, self.Yif,
            self.lambda_yi, lambda_gi, self.rho,
            g_initial, mu_initial, eta_initial, phi_initial, theta_initial, delta_initial,
            Su, Sy, Se, 
            n_cav, k_on, Qi_stack, Ri_stack, K, P, 
            self.acel_max, self.dcel_max, self.spacing_max, self.spacing_min, self.s_star, Tstep
        ) = [pickle.loads(value) for value in values]

        self.u_limit = np.array([self.dcel_max,self.acel_max])
        self.s_limit = np.array([self.spacing_min-self.s_star,self.spacing_max-self.s_star])  

        # problem size
        m = self.uini.ndim         # the size of control input of each subsystem
        p = self.yini.shape[0]     # the size of output of each subsystem
        
        # time horizon
        Tini = int(self.Uip.shape[0]/m)
        N = int(self.Uif.shape[0]/m)
        kappa = int(self.Uip.shape[1])

        if Hankel_update_flag:
            # generate new Hankel Matrices
            if Hankel_update_method == 1 :
                off_col = kappa - (curr_step + 1)
                if  off_col > k_on[self.cav_id]:
                    result = Hankel_substitute_col_1(Tini,N,Su,Sy,Se,self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,curr_step,p,g_initial,mu_initial)
                else: 
                    result = Hankel_substitute_col_2(Tini,N,int(k_on[self.cav_id]),Su,Sy,Se,self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,curr_step,p,g_initial,mu_initial)         
            
                # calculate inverse of matrices Hgi and Hz
                if Inv_method == 2 or curr_step == 0: # inverse by numpy
                    [self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,g_initial,mu_initial,c] = result

                    M = block_diag(self.rho/2*np.eye(Tini),self.rho/2*np.eye(Tini),self.lambda_yi*np.eye(p*Tini),Ri_stack+self.rho/2*np.eye(N),self.rho/2*np.eye(N),Qi_stack+self.rho/2*P.T@P)
                    H = np.vstack((self.Uip,self.Eip,self.Yip,self.Uif,self.Eif,self.Yif))

                    # Hgi_vert
                    # Hgi_1 = self.Yif.T@Qi_stack@self.Yif+self.Uif.T@Ri_stack@self.Uif+\
                    #         lambda_gi*np.eye(kappa)+self.lambda_yi*self.Yip.T@self.Yip+\
                    #         self.rho/2*(np.eye(kappa)+self.Yif.T@P.T@P@self.Yif+self.Uif.T@self.Uif)+\
                    #             self.rho/2*(self.Eif.T@self.Eif+self.Uip.T@self.Uip+self.Eip.T@self.Eip)
                    Hgi = H.T@M@H + (self.rho/2+lambda_gi)*np.eye(kappa)
                    # print("Condition number of initial Hgi:", np.linalg.cond(Hgi))

                    # print("Results close?", np.allclose(Hgi_1, Hgi))

                    time_hgi_start = time.time()
                    Hgi_vert = np.linalg.inv(Hgi)

                    # time_hgi_start = time.time()
                    # Hgi_vert_0 = np.linalg.inv(Hgi)
                    A = (lambda_gi+self.rho/2)*np.eye(kappa)
                    A_vert = np.linalg.inv(A)
                    temp = np.linalg.inv(M)+H@A_vert@H.T
                    Mmid_vert = np.linalg.inv(temp)
                    Hgi_vert_0 = A_vert - A_vert@H.T@Mmid_vert@H@A_vert

                    print('Hgi True?',np.allclose(Hgi_vert_0,Hgi_vert))
                    # print(Hgi_vert_0[0,0],Hgi_vert_0[1,2])
                    # print(Hgi_vert[0,0],Hgi_vert[1,2])
                    Hgi_vert = Hgi_vert_0
                    # print(f"Hgi_vert consume{time.time()-time_hgi_start}", flush=True)
                    print('Hgi cond',np.linalg.cond(Hgi))
                    rs.mset({f'Hgi_vert_in_CAV_{self.cav_id}':pickle.dumps(Hgi_vert)})
                    
                elif Inv_method == 1: # inverse by recursion
                    H = np.vstack((self.Uip,self.Eip,self.Yip,self.Uif,self.Eif,self.Yif))
                    c = result[-1]

                    Hgi_vert_last = pickle.loads(rs.mget(f'Hgi_vert_in_CAV_{self.cav_id}')[0])

                    if off_col > k_on[self.cav_id]:
                        M = block_diag(self.rho/2*np.eye(Tini),self.rho/2*np.eye(Tini),self.lambda_yi*np.eye(p*Tini),Ri_stack+self.rho/2*np.eye(N),self.rho/2*np.eye(N),Qi_stack+self.rho/2*P.T@P)
                        Hgi_vert = Matrix_inv_phase1(c,M,H,self.rho,lambda_gi,Hgi_vert_last,kappa)
                        rs.mset({f'Hgi_vert_in_CAV_{self.cav_id}':pickle.dumps(Hgi_vert)})
                    else:
                        M = block_diag(self.rho/2*np.eye(Tini),self.rho/2*np.eye(Tini),self.lambda_yi*np.eye(p*Tini),Ri_stack+self.rho/2*np.eye(N),self.rho/2*np.eye(N),Qi_stack+self.rho/2*P.T@P)
                        Hgi_vert = Matrix_inv_phase2(c,M,H,self.rho,lambda_gi,Hgi_vert_last,kappa,int(k_on[self.cav_id]))
                        rs.mset({f'Hgi_vert_in_CAV_{self.cav_id}':pickle.dumps(Hgi_vert)})

                        # [self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,g_initial,mu_initial,c] = result
                        # H = np.vstack((self.Uip,self.Eip,self.Yip,self.Uif,self.Eif,self.Yif))
                        # Hgi_2 = H.T@M@H + (self.rho/2+lambda_gi)*np.eye(kappa)
                        # Hgi_vert_2 = np.linalg.inv(Hgi_2)
                        # print("PHASE 2 Results close?", np.allclose(Hgi_vert_2, Hgi_vert), flush=True)
            
            elif Hankel_update_method == 2:
                result = Hankel_substitute_col_1(Tini,N,Su,Sy,Se,self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,curr_step,p,g_initial,mu_initial)
                [self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,g_initial,mu_initial,c] = result

                M = block_diag(self.rho/2*np.eye(Tini),self.rho/2*np.eye(Tini),self.lambda_yi*np.eye(p*Tini),Ri_stack+self.rho/2*np.eye(N),self.rho/2*np.eye(N),Qi_stack+self.rho/2*P.T@P)
                H = np.vstack((self.Uip,self.Eip,self.Yip,self.Uif,self.Eif,self.Yif))
                Hgi = H.T@M@H + (self.rho/2+lambda_gi)*np.eye(kappa)

                time_hgi_start = time.time()
                Hgi_vert = np.linalg.inv(Hgi)
                # rs.mset({f'Hgi_vert_in_CAV_{self.cav_id}':pickle.dumps(Hgi_vert)})
            
            # Hz_vert
            [self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,g_initial,mu_initial,c] = result
            if self.cav_id != n_cav-1:
                Hz = self.rho/2*np.eye(int(kappa))+self.rho/2*self.Yif.T@K.T@K@self.Yif
            else:
                Hz = self.rho/2*np.eye(int(kappa))

            # time_hz_start = time.time()
            Hz_vert = np.linalg.pinv(Hz)
            # print(f"Hz_vert consume{time.time()-time_hz_start}", flush=True)

        # 调用deepc
        u_opt,g_opt,mu_opt,eta_opt,phi_opt,theta_opt,delta_opt,real_iter_num = dDeeP_LCC(Tini,N,kappa,curr_step,n_cav,self.cav_id,self.Uip,self.Yip,self.Uif,\
            self.Yif,self.Eip,self.Eif,self.uini,self.yini,self.eini,g_initial,mu_initial,eta_initial,phi_initial,theta_initial,delta_initial,\
            self.lambda_yi,self.u_limit,self.s_limit,self.rho,Hgi_vert,Hz_vert,rs)

        # save initial value of dual variables for next timestep
        rs.mset({
            f'g_initial_in_CAV_{self.cav_id}': pickle.dumps(g_opt),
            f'mu_initial_in_CAV_{self.cav_id}': pickle.dumps(mu_opt),
            f'eta_initial_in_CAV_{self.cav_id}': pickle.dumps(eta_opt),
            f'phi_initial_in_CAV_{self.cav_id}': pickle.dumps(phi_opt),
            f'theta_initial_in_CAV_{self.cav_id}': pickle.dumps(theta_opt),
            f'delta_initial_in_CAV_{self.cav_id}': pickle.dumps(delta_opt),
            f'Uip_in_CAV_{self.cav_id}': pickle.dumps(self.Uip),
            f'Uif_in_CAV_{self.cav_id}': pickle.dumps(self.Uif),
            f'Eip_in_CAV_{self.cav_id}': pickle.dumps(self.Eip),
            f'Eif_in_CAV_{self.cav_id}': pickle.dumps(self.Eif),
            f'Yip_in_CAV_{self.cav_id}': pickle.dumps(self.Yip),
            f'Yif_in_CAV_{self.cav_id}': pickle.dumps(self.Yif)
        }) 

        use_time = time.time() - start_time
        # print(f"total computation time: {use_time}",flush=True)


        # cost
        y_cost = (self.Yif@g_opt).T@Qi_stack@(self.Yif@g_opt)
        u_cost = (u_opt).T@Ri_stack@(u_opt)
        gi_cost = lambda_gi*g_opt.T@g_opt
        sig_yi_cost = self.lambda_yi*(self.Yip@g_opt-(self.yini.T).reshape(-1)).T@(self.Yip@g_opt-(self.yini.T).reshape(-1))
        print(f'y/u/gi/sig_yi cost:{y_cost}/{u_cost}/{gi_cost}/{sig_yi_cost}',flush=True)

        # give messages to clients
        msg_send = [u_opt[0],real_iter_num,use_time]
        msg_bytes_send = pickle.dumps(msg_send)
        try:
            await websocket.send(msg_bytes_send)
        except websockets.ConnectionClosedOK:
            print('Connection closed by the client',flush = True)                   