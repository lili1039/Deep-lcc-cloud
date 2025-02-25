import numpy as np
import pickle
import time
import redis
import math
import websockets
from scipy.linalg import block_diag 
import socket
import threading
import queue

# update method
Hankel_update_method = 1
# 1: with base dataset 2: without base dataset
Inv_method = 2
# 1: Inv by Recursion 2: Inv by numpy

# check stopping criteria
Is_Check =  True # 要在main-server.py对应改一下

# communication
Com_way = 2
# 1:TCP 2:Redis

iteration_num = 1000 # 要在main-server.py对应改一下

def send_data(host, port, data, max_retries=100, retry_interval=1):
    """向指定地址和端口的 TCP 服务器发送数据，若连接失败则不断重试"""
    attempt = 0
    while attempt < max_retries:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((host, port))
                sock.sendall(pickle.dumps(data))
                # print(f"数据成功发送到 {host}:{port}")
                return  # 发送成功后返回
        except (ConnectionRefusedError, TimeoutError) as e:
            print(f"连接 {host}:{port} 失败，错误: {e}，尝试重新连接 ({attempt + 1}/{max_retries})",flush=True)
            attempt += 1
            time.sleep(retry_interval)
    
    print(f"多次尝试后仍无法连接 {host}:{port}，放弃发送。")

def receive_data(port, stop_event, data_queue): 
    """监听指定端口，接收数据并存入队列"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("0.0.0.0", port))
        server_sock.listen(1)
        print(f"接收服务器已启动，监听端口: {port}")

        while not stop_event.is_set():
            try:
                conn, _ = server_sock.accept()
                with conn:
                    data = pickle.loads(conn.recv(4096))
                    data_queue.put(data)  # 将数据存入队列
            except OSError:
                break  # ✅ 关键：server_sock 关闭后抛出 OSError，直接退出循环
            except Exception as e:
                print(f"接收数据时出错: {e}")
                break  # 避免无限循环错误

def get_data_by_type(data_queue, desired_type):
    """从队列中获取指定类型的数据，如果没有则阻塞等待"""
    start_time = time.time()
    while True:
        try:
            # 获取队列中的第一个数据
            data = data_queue.get()  # 阻塞直到队列中有数据
            
            if data['type'] == desired_type:
                # 找到符合条件的类型，返回
                return data['data']
            else:
                # 如果类型不符，将数据放回队列等待下一次处理
                data_queue.put(data)
            
            # if time.time() - start_time >= timeout:
            #     print(f"超时未收到数据:{desired_type}")
        except queue.Empty:
            print("队列为空，等待数据...",flush=True)
            continue

def dDeeP_LCC(Tini,N,kappa,timestep,n_cav,cav_id,Uip,Yip,Uif,Yif,Eip,Eif,ui_ini,yi_ini,ei_ini,g_initial,mu_initial,eta_initial,phi_initial,theta_initial,delta_initial,lambda_yi,u_limit,s_limit,rho,Hgi_vert,Hz_vert,rs,data_queue):

    # stoping criterion
    error_absolute = 0.1  # 0.1
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

    # 计算当前容器的端口
    my_port = 5005 + cav_id
    my_host = f"veh-{cav_id}"
    next_port = 5005 + (cav_id + 1) if cav_id != n_cav - 1 else None
    prev_port = 5005 + (cav_id - 1) if cav_id != 0 else None
    prev_host = f"veh-{cav_id - 1}" if cav_id != 0 else None
    next_host = f"veh-{cav_id + 1}" if cav_id != n_cav - 1 else None
    
    other_port = [5005 + i for i in range(n_cav) if i != cav_id]
    other_host = [f"veh-{i}" for i in range(n_cav) if i != cav_id]

    for k in range(iteration_num):
        # time_step12_start = time.time()
        # Step1: update g
            # updata eta_bar
            # 对于i=1~n-1的CAV计算eta_bar并发送给后一个CAVi+1

        if Com_way == 2:
            if cav_id != n_cav-1:
                eta_bar = eta - rho*K@Yif@z
                rs.mset({f'eta_bar_{cav_id}_{timestep}_{k}':pickle.dumps(eta_bar)})

            if cav_id != 0:
                while True:
                    if rs.mget(f'eta_bar_{cav_id-1}_{timestep}_{k}')[0] != None:
                        eta_bar_former = pickle.loads(rs.mget(f'eta_bar_{cav_id-1}_{timestep}_{k}')[0])
                        break
        elif Com_way == 1:
            if cav_id != n_cav-1:
                eta_bar = eta - rho * K @ Yif @ z
                send_data(next_host, my_port, {'type':f'eta_bar_{cav_id}_{timestep}_{k}','data':eta_bar})

            if cav_id != 0:
                eta_bar_former = get_data_by_type(data_queue, f'eta_bar_{cav_id-1}_{timestep}_{k}')

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
        
        if Com_way == 2:
            if cav_id != 0:
                epsilon_bar = Eif@g_plus
                rs.mset({f'epsilon_bar_{cav_id}_{timestep}_{k}':pickle.dumps(epsilon_bar)})

            if cav_id != n_cav-1:
                while True:
                    if rs.mget(f'epsilon_bar_{cav_id+1}_{timestep}_{k}')[0] != None:
                        epsilon_bar_latter = pickle.loads(rs.mget(f'epsilon_bar_{cav_id+1}_{timestep}_{k}')[0])
                        break
        elif Com_way == 1:
            if cav_id != 0:
                epsilon_bar = Eif@g_plus
                send_data(prev_host,my_port,{'type':f'epsilon_bar_{cav_id}_{timestep}_{k}','data':epsilon_bar})

            if cav_id != n_cav-1:
                epsilon_bar_latter = get_data_by_type(data_queue, f'epsilon_bar_{cav_id+1}_{timestep}_{k}')
        
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
            if Is_Check:
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

        #         # 计算迭代停止条件并存入数据库
        #         temp_pri0 = Agi@g_plus
        #         error_pri0 = np.linalg.norm(temp_pri0 - beqg)
        #         tolerance_pri0 = math.sqrt(beqg.shape[0])*error_absolute + \
        #             error_relative*max(np.linalg.norm(temp_pri0),np.linalg.norm(beqg))
                
        #         error_pri1 = np.linalg.norm(g_plus - z_plus)
        #         tolerance_pri1 = math.sqrt(kappa)*error_absolute + \
        #             error_relative*max(np.linalg.norm(g_plus),np.linalg.norm(z_plus))

        #         error_dual1 = rho*np.linalg.norm(z_plus-z)
        #         tolerance_dual1 = math.sqrt(kappa)*error_absolute + error_relative*np.linalg.norm(mu_plus)

        #         error_pri2 = np.linalg.norm(epsilon_bar_latter - K@Yif@z_plus)
        #         tolerance_pri2 = math.sqrt(N)*error_absolute + \
        #             error_relative*max(np.linalg.norm(epsilon_bar_latter),np.linalg.norm(K@Yif@z_plus))
                
        #         Eif_latter = pickle.loads(rs.mget(f'Eif_in_CAV_{cav_id+1}')[0])
        #         error_dual2 = rho*np.linalg.norm(Eif_latter.T@K@Yif@(z_plus-z))    
        #         tolerance_dual2 = math.sqrt(N)*error_absolute + error_relative*np.linalg.norm(Eif_latter.T@eta_plus)  
                
        #         error_pri3 = np.linalg.norm(s_plus - P@Yif@g_plus)
        #         tolerance_pri3 = math.sqrt(s_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(s_plus),np.linalg.norm(P@Yif@g_plus))
                
        #         error_dual3 = rho*np.linalg.norm(Yif.T@P.T@(s_plus-s))
        #         tolerance_dual3 = math.sqrt(kappa)*error_absolute + error_relative*np.linalg.norm(Yif.T@P.T@phi_plus)
                
        #         error_pri4 =  np.linalg.norm(u_plus-Uif@g_plus)
        #         tolerance_pri4 = math.sqrt(N)*error_absolute + error_relative*max(np.linalg.norm(Uif@g_plus),np.linalg.norm(u_plus))
                
        #         error_dual4 = rho*np.linalg.norm(Uif.T@(u_plus-u))
        #         tolerance_dual4 = math.sqrt(kappa)*error_absolute + error_relative*np.linalg.norm(Uif.T@theta_plus)

        #         # print(f'check:calculation {time.time()-start_time_check}',flush=True)

        #         if Com_way == 2:
        #             if error_pri0 <= tolerance_pri0 and error_pri1 <= tolerance_pri1 and error_dual1 <= tolerance_dual1 and error_pri2 <= tolerance_pri2 and error_dual2 <= tolerance_dual2 and error_pri3 <= tolerance_pri3 and error_dual3 <= tolerance_dual3 and error_pri4 <= tolerance_pri4 and error_dual4 <= tolerance_dual4:
        #                 rs.mset({f'rollout_flag_{cav_id}_{timestep}_{k}':pickle.dumps(1)})
        #             else:
        #                 rs.mset({f'rollout_flag_{cav_id}_{timestep}_{k}':pickle.dumps(0)})
                        
        #         elif Com_way == 1:
        #             if error_pri0 <= tolerance_pri0 and error_pri1 <= tolerance_pri1 and error_dual1 <= tolerance_dual1 and error_pri2 <= tolerance_pri2 and error_dual2 <= tolerance_dual2 and error_pri3 <= tolerance_pri3 and error_dual3 <= tolerance_dual3 and error_pri4 <= tolerance_pri4 and error_dual4 <= tolerance_dual4:
        #                 for host in other_host:
        #                     send_data(host, my_port, {'type':f'rollout_flag_{cav_id}_{timestep}_{k}','data':True})
        #                 my_flag = True
        #             else:
        #                 for host in other_host:
        #                     send_data(host, my_port, {'type':f'rollout_flag_{cav_id}_{timestep}_{k}','data':False})
        #                 my_flag = False
                
        #         # print(f'check:send {time.time()-start_time_check}',flush=True)
        #         # print(f'rollout_flag_{cav_id}_{timestep}_{k} send.',flush=True)

        #     eta = eta_plus

        # else:
        #     qz = -mu/2 - rho/2*g_plus
        #     z_plus = -Hz_vert@qz
            
        #     s_temp = P@Yif@g_plus - phi/rho
        #     s_plus = np.minimum(np.maximum(s_temp,s_limit[0]*np.ones(N)),s_limit[1]*np.ones(N)) # projection in the box constraint

        #     u_temp = Uif@g_plus-theta/rho
        #     u_plus = np.minimum(np.maximum(u_temp,u_limit[0]*np.ones(N)),u_limit[1]*np.ones(N)) # projection in the box constraint

        #     mu_plus = mu + rho*(g_plus-z_plus)
        #     phi_plus = phi + rho*(s_plus - P@Yif@g_plus)
        #     theta_plus = theta + rho*(u_plus - Uif@g_plus)
        #     delta_plus = delta + rho*(Agi@g_plus-beqg)

        #     if Is_Check:

        #         start_time_check = time.time()

        #         temp_pri0 =  Agi@g_plus
        #         error_pri0 = np.linalg.norm(temp_pri0 - beqg)
        #         tolerance_pri0 = math.sqrt(beqg.shape[0])*error_absolute + \
        #             error_relative*max(np.linalg.norm(temp_pri0),np.linalg.norm(beqg))
                
        #         error_pri1 = np.linalg.norm(g_plus - z_plus)
        #         tolerance_pri1 = math.sqrt(g_plus.shape[0])*error_absolute + \
        #             error_relative*max(np.linalg.norm(g_plus),np.linalg.norm(z_plus))
                
        #         error_dual1 = rho*np.linalg.norm(z_plus-z)            
        #         tolerance_dual1 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(mu_plus)
                
        #         error_pri3 = np.linalg.norm(s_plus - P@Yif@g_plus)
        #         tolerance_pri3 = math.sqrt(s_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(s_plus),np.linalg.norm(P@Yif@g_plus))
                
        #         error_dual3 = rho*np.linalg.norm(Yif.T@P.T@(s_plus-s))
        #         tolerance_dual3 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Yif.T@P.T@phi_plus)
                
        #         error_pri4 =  np.linalg.norm(u_plus-Uif@g_plus)
        #         tolerance_pri4 = math.sqrt(u_plus.shape[0])*error_absolute + error_relative*max(np.linalg.norm(Uif@g_plus),np.linalg.norm(u_plus))
                
        #         error_dual4 = rho*np.linalg.norm(Uif.T@(u_plus-u))
        #         tolerance_dual4 = math.sqrt(z_plus.shape[0])*error_absolute + error_relative*np.linalg.norm(Uif.T@theta_plus)

        #         # print(f'check:calculation {time.time()-start_time_check}',flush=True)

        #         if Com_way == 2:
        #             if error_pri0 <= tolerance_pri0 and error_pri1 <= tolerance_pri1 and error_dual1 <= tolerance_dual1 and error_pri3 <= tolerance_pri3 and error_dual3 <= tolerance_dual3 and error_pri4 <= tolerance_pri4 and error_dual4 <= tolerance_dual4:
        #                 rs.mset({f'rollout_flag_{cav_id}_{timestep}_{k}':pickle.dumps(1)})
                        
        #             else:
        #                 rs.mset({f'rollout_flag_{cav_id}_{timestep}_{k}':pickle.dumps(0)})
                        
        #         elif Com_way == 1:
        #             if error_pri0 <= tolerance_pri0 and error_pri1 <= tolerance_pri1 and error_dual1 <= tolerance_dual1 and error_pri3 <= tolerance_pri3 and error_dual3 <= tolerance_dual3 and error_pri4 <= tolerance_pri4 and error_dual4 <= tolerance_dual4:
        #                 for host in other_host:
        #                     send_data(host, my_port, {'type':f'rollout_flag_{cav_id}_{timestep}_{k}','data':True})
        #                 my_flag = True
        #             else:
        #                 for host in other_host:
        #                     send_data(host, my_port, {'type':f'rollout_flag_{cav_id}_{timestep}_{k}','data':False})
        #                 my_flag = False
        
                
        #         print(f'check:send {time.time()-start_time_check}',flush=True)
        #         print(f'rollout_flag_{cav_id}_{timestep}_{k} send.',flush=True)

        # # Step3: error计算完毕，可以由main-server检查stop criteria是否满足
        # # error和tolerance计算完成后，将check_ready标为1，表示该容器已经上传好信息，等待停止迭代信号
        # g = g_plus
        # z = z_plus
        # u = u_plus
        # s = s_plus
        # mu = mu_plus
        # phi = phi_plus
        # theta = theta_plus
        # delta = delta_plus

        # if k == iteration_num-1:
        #     break
        # else:
        #     if Is_Check:
        #         if Com_way == 2:
        #             while True:
        #                 check_flag_bytes=rs.mget(f'rollout_flag_total_{timestep}_{k}')[0]
        #                 if check_flag_bytes==None:
        #                         continue
        #                 else:
        #                     check_flag = pickle.loads(check_flag_bytes)
        #                     break

        #             if check_flag == 0:
        #                 continue
        #             elif check_flag==1:
        #                 break

        #         elif Com_way == 1:

        #             while True:
        #                 check_flag = []
        #                 for i in range(n_cav):
        #                     if i != cav_id:
        #                         flag = get_data_by_type(data_queue, f'rollout_flag_{i}_{timestep}_{k}')
        #                         check_flag.append(flag)
        #                     else:
        #                         check_flag.append(my_flag)
                        
        #                 # print(check_flag,flush=True)
        #                 if False in check_flag:
        #                     iter_flag = 0
        #                     # if cav_id == n_cav-1:
        #                     # print(check_flag,flush=True)
        #                     break
        #                 elif all(var == True for var in check_flag):
        #                     iter_flag = 1
        #                     break
        #                 else:
        #                     continue

        #             if iter_flag == 1:
        #                 # print(f' {timestep}_{k} I break.',flush=True)
        #                 break
        #             else:
        #                 # print(f' {timestep}_{k} I continue.',flush=True)
        #                 continue

        #     else:
        #         continue

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
        # read_redis_start = time.time()
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

        # read_redis_start = time.time()
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

# solely add a new column at the last
def Hankel_substitute_col_0(Tini,N,Su,Sy,Se,Uip,Uif,Yip,Yif,Eip,Eif,k,p,g_initial,mu_initial):
    new_up = Su[0, k:k + Tini]
    Up_temp = np.hstack((Uip[:,:],new_up.reshape([int(Tini),1])))

    new_uf = Su[0, k + Tini:k + Tini + N]
    Uf_temp = np.hstack((Uif[:,:],new_uf.reshape([int(N),1])))

    new_yp = Sy[:, k:k + Tini].reshape(-1, order='F')
    Yp_temp = np.hstack((Yip[:,:],new_yp.reshape([int(p*Tini),1])))

    new_yf = Sy[:, k + Tini:k + Tini + N].reshape(-1, order='F')
    Yf_temp = np.hstack((Yif[:,:],new_yf.reshape([int(p*N),1])))

    new_ep = Se[0, k:k + Tini]
    Ep_temp = np.hstack((Eip[:,:],new_ep.reshape([int(Tini),1])))

    new_ef = Se[0, k + Tini:k + Tini + N]
    Ef_temp = np.hstack((Eif[:,:],new_ef.reshape([int(N),1])))

    g_initial = np.hstack((g_initial[:],0))
    mu_initial = np.hstack((mu_initial[:],0))

    c = np.hstack((new_up,new_ep,new_yp,new_uf,new_ef,new_yf))

    return [Up_temp,Uf_temp,Yp_temp,Yf_temp,Ep_temp,Ef_temp,g_initial,mu_initial,c]

# delete the first column and add a new column at the last
def Hankel_substitute_col_1(Tini,N,Su,Sy,Se,Uip,Uif,Yip,Yif,Eip,Eif,k,p,g_initial,mu_initial):
    new_up = Su[0, k:k + Tini]
    Up_temp = np.hstack((Uip[:,1:],new_up.reshape([int(Tini),1])))

    new_uf = Su[0, k + Tini:k + Tini + N]
    Uf_temp = np.hstack((Uif[:,1:],new_uf.reshape([int(N),1])))

    new_yp = Sy[:, k:k + Tini].reshape(-1, order='F')
    Yp_temp = np.hstack((Yip[:,1:],new_yp.reshape([int(p*Tini),1])))

    new_yf = Sy[:, k + Tini:k + Tini + N].reshape(-1, order='F')
    Yf_temp = np.hstack((Yif[:,1:],new_yf.reshape([int(p*N),1])))

    new_ep = Se[0, k:k + Tini]
    Ep_temp = np.hstack((Eip[:,1:],new_ep.reshape([int(Tini),1])))

    new_ef = Se[0, k + Tini:k + Tini + N]
    Ef_temp = np.hstack((Eif[:,1:],new_ef.reshape([int(N),1])))

    g_initial = np.hstack((g_initial[1:],0))
    mu_initial = np.hstack((mu_initial[1:],0))

    c = np.hstack((new_up,new_ep,new_yp,new_uf,new_ef,new_yf))

    return [Up_temp,Uf_temp,Yp_temp,Yf_temp,Ep_temp,Ef_temp,g_initial,mu_initial,c]

# delete the first column of Online part and add a new column at the last
def Hankel_substitute_col_2(Tini,N,k_on,Su,Sy,Se,Uip,Uif,Yip,Yif,Eip,Eif,k,p,g_initial,mu_initial):
    # k_on is the index of the beginning of Online part, and also the minimal number of column that ensure offline data satisfying P.E. condition.

    new_up = Su[0, k:k + Tini]
    Up_temp = np.hstack((Uip[:,0:k_on],Uip[:,k_on+1:],new_up.reshape([int(Tini),1])))

    new_uf = Su[0, k + Tini:k + Tini + N]
    Uf_temp = np.hstack((Uif[:,0:k_on],Uif[:,k_on+1:],new_uf.reshape([int(N),1])))

    new_yp = Sy[:, k:k + Tini].reshape(-1, order='F')  # Reshape column-wise
    Yp_temp = np.hstack((Yip[:,0:k_on],Yip[:,k_on+1:],new_yp.reshape([int(p*Tini),1])))

    new_yf = Sy[:, k + Tini:k + Tini + N].reshape(-1, order='F')
    Yf_temp = np.hstack((Yif[:,0:k_on],Yif[:,k_on+1:],new_yf.reshape([int(p*N),1])))

    new_ep = Se[0, k:k + Tini]
    Ep_temp = np.hstack((Eip[:,0:k_on],Eip[:,k_on+1:],new_ep.reshape([int(Tini),1])))

    new_ef = Se[0, k + Tini:k + Tini + N]
    Ef_temp = np.hstack((Eif[:,0:k_on],Eif[:,k_on+1:],new_ef.reshape([int(N),1])))

    g_initial = np.hstack((g_initial[0:k_on],g_initial[k_on+1:],0))
    mu_initial = np.hstack((mu_initial[0:k_on],mu_initial[k_on+1:],0))

    c = np.hstack((new_up,new_ep,new_yp,new_uf,new_ef,new_yf))

    return [Up_temp,Uf_temp,Yp_temp,Yf_temp,Ep_temp,Ef_temp,g_initial,mu_initial,c]


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

    async def solver(self,websocket, path, data_queue):
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

        # 从数据库获取初始参数和数据
        keys = [
            f'Uip_in_CAV_{self.cav_id}',
            f'Uif_in_CAV_{self.cav_id}',
            f'Eip_in_CAV_{self.cav_id}',
            f'Eif_in_CAV_{self.cav_id}',
            f'Yip_in_CAV_{self.cav_id}',
            f'Yif_in_CAV_{self.cav_id}',
            f'lambda_yi',
            f'lambda_gi_max',
            f'lambda_gi_min',
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
            self.lambda_yi, lambda_gi_max, lambda_gi_min, self.rho,
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
        # lambda_gi_on = np.linspace(lambda_gi_max, lambda_gi_min, int(kappa-k_on[self.cav_id]))
        kappa_max = 400
        kappa0 = 300-60+1
        
        # print(f'1prepare {time.time()-start_time}',flush=True)

        start_time = time.time()

        if Hankel_update_flag:
            # generate new Hankel Matrices
            if Hankel_update_method == 1 :
                if kappa < kappa_max:
                    print(f"列数{kappa},增加新一列",flush=True)
                    result = Hankel_substitute_col_0(Tini,N,Su,Sy,Se,self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,curr_step,p,g_initial,mu_initial)
                    kappa = kappa+1
                else:
                    off_col = int(max(kappa - curr_step,k_on[self.cav_id]))
                    if  off_col > k_on[self.cav_id]:
                        print(f"列数{kappa},offline剩余{off_col}列，删掉第一列增加新一列",flush=True)
                        result = Hankel_substitute_col_1(Tini,N,Su,Sy,Se,self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,curr_step,p,g_initial,mu_initial)
                    else: 
                        print(f"列数{kappa},offline剩余{off_col}列，删掉online第一列增加新一列",flush=True)
                        result = Hankel_substitute_col_2(Tini,N,int(k_on[self.cav_id]),Su,Sy,Se,self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,curr_step,p,g_initial,mu_initial)         

                # calculate inverse of matrices Hgi and Hz
                if Inv_method == 2 or curr_step == 0: # inverse by numpy
                    [self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,g_initial,mu_initial,c] = result

                    M = block_diag(self.rho/2*np.eye(Tini),self.rho/2*np.eye(Tini),self.lambda_yi*np.eye(p*Tini),Ri_stack+self.rho/2*np.eye(N),self.rho/2*np.eye(N),Qi_stack+self.rho/2*P.T@P)
                    H = np.vstack((self.Uip,self.Eip,self.Yip,self.Uif,self.Eif,self.Yif))

                    # Hgi_vert
                    lambda_gi = np.linspace(lambda_gi_max, lambda_gi_min,kappa)
                    Hgi = H.T@M@H + (self.rho/2+lambda_gi)*np.eye(kappa)
                    Hgi_vert = np.linalg.inv(Hgi)
                    
                    # Hz_vert
                    if self.cav_id != n_cav-1:
                        Hz = self.rho/2*np.eye(int(kappa))+self.rho/2*self.Yif.T@K.T@K@self.Yif
                        Hz_vert = np.linalg.inv(Hz)

                    else:
                        Hz_vert = (2/self.rho)*np.eye(int(kappa))
            
            elif Hankel_update_method == 2:
                result = Hankel_substitute_col_1(Tini,N,Su,Sy,Se,self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,curr_step,p,g_initial,mu_initial)
                [self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,g_initial,mu_initial,c] = result

                M = block_diag(self.rho/2*np.eye(Tini),self.rho/2*np.eye(Tini),self.lambda_yi*np.eye(p*Tini),Ri_stack+self.rho/2*np.eye(N),self.rho/2*np.eye(N),Qi_stack+self.rho/2*P.T@P)
                H = np.vstack((self.Uip,self.Eip,self.Yip,self.Uif,self.Eif,self.Yif))
                Hgi = H.T@M@H + (self.rho/2+lambda_gi)*np.eye(kappa)

                # time_hgi_start = time.time()
                Hgi_vert = np.linalg.inv(Hgi)
                if self.cav_id != n_cav-1:
                    Hz = self.rho/2*np.eye(int(kappa))+self.rho/2*self.Yif.T@K.T@K@self.Yif
                    Hz_vert = np.linalg.pinv(Hz)
                else:
                    Hz_vert = (2/self.rho)*np.eye(int(kappa))
            
            [self.Uip,self.Uif,self.Yip,self.Yif,self.Eip,self.Eif,g_initial,mu_initial,c] = result

        elif ~Hankel_update_flag:
            lambda_gi = np.full(kappa, lambda_gi_max)
            if curr_step == 0:
                M = block_diag(self.rho/2*np.eye(Tini),self.rho/2*np.eye(Tini),self.lambda_yi*np.eye(p*Tini),Ri_stack+self.rho/2*np.eye(N),self.rho/2*np.eye(N),Qi_stack+self.rho/2*P.T@P)
                H = np.vstack((self.Uip,self.Eip,self.Yip,self.Uif,self.Eif,self.Yif))

                # Hgi_vert
                Hgi = H.T@M@H + (self.rho/2+lambda_gi)*np.eye(kappa)
                Hgi_vert = np.linalg.inv(Hgi)
                
                rs.mset({f'Hgi_vert_in_CAV_{self.cav_id}':pickle.dumps(Hgi_vert)})
                
                # Hzi_vert
                if self.cav_id != n_cav-1:
                    Hz = self.rho/2*np.eye(int(kappa))+self.rho/2*self.Yif.T@K.T@K@self.Yif
                    Hz_vert = np.linalg.inv(Hz)
                else:
                    Hz_vert = (2/self.rho)*np.eye(int(kappa))
                
                rs.mset({f'Hz_vert_in_CAV_{self.cav_id}':pickle.dumps(Hz_vert)})
            else:
                Hgi_vert = pickle.loads(rs.mget(f'Hgi_vert_in_CAV_{self.cav_id}')[0])
                Hz_vert = pickle.loads(rs.mget(f'Hz_vert_in_CAV_{self.cav_id}')[0])

        
        # 调用deepc
        u_opt,g_opt,mu_opt,eta_opt,phi_opt,theta_opt,delta_opt,real_iter_num = dDeeP_LCC(Tini,N,kappa,curr_step,n_cav,self.cav_id,self.Uip,self.Yip,self.Uif,\
            self.Yif,self.Eip,self.Eif,self.uini,self.yini,self.eini,g_initial,mu_initial,eta_initial,phi_initial,theta_initial,delta_initial,\
            self.lambda_yi,self.u_limit,self.s_limit,self.rho,Hgi_vert,Hz_vert,rs,data_queue)

        use_time = time.time() - start_time
        print(f"total computation time: {use_time}; iteration number: {real_iter_num}",flush=True)

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

        # cost
        # y_cost = (self.Yif@g_opt).T@Qi_stack@(self.Yif@g_opt)
        # u_cost = (u_opt).T@Ri_stack@(u_opt)
        # gi_cost = lambda_gi*g_opt.T@g_opt
        # sig_yi_cost = self.lambda_yi*(self.Yip@g_opt-(self.yini.T).reshape(-1)).T@(self.Yip@g_opt-(self.yini.T).reshape(-1))
        # print(f'y/u/gi/sig_yi cost:{y_cost}/{u_cost}/{gi_cost}/{sig_yi_cost}',flush=True)

        # give messages to clients
        msg_send = [u_opt[0],real_iter_num,use_time]
        msg_bytes_send = pickle.dumps(msg_send)
        try:
            await websocket.send(msg_bytes_send)
        except websockets.ConnectionClosedOK:
            print('Connection closed by the client',flush = True)      