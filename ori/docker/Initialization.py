import numpy as np
import csv

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


class InitParameter:
    def __init__(self):
        # 车辆参数
        self.Num_veh = 8

        data = []
        with open('vehicle_param.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                data.append(row)

        self.Mass   = np.array(list(map(eval,[row[1] for row in data])))
        self.Radius = np.array(list(map(eval,[row[2] for row in data])))
        self.f      = np.array(list(map(eval,[row[3] for row in data])))
        self.Ca     = np.array(list(map(eval,[row[4] for row in data])))
        self.Eta    = np.array(list(map(eval,[row[5] for row in data])))
        self.Tao    = np.array(list(map(eval,[row[6] for row in data])))
        self.g      = 9.8
        # 控制参数
        self.Np = 20
        self.Time_step = 0.1
        self.Time_sim = 8
        self.Num_step = 100            # Simulation setps

        self.d = 20

class InitVariable:
    def __init__(self, param):
        Num_step = param.Num_step
        Num_veh = param.Num_veh
        Time_step = param.Time_step

        # Leading vehicle
        self.a0 = np.zeros([Num_step,1])
        self.v0 = np.zeros([Num_step,1])
        self.x0 = np.zeros([Num_step,1])
        v_init = 20

        # 领导车辆期望运动轨迹
        self.v0[0] = 20
        self.a0[10:20] = 3
        for i in range(1,Num_step):
            self.v0[i] = self.v0[i-1]+self.a0[i]*Time_step
            self.x0[i] = self.x0[i-1]+self.v0[i]*Time_step

        self.Position = np.zeros([1,Num_veh])     # postion of each vehicle;
        self.Velocity = np.zeros([1,Num_veh])     # velocity of each vehicle;
        self.Torque   = np.zeros([1,Num_veh])     # Braking or Tracking Torque of each vehicle;

        # 各车辆状态量的初始值，处于稳态
        for i in range(Num_veh):
            self.Position[0,i] = self.x0[0] - (i+1)*param.d
            self.Velocity[0,i] = v_init
            self.Torque[0,i]   = (param.Mass[i]*param.g*param.f[i]+param.Ca[i]*self.Velocity[0,i]**2)*param.Radius[i]/param.Eta[i]


class InitAssumeState:
    def __init__(self,param,var):
        Np = param.Np
        Num_veh = param.Num_veh

        self.Pa = np.zeros([Np+1,Num_veh])      # Assumed postion of each vehicle;
        self.Va = np.zeros([Np+1,Num_veh])      # Assumed velocity of each vehicle;
        self.Ta = np.zeros([Np+1,Num_veh])
        self.ua = np.zeros([Np,Num_veh])      # Assumed Braking or Tracking Torque input of each vehicle;

        # Initialzie the assumed state for the first computation: constant speed
        for i in range(Num_veh):
            self.ua[:,i] = var.Torque[0,i]    # 初始假定序列持续施加以稳态输入
            self.Pa[0,i] = var.Position[0,i]  # 每辆车的初始位置
            self.Va[0,i] = var.Velocity[0,i]  # 每辆车的初始速度
            self.Ta[0,i] = var.Torque[0,i]    # 每辆车的力矩
            Vehicle_Type = [param.Mass[i],param.Radius[i],param.g,param.f[i],param.Eta[i],param.Ca[i],param.Tao[i]]
            Xi = [self.Pa[0,i],self.Va[0,i],self.Ta[0,i]]
            for k in range(Np):
                Xi = NonlinearDynamic(Xi,self.ua[k,i],param.Time_step,Vehicle_Type)
                self.Pa[k+1,i] = Xi[0]; self.Va[k+1,i] = Xi[1]; self.Ta[k+1,i] = Xi[2]