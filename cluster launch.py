import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from numpy.linalg import norm
from numpy import cos
from numpy import sin
from numpy import pi



def show_B(t, Ac, N, t_convergence):
    plt.figure()
    for j in range(4):
        plt.subplot(2, 2, j+1)
        for i in range(N):
            plt.plot(t[2:], Ac[i, 3+j, 2:], linewidth=0.5)
            plt.xlabel('t')
        B = 'B' + str(j+1)
        plt.ylabel(B)
        plt.legend(['1', '2', '3', '4', '5'])
        plt.axvline(x=t_convergence, color='g', linestyle='--')
        plt.grid(True, linestyle=":", alpha=0.5)


def show_trajectories_LVLH(t, X, N):
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    for i in range(N):
        ax2.plot(X[i, 0, :], X[i, 1, :], X[i, 2, :], label='parametric curve', linewidth=0.5)
    plt.legend(['1', '2', '3', '4', '5'])
    ax2.set_xlabel('x, м', fontsize=15)
    ax2.set_ylabel('y, м', fontsize=15)
    ax2.set_zlabel('z, м', fontsize=15)
    lim2 = abs(X).max()
    ax2.set_xlim(-lim2, lim2)
    ax2.set_ylim(-lim2, lim2)
    ax2.set_zlim(-lim2, lim2)
    ax2.set_title('Траектории спутников в ОСК')


def show_Lyapunov(t, Ac, N):
    plt.figure()
    for i in range(N):
        plt.plot(t[2:], Ac[i, 0, 2:], linewidth=0.5)
    plt.title('Функция Ляпунова')
    plt.xlabel('t')
    plt.ylabel('V')
    plt.legend(['спутник 1', 'спутник 2', 'спутник 3', 'спутник 4', 'спутник 5'])
    plt.grid(True, linestyle=":", alpha=0.5)


def show_control_(t, Ac, N, t_convergence, u_max):
    plt.figure()
    for j in range(3):
        plt.subplot(3, 1, j+1)
        for i in range(N):
            plt.plot(t[2:], Ac[i, 7+j, 2:], linewidth=0.5)
        plt.xlabel('t')
        u = 'u' + chr(120+j) + ' м/с^2'
        plt.ylabel(u)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.axvline(x=t_convergence, color='g', linestyle='--')
        plt.axhline(y=u_max, color='r', linestyle='--')
        plt.legend(['1', '2', '3', '4', '5'], loc='right')

#
# def boxplotDiagram():



def RightPart_u(t, x, a_u):
    mu = 398600.4415*10**9
    J2 = 1082.6*10**(-6)
    RE = 6.378*10**(6)
    l = (norm(x[:3]))
    Z = np.array([0, 0, x[2]])
    delta = (3/2) * J2 * mu * (RE**2)
    dxdt = np.zeros(6)
    dxdt[:3] = x[3:]                #velocity
    dxdt[3:] = -mu*x[:3]/l**3 + a_u + delta/(l**5) * ((5*x[2]**2/l**2 - 1) * x[:3] - 2*Z) #acceleration
    return dxdt


def Set_X0(C, w):
    X0 = np.array([C[3] + 2 * C[1], C[5], 2 * C[0] + C[2], -3 * w * C[0] - 2 * w * C[2], w * C[4], w * C[1]])
    return(X0)


def Set_x0(X0, x0_0):
    x0 = LVLH2IF(X0, x0_0)
    return x0


def get_G(x1):
    R1 = x1.copy()[:3]
    V1 = x1.copy()[3:]
    G = np.zeros((3, 3))
    G[:, 2] = R1 / norm(R1)  # матрица перехода от иск к оск
    G[:, 1] = np.cross(R1, V1) / norm(np.cross(R1, V1))
    G[:, 0] = np.cross(G[:, 1], G[:, 2])
    return G


def get_omega(x1):
    R1 = x1.copy()[:3]
    V1 = x1.copy()[3:]
    omega = np.cross(R1, V1) / norm(np.cross(R1, V1)) * (norm(V1) / norm(R1))
    return omega


def IF2LVLH(x2, x1):
    X = np.zeros((6))
    R1 = x1.copy()[:3]
    V1 = x1.copy()[3:]
    R2 = x2.copy()[:3]
    V2 = x2.copy()[3:]
    G = get_G(x1)
    omega = get_omega(x1)
    X[:3] = np.dot(G.T, (R2 - R1))
    X[3:] = np.dot(G.T, (V2 - V1) - np.cross(omega, (R2 - R1)))
    return X


def LVLH2IF(X_2, x_1):
    omega = get_omega(x_1)
    G = get_G(x_1)
    x_2 = np.zeros(6)
    x_2[:3] = x_1[:3] + np.dot(G, X_2[:3])
    x_2[3:] = x_1[3:] + np.dot(G, X_2[3:]) + np.cross(omega, np.dot(G, X_2[:3]))
    return x_2


def a_LVLH2IF(a, G):
    A = np.dot(G, a)
    return A


def C2X(C, w):
    X = np.array([C[3] + 2 * C[1], C[5], 2 * C[0] + C[2], -3 * w * C[0] - 2 * w * C[2], w * C[4], w * C[1]])
    return X


def q_current(x0, x2):
    X = IF2LVLH(x2, x0)
    rho = norm(x2[:3]) - norm(x0[:3])
    a0 = norm(x0[:3])
    phi = np.arctan2(X[0], (X[2] + a0))
    theta = np.arcsin(X[1] / norm(x2[:3]))
    phi_dot = (X[3] * cos(phi) - X[5] * sin(phi)) / (cos(theta) * (rho + a0))
    theta_dot = (X[4] * cos(theta) - X[3] * sin(phi) * sin(theta) - X[5] * cos(phi) * sin(theta)) / (rho + a0)
    rho_dot = X[3] * sin(phi) / cos(theta) + X[5] * cos(phi) / cos(theta) + (rho + a0) * theta_dot * np.tan(theta)
    q = np.array([phi, theta, rho, phi_dot, theta_dot, rho_dot])
    return q


def q2B(q, omega_i, a0):
    B1 = q[3] / norm(omega_i) + 2 * q[2] / a0
    B2 = np.sqrt((q[5] / (norm(omega_i) * a0)) ** 2 + (-2 * q[3] / norm(omega_i) - 3 * q[2] / a0) ** 2)
    B3 = np.sqrt((q[4] / norm(omega_i)) ** 2 + q[1] ** 2)
    B4 = -2 * q[5] / (norm(omega_i) * a0) + q[0]
    B = np.array([B1, B2, B3, B4])
    return B


def C2B(C, w, omega, a0, x0):
    X = C2X(C, w)
    x = LVLH2IF(X, x0)
    q = q_current(x0, x)
    B = q2B(q, omega, a0)
    return B


def calculate_B4_req(N, a0, x0, B, satX_num, a_req, B4x):
    w = np.sqrt(mu / (norm(x0[:3])**3))
    omega = get_omega(x0)
    a = np.array([0, 0, 0, a_req, 0, 0])
    a_inB = C2B(a, w, omega, a0, x0)[3]
    a_inB05 = C2B(a/2, w, omega, a0, x0)[3]
    B4_req = np.zeros(N)
    B4_req_left_neiqhbour = B4x - a_inB05
    B4_req_right_neighbour = B4x + a_inB05
    if (satX_num == N-1):
        B4_req[satX_num - 1] = B4_req_left_neiqhbour
        for j in reversed(range(satX_num - 1)):
            B4_req[j] = B4_req[j + 1] - a_inB
    elif(satX_num == 0):
      #  print('=')
        B4_req[satX_num + 1] = B4_req_right_neighbour
        for i in range(satX_num + 2, N):
            B4_req[i] = B4_req[i - 1] + a_inB

    else:
        B4_req[satX_num+1] = B4_req_right_neighbour
        B4_req[satX_num-1] = B4_req_left_neiqhbour
        B4_req[satX_num] = B4x
        for i in range(satX_num+2, N):
            B4_req[i] = B4_req[i-1] + a_inB
        for j in reversed(range(satX_num-1)):
            B4_req[j] = B4_req[j+1] - a_inB
    return B4_req                                          #[B4_1, B4_2, ... , B4_(satX_num - 1), B4_(satX_num), B4_(satX_num + 1), ... B4_N]



def calculate_a_u(k, B_req, omega, q, C_q, G, x0):
    ka = k[0]
    kb = k[1]
    k_phi = k[2]
    k_theta = k[3]
    k_rho = k[4]
    k2 = k[5]
    k4 = k[6]
    B2_req = B_req[1]
    B3_req = B_req[2]
    B4_req = B_req[3]
    a0 = norm(x0[:3])
    #print(B4_req)
    phi_DoubleDot = -(C_q[0] / omega + k2*(C_q[1] - B2_req) * 2 * (2*C_q[0] - q[2]/a0) / (C_q[1] * omega)) * k_phi * C_q[1]**2
    theta_DoubleDot = -q[4] * (C_q[2] - B3_req) / (omega ** 2) * k_theta * C_q[2]
    rho_DoubleDot = -(k2*(C_q[1] - B2_req) * q[5] / (C_q[1]*omega ** 2 * a0 ** 2) - k4*2 * (C_q[3] - B4_req) / (omega * a0)) * k_rho * C_q[1]**2
    #print(rho_DoubleDot)


    if (3*omega * C_q[0] * (C_q[3] - B4_req)*C_q[1]**2 * C_q[2]**2 * k4 * k_rho *k_phi *k_theta < -phi_DoubleDot**2 * C_q[2]**2* k_theta *k_rho - theta_DoubleDot**2 * C_q[1]**2 * k_phi * k_rho - rho_DoubleDot**2 * C_q[2]**2 *k_theta * k_phi):
        phi_DoubleDot = -ka * omega * C_q[0]
        theta_DoubleDot = 0
        rho_DoubleDot = kb * omega * a0 * 0.5 * (C_q[3] - B4_req) - 3 * (omega ** 2) * a0 * C_q[0] * 0.5
        #print("blink")
        #V = (C_q[0]**2 + (C_q[3] - B4_req)**2) * 0.5
        V = (C_q[0] ** 2 + (C_q[3] - B4_req) ** 2 + (C_q[1] - B2_req) ** 2 + (C_q[2] - B3_req) ** 2) * 0.5

    else:
        V = (C_q[0] ** 2 + (C_q[3] - B4_req) ** 2 + (C_q[1] - B2_req) ** 2 + (C_q[2] - B3_req) ** 2) * 0.5

    a_u = np.zeros(3)
    a_u[0] = phi_DoubleDot * (q[2] + a0) * cos(q[1]) + 2*q[3]*q[5]*cos(q[1]) - 2*q[4]*q[3]*(q[2] + a0)*sin(q[1])
    a_u[1] = (q[2] + a0) * (theta_DoubleDot - q[3]**2 * cos(q[1]) * sin(q[1])) + 2*q[5]*q[4]
    a_u[2] = rho_DoubleDot - (q[2] + a0) * (q[3]**2 * (cos(q[1]))**2 + q[4]**2)
    A_u = a_LVLH2IF(a_u, G)
    if(k.all() ==0):
        A_u = np.zeros(3)
    if (norm(A_u) > u_max):
        A_u = A_u / norm(A_u) * u_max
    return A_u, V



def rk4(fnc, t0, tf, x0, n, u):
    t = np.linspace(t0, tf, n)
    h = t[1] - t[0]
    x = np.zeros((x0.shape[0], n))
    x[:, 0] = x0
    for i in range(1, n):
        k1 = fnc(t[i-1], x[:, i-1], u)
        k2 = fnc(t[i-1] + h * 0.5, x[:, i-1] + h * 0.5 * k1, u)
        k3 = fnc(t[i-1] + h * 0.5, x[:, i-1] + h * 0.5 * k2, u)
        k4 = fnc(t[i-1] + h, x[:, i-1] + h * k3, u)
        x[:, i] = x[:, i - 1] + (1/6) * (k1 + 2*k2 + 2*k3 + k4) * h
    return t, x


def rk4_u_OneStep(fnc, h, t_im, x2_im, x1_im, k, B_req): #значения на каждом шаге
    a0 = norm(x1_im[:3])
    G_i = get_G(x1_im)
    omega_i = get_omega(x1_im)
    q = q_current(x1_im, x2_im)  # [phi, theta, rho, phi_dot, theta_dot, rho_dot]
    if (np.isnan(q[1])):
        print('***')
    #print(q)
    B = q2B(q, omega_i, a0)

    a_u, V = calculate_a_u(k, B_req, norm(omega_i), q, B, G_i, x1_im)  # уже в ИСК л
    #print(B_req)
    A_ = np.array([V, q[4], q[5], B[0], B[1], B[2], B[3], a_u[0], a_u[1], a_u[2]])

    k1 = fnc(t_im, x2_im, a_u)
    k2 = fnc(t_im + h * 0.5, x2_im + h * 0.5 * k1, a_u)
    k3 = fnc(t_im + h * 0.5, x2_im + h * 0.5 * k2, a_u)
    k4 = fnc(t_im + h, x2_im + h * k3, a_u)
    x2_i = x2_im + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * h
    return x2_i, A_

def Global_rk4(N, fnc, t0, tf, T, x0_arr, X0_arr, x0, xl, n, k, B_req):
    t = np.linspace(t0, tf, n)
    h = tau

    x = xl
    X = np.zeros((N, 6, n))
    for j in range(N):
        X[j, :, int(T*j / tau) + 1] = IF2LVLH(x[j, :, int(T*j/tau) + 1], x0[:,  int(T*j/tau) + 1])

    Ac = np.zeros((N, 10, n))

    for j in range(N):
        for i in range(int(T*j/tau) + 2, int(T*(N - 1)/tau) + 2):
            x[j, :, i], A = rk4_u_OneStep(fnc, h, t[i - 1], x[j, :, i - 1], x0[:, i - 1], k[j, :], B_req[j, :])
            #print(x[j, :, i])
            X[j, :, i] = IF2LVLH(x[j, :, i], x0[:, i])
            #print(X[j, :, i])
            Ac[j, :, i] = A
    #print(0, x[0, :, int(T*(N - 1)/tau)])

    for i in range(int(T*(N - 1)/tau) + 2, n):
        #print(i)
        for j in range(N):
            # if(np.array_equal(x[j, :, i-1], np.zeros(6))):
            #         #print(x[j, :, i-1])
            #     continue
            # else:
            x[j, :, i], A = rk4_u_OneStep(fnc, h, t[i-1], x[j, :, i-1], x0[:, i-1], k[j, :], B_req[j, :])
            X[j, :, i] = IF2LVLH(x[j, :, i], x0[:, i])
            Ac[j, :, i] = A

        true=0
        t_convergence = n-1
        for j in range(N):
            #if (abs(Ac[j, 4, i] - Ac[j+1, 4, i]) < 1e-7):
            if(abs(Ac[j, 6, i] - B_req[j, 3]) < 5e-6 and abs(Ac[j, 3, i] - B_req[j, 0]) < 1e-6 and abs(Ac[j, 4, i] - Ac[N-j -1, 4, i]) < 0.03e-6 and abs(Ac[j, 4, i]) < 1.7e-5 ):# and
                    # abs((abs(Ac[j, 6, i] - Ac[j + 1, 6, i]) - abs(Ac[j + 1, 6, i] - Ac[j + 2, 6, i]))) < 1e-7 and
                    # abs((abs(Ac[j, 4, i] - Ac[j + 1, 4, i]) - abs(Ac[j + 1, 4, i] - Ac[j + 2, 4, i]))) < 1e-7):
                true += 1
        if (true == N):
            t_convergence = i
            break
        else:
            continue

    return t, x, X, Ac, t_convergence


def Regrouping(N, fnc, t0, tf, x_, X_, x0, n, k, satX_num, a_req):
    t = np.linspace(t0, tf, n)
    h = t[1] - t[0]

    x = np.zeros((N, 6, n))
    X = np.zeros((N, 6, n))

    x[:, :, 0] = x_[:, :]
    X[:, :, 0] = X_[:, :]

    a0 = norm(x0[:3, 0])
    Ac = np.zeros((N, 10, n))
    B = np.zeros((N, 4))
    B_req = np.zeros((N, 4))
    k[satX_num, :] = np.zeros((1, 7))
    for i in range(1, n):
        omega_i = get_omega(x0[:, i - 1])
        for j in range(N):
            q = q_current(x0[:, i - 1], x[j, :, i-1])  # [phi, theta, rho, phi_dot, theta_dot, rho_dot]
            B[j, :] = q2B(q, omega_i, a0)
            B[satX_num, 0] = 0

        B4_req = calculate_B4_req(N, a0, x0[:, i-1], B, satX_num, a_req, B[satX_num, 3])
        B_req[:, 3] = B4_req
        B_req[satX_num, :] = B[satX_num, :]

        for j in range(N):
            x[j, :, i], A_ = rk4_u_OneStep(fnc, h, t[i-1], x[j, :, i-1], x0[:, i-1], k[j, :], B_req[j, :])
            X[j, :, i] = IF2LVLH(x[j, :, i], x0[:, i])
            Ac[j, :, i] = A_

        true = 0
        t_convergence = n-1
        for j in range(N-2):
            if (abs(Ac[j, 6, i] - B4_req[j]) < 1e-7 and abs((abs(Ac[j, 6, i] - Ac[j+1, 6, i]) - abs(Ac[j+1, 6, i] - Ac[j+2, 6, i]))) < 1e-10 and Ac[j, 4, i] < 1.5e-5):
                true += 1
        if (true == N):
            t_convergence = i
            break
        else:
            continue
    return t, x, X, Ac, t_convergence


def cluster_launch(T, N, n, x0, C, w):
    x = np.zeros((N, 6, n))
    for j in range(N):
        x_rand = np.zeros(6)
        x_rand[3:] = np.random.uniform(0, 0.1, 3)
        x[j, :, int(T*j / tau) + 1] = LVLH2IF(C2X(C[j, :], w) + x_rand, x0[:, int(T*j / tau) + 1]) # переделать рандом
        #print(x[j, :, int(T*j / tau)])
    return x


# параметры
N = 5
T = 20
t0 = 0
tf =60*60*10
t0_2 = tf
tf_2 = tf + 60*60*5
r_0 = (6771)*(10**3)
mu = 398600.4415 * (10**9)
v_1 = np.sqrt(mu/r_0)
x0_0 = np.array([0.0, r_0, 0.0, -v_1, 0.0, 0.0])
omega = get_omega(x0_0)
a0 = norm(x0_0[:3])
tau = 1
n1 = int((tf - t0) / tau)
n2 = int((tf_2 - t0_2) / tau)
i = pi/180 * 0.0
u = 0
o = 0
satX_num = int(random.choice(np.linspace(0, N-1, N)))
print(satX_num + 1)
a_req =50
u_max = 0.002


A = np.array([[cos(o)*cos(u) - sin(o)*cos(i)*sin(u), -sin(u)*cos(o) - sin(o)*cos(i)*cos(u), sin(i)*sin(o)],
              [sin(o)*cos(u) + cos(o)*cos(i)*sin(u), -sin(o)*sin(u) + cos(o)*cos(i)*cos(u), -cos(o)*sin(i)],
              [sin(i)*sin(u)                       , sin(i)*cos(u)                        , cos(i)]])

x0_0[:3] = np.dot(A, x0_0[:3])
x0_0[3:] = np.dot(A, x0_0[3:])
t, x0 = rk4(RightPart_u, t0, tf, x0_0, n1, 0)
#print(x0[:, :4])
t2, x0_2 = rk4(RightPart_u, t0_2, tf_2, x0[:, -1], n2, 0)



#несколько спутников
w = np.sqrt(mu / (r_0**3))
C1 = np.array([10.0, 10.0, 50.0, 0.0, 100.0, 10.0]) #в оск для спутника
C2 = np.array([10.0, 10.0, 50.0, 0.0, 100.0, 10.0])
C3 = np.array([10.0, 10.0, 50.0, 0.0, 100.0, 10.0])
C4 = np.array([10.0, 10.0, 50.0, 0.0, 100.0, 10.0])
C5 = np.array([10.0, 10.0, 50.0, 0.0, 100.0, 10.0])
C = np.array([C1, C2, C3, C4, C5])

X0_arr = np.zeros((N, 6))
for i in range(N):
    X0_arr[i, :] = Set_X0(C[i, :], w)

x0_arr = np.zeros((N, 6))
for i in range(N):
    x0_arr[i, :] = Set_x0(X0_arr[i, :], x0_0)


#коэффициенты управления
# k1 = np.array([1e-3, 1e-1, 1e-5, 1e3, 1e6, 1e2, 1e1])
# k2 = np.array([1e-3, 1e-1, 1e-5, 1e3, 1e6, 1e2, 1e1])
# k3 = np.array([1e-3, 1e-1, 1e-5, 1e3, 1e6, 1e2, 1e1])
# k4 = np.array([1e-3, 1e-1, 1e-5, 1e3, 1e6, 1e2, 1e1])
# k5 = np.array([1e-3, 1e-1, 1e-5, 1e3, 1e6, 1e2, 1e1])
# k_1 = np.array([k1, k2, k3, k4, k5])                   #НЕ МЕНЯТЬ!!!!!! - ВРЕМЯ СХОДИМОСТИ НЕ МЕНЬШЕ 9 ЧАСОВ без J2, tau=0.5
k1 = np.array([1e-3, 1e-1, 1e-5, 1e3, 1e5, 1e2, 1e1])
k2 = np.array([1e-3, 1e-1, 1e-5, 1e3, 1e5, 1e2, 1e1])
k3 = np.array([1e-3, 1e-1, 1e-5, 1e3, 1e5, 1e2, 1e1])
k4 = np.array([1e-3, 1e-1, 1e-5, 1e3, 1e5, 1e2, 1e1])
k5 = np.array([1e-3, 1e-1, 1e-5, 1e3, 1e5, 1e2, 1e1])
k_1 = np.array([k1, k2, k3, k4, k5])                   #НЕ МЕНЯТЬ!!!!!! - ВРЕМЯ СХОДИМОСТИ НЕ МЕНЬШЕ 9 ЧАСОВ, с J2, tau=1

k12 = np.array([1e-3, 1e-1, 1e-3, 1e6, 1e1, 1e2, 1e-1])
k22 = np.array([1e-3, 1e-1, 1e-3, 1e6, 1e1, 1e2, 1e-1])
k32 = np.array([1e-3, 1e-1, 1e-3, 1e6, 1e1, 1e2, 1e-1])
k42 = np.array([1e-3, 1e-1, 1e-3, 1e6, 1e1, 1e2, 1e-1])
k52 = np.array([1e-3, 1e-1, 1e-3, 1e6, 1e1, 1e2, 1e-1])
k_2 = np.array([k12, k22, k32, k42, k52])              # НЕ МЕНЯТЬ, СХОДИТСЯ МЕНЬШЕ ЧЕМ ЗА 10  ЧАСОВ, без J2, tau=1
# k12 = np.array([1e-3, 1e-1, 1e-3, 1e6, 1e-1, 1e2, 1e-1])
# k22 = np.array([1e-3, 1e-1, 1e-3, 1e6, 1e-1, 1e2, 1e-1])
# k32 = np.array([1e-3, 1e-1, 1e-3, 1e6, 1e-1, 1e2, 1e-1])
# k42 = np.array([1e-3, 1e-1, 1e-3, 1e6, 1e-1, 1e2, 1e-1])
# k52 = np.array([1e-3, 1e-1, 1e-3, 1e6, 1e-1, 1e2, 1e-1])
# k_2 = np.array([k12, k22, k32, k42, k52])


C1_req = np.array([0, 0, 0, 0, 0, 0])
C2_req = np.array([0, 0, 0, 50, 0, 0])
C3_req = np.array([0, 0, 0, 100, 0, 0])
C4_req = np.array([0, 0, 0, 150, 0, 0])
C5_req = np.array([0, 0, 0, 200, 0, 0])


B_req1 = C2B(C1_req, w, omega, a0, x0_0)
B_req2 = C2B(C2_req, w, omega, a0, x0_0)
B_req3 = C2B(C3_req, w, omega, a0, x0_0)
B_req4 = C2B(C4_req, w, omega, a0, x0_0)
B_req5 = C2B(C5_req, w, omega, a0, x0_0)
B_req = np.array([B_req1, B_req2, B_req3, B_req4, B_req5])

# xl = cluster_launch(T, N, n1, x0, C, w)
# t1, x1, X1, Ac1, t_convergence1 = Global_rk4(N, RightPart_u, t0, tf, T, x0_arr, X0_arr, x0, xl, n1, k_1, B_req)
# print(t_convergence*tau)
x1_ = np.array([[ 6.36776994e+05,  6.74094358e+06, -1.01076041e+00, -7.63864615e+03,
   7.21577481e+02,  2.29376301e-04],
 [ 6.36677348e+05,  6.74095299e+06, -9.74210341e-01, -7.63865682e+03,
   7.21464555e+02,  3.47981124e-04],
 [ 6.36577707e+05,  6.74096240e+06, -1.02703180e+00, -7.63866745e+03,
   7.21351950e+02, -4.47145193e-05],
 [ 6.36478062e+05,  6.74097181e+06, -1.02954912e+00, -7.63867813e+03,
   7.21238867e+02,  1.30999766e-05],
 [ 6.36378416e+05,  6.74098123e+06, -1.02938913e+00, -7.63868879e+03,
   7.21125875e+02,  1.63266867e-04]])
X1_ = np.array([[-6.01951251e+00, -1.01076041e+00, -4.69454305e+01,  1.06573955e-01,
   2.29376301e-04, -1.54378713e-04],
 [ 9.40698882e+01, -9.74210341e-01, -4.69434033e+01,  1.06570519e-01,
   3.47981124e-04, -1.66353808e-04],
 [ 1.94154543e+02, -1.02703180e+00, -4.69452080e+01,  1.06572394e-01,
  -4.47145193e-05,  1.39086310e-04],
 [ 2.94243292e+02, -1.02954912e+00, -4.69470152e+01,  1.06570951e-01,
   1.30999766e-05, -3.10739068e-05],
 [ 3.94332728e+02, -1.02938913e+00, -4.69455930e+01,  1.06558735e-01,
   1.63266867e-04, -1.08284575e-04]])
Ac1_ = np.array([[ 2.48052514e-11,  3.36848132e-11,  1.40224319e-04,  2.36812346e-08,
   6.98069324e-06,  1.52243041e-07, -9.25573313e-07, -3.36423488e-05,
  -3.53792250e-04,  9.69732595e-15],
 [ 2.47907219e-11,  5.12084432e-11,  1.22999243e-04,  2.40249807e-08,
   6.98097766e-06,  1.50859816e-07,  1.38610904e-05, -3.29903807e-05,
  -3.47000687e-04,  1.28386241e-14],
 [ 2.47743827e-11, -6.79875984e-12,  8.14453243e-05,  2.43675127e-08,
   6.98162391e-06,  1.51793936e-07,  2.86533942e-05,  3.88627142e-09,
  -2.05662190e-09,  8.26054237e-10],
 [ 2.47283140e-11,  1.73951216e-12, -8.71431797e-05,  2.47125320e-08,
   6.98202401e-06,  1.52063382e-07,  4.34794169e-05,  3.89225620e-09,
  -2.05713624e-09, -2.12103252e-10],
 [ 2.47739924e-11,  2.39176052e-11,  1.76099915e-04,  2.50412395e-08,
   6.98176209e-06,  1.53512160e-07,  5.81929781e-05, -3.20606612e-05,
  -3.37358168e-04,  8.67876890e-15]])
#t2, x2, X2, Ac2, t_convergence2 = Regrouping(N, RightPart_u, t0_2, tf_2, x1[:, :, -1], X1[:, :, -1], x0_2, n2, k_2, satX_num, a_req)

t2, x2, X2, Ac2, t_convergence = Regrouping(N, RightPart_u, t0_2, tf_2, x1_, X1_, x0_2, n2, k_2, satX_num, a_req)
# t_full = np.hstack([t1[1:t_convergence1], t2[1:t_convergence2]])
# X_full = np.concatenate([X1[:, :, 1:t_convergence1], X2[:, :, 1:t_convergence2]], axis=2)
# Ac_full = np.concatenate([Ac1[:, :, 1:t_convergence1], Ac2[:, :, 1:t_convergence2]], axis=2)
# max = 0

# for i in range(n1+n2):
#     if (norm(Ac_full[:, 7:, i]) > max):
#         max = norm(Ac_full[:, 7:, i])
# print(max)
# print(t_full[-1])

#show_B(t_full, Ac_full, N, t_full[-1])
show_B(t2[:t_convergence], Ac2[:, :, :t_convergence], N, t2[t_convergence])
# show_trajectories_LVLH(t[:t_convergence], X1[:, :, :t_convergence], N)
show_trajectories_LVLH(t2, X2[:, :, :t_convergence], N)
#show_trajectories_LVLH(t_full, X_full, N)
# show_Lyapunov(t2, Ac2, N)
#show_control_(t1[:t_convergence], Ac1[:, :, :t_convergence], N, t1[t_convergence], u_max)
#show_control_(t_full, Ac_full, N, t_full, u_max)

plt.show()




# num_of_U = 3
# num_of_exp = 5
# U = np.zeros(num_of_U)
# T_c = np.zeros((num_of_U, num_of_exp))
# for l in range(num_of_U):
#     U[l] = u_max
#     for k in range(num_of_exp):
#         print('исследование:', k+1, ' для u', l+1, '=', U[l])
#         xl = cluster_launch(T, N, n1, x0, C, w)
#         t1, x1, X1, Ac1, t_convergence = Global_rk4(N, RightPart_u, t0, tf, T, x0_arr, X0_arr, x0, xl, n1, k_1, B_req)
#        # t2, x2, X2, Ac2, t_convergence = Regrouping(N, RightPart_u, t0_2, tf_2, x1[:, :, -1], X1[:, :, -1], x0_2, n2, k_2, satX_num, a_req)
#         T_c[l, k] = t_convergence*tau
#     u_max -= 0.0001
#
# np.save('U', U)
# np.save('Convergence_Time', T_c)
#
# T_c = np.load('Convergence_Time.npy')
# U = np.load('U.npy')
# print(T_c, U)
#
# T_pd = pd.DataFrame(data=T_c.T, columns=U)
# T_pd_melted = pd.melt(T_pd)
#
# sns.boxplot(x='variable', y='value', data=T_pd_melted)
# plt.xlabel('Максимальное управляющее ускорение')
# plt.ylabel('Время сходмости')
# plt.savefig('boxplot.png')
# plt.show()



