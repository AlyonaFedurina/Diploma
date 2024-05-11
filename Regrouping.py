import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import cos
from numpy import sin
from numpy import pi


def show_B(t, Ac, N):
    plt.figure()
    for j in range(4):
        plt.subplot(2, 2, j+1)
        for i in range(N):
            plt.plot(t[1:], Ac[i, 3+j, 1:], linewidth=0.5)
            plt.xlabel('t')
        B = 'B' + str(j+1)
        plt.ylabel(B)
        plt.legend(['1', '2', '3', '4', '5'])
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
        plt.plot(t[1:], Ac[i, 0, 1:], linewidth=0.5)
    plt.title('Функция Ляпунова')
    plt.xlabel('t')
    plt.ylabel('V')
    plt.legend(['спутник 1', 'спутник 2', 'спутник 3', 'спутник 4', 'спутник 5'])
    plt.grid(True, linestyle=":", alpha=0.5)


def show_control_(t, Ac, N):
    plt.figure()
    for j in range(3):
        plt.subplot(3, 1, j+1)
        for i in range(N):
            plt.plot(t[1:], Ac[i, 7+j, 1:], linewidth=0.5)
        plt.xlabel('t')
        u = 'u' + chr(120+j) + ' м/с^2'
        plt.ylabel(u)
    plt.legend(['1', '2', '3', '4', '5'], loc='right')
    plt.grid(True, linestyle=":", alpha=0.5)


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
    x0 = BF2IF(X0, x0_0)
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


def IF2BF(x2, x1):
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


def BF2IF(X_2, x_1):
    omega = get_omega(x_1)
    G = get_G(x_1)
    x_2 = np.zeros(6)
    x_2[:3] = x_1[:3] + np.dot(G, X_2[:3])
    x_2[3:] = x_1[3:] + np.dot(G, X_2[3:]) + np.cross(omega, np.dot(G, X_2[:3]))
    return x_2


def a_BF2IF(a, G):
    A = np.dot(G, a)
    return A


def C2x(C, w):
    x = np.array([C[3] + 2 * C[1], C[5], 2 * C[0] + C[2], -3 * w * C[0] - 2 * w * C[2], w * C[4], w * C[1]])
    return x


def q_current(x0, x2):
    X = IF2BF(x2, x0)
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
    X = C2x(C, w)
    x = BF2IF(X, x0)
    q = q_current(x0, x)
    B = q2B(q, omega, a0)
    return B


def calculate_a_u(k, B_req, omega, q, C_q, G):
    ka = k[0]
    kb = k[1]
    k_phi = k[2]
    k_theta = k[3]
    k_rho = k[4]
    k2 = k[5]
    k4 = k[6]
    B2_req = 0#B_req[1]
    B3_req = 0#B_req[2]
    B4_req = B_req[3]
    a0 = norm(x0[:3])

    phi_DoubleDot = -(C_q[0] / omega + k2*(C_q[1] - B2_req) * 2 * (2*C_q[0] - q[2]/a0) / (C_q[1] * omega)) * k_phi * C_q[1]**2
    theta_DoubleDot = -q[4] * (C_q[2] - B3_req) / (omega ** 2) * k_theta * C_q[2]
    rho_DoubleDot = -(k2*(C_q[1] - B2_req) * q[5] / (C_q[1]*omega ** 2 * a0 ** 2) - k4*2 * (C_q[3] - B4_req) / (omega * a0)) * k_rho * C_q[1]**2


    if (3*omega * C_q[0] * (C_q[3] - B4_req)*C_q[1]**2 * C_q[2]**2 * k4 <= -phi_DoubleDot**2 * C_q[2]**2/( k_phi) - theta_DoubleDot**2 *C_q[1]**2 / (k_theta) - rho_DoubleDot**2 * C_q[2]**2/ (k_rho)):
        phi_DoubleDot = -ka * omega * C_q[0]
        theta_DoubleDot = 0
        rho_DoubleDot = kb * omega * a0 * 0.5 * (C_q[3] - B4_req) - 3 * (omega ** 2) * a0 * C_q[0] * 0.5
        #print("blink")
        V = (C_q[0]**2 + (C_q[3] - B4_req)**2) * 0.5

    else:
        V = (C_q[0] ** 2 + (C_q[3] - B4_req) ** 2 + (C_q[1] - B2_req) ** 2 + (C_q[2] - B3_req) ** 2) * 0.5

    a_u = np.zeros(3)
    a_u[0] = phi_DoubleDot * (q[2] + a0) * cos(q[1]) + 2*q[3]*q[5]*cos(q[1]) - 2*q[4]*q[3]*(q[2] + a0)*sin(q[1])
    a_u[1] = (q[2] + a0) * (theta_DoubleDot - q[3]**2 * cos(q[1]) * sin(q[1])) + 2*q[5]*q[4]
    a_u[2] = rho_DoubleDot - (q[2] + a0) * (q[3]**2 * (cos(q[1]))**2 + q[4]**2)
    A_u = a_BF2IF(a_u, G)
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


def rk4_u_OneStep(fnc, h, a0, t_im, x2_im, x1_im, k, B_req): #значения на каждом шаге
    G_i = get_G(x1_im)
    omega_i = get_omega(x1_im)
    q = q_current(x1_im, x2_im)  # [phi, theta, rho, phi_dot, theta_dot, rho_dot]
    B = q2B(q, omega_i, a0)

    a_u, V = calculate_a_u(k, B_req, norm(omega_i), q, B, G_i)  # уже в ИСК л
    A = np.array([V, q[4], q[5], B[0], B[1], B[2], B[3], a_u[0], a_u[1], a_u[2]])

    k1 = fnc(t_im, x2_im, a_u)
    k2 = fnc(t_im + h * 0.5, x2_im + h * 0.5 * k1, a_u)
    k3 = fnc(t_im + h * 0.5, x2_im + h * 0.5 * k2, a_u)
    k4 = fnc(t_im + h, x2_im + h * k3, a_u)
    x2_i = x2_im + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * h
    return x2_i, A

def Global_rk4(N, fnc, t0, tf, x0_arr, X0_arr, x0, n, k, B_req, a_req):
    t = np.linspace(t0, tf, n)
    h = t[1] - t[0]

    x = np.zeros((N, 6, n))
    X = np.zeros((N, 6, n))
    for i in range(N):
        x[i, :, 0] = x0_arr[i, :]
        X[i, :, 0] = X0_arr[i, :]

    a0 = norm(x0[:3, 0])
    Ac = np.zeros((N, 10, n))
    for i in range(1, n):
        for j in range(N):
            x[j, :, i], A = rk4_u_OneStep(fnc, h, a0, t[i-1], x[j, :, i-1], x0[:, i-1], k[j, :], B_req[j, :])
            X[j, :, i] = IF2BF(x[j, :, i], x0[:, i])
            Ac[j, :, i] = A

    return t, x, X, Ac




# параметры
t0 = 0.0                            #start time
tf =60*60*24                       #end time
r_0 = (6771)*(10**3)
mu = 398600.4415 * (10**9)
v_1 = np.sqrt(mu/r_0)
x0_0 = np.array([0.0, r_0, 0.0, -v_1, 0.0, 0.0])
omega = get_omega(x0_0)
a0 = norm(x0_0[:3])
n = 10000
i = pi/180 * 0.0
u = 0
o = 0
N = 5
a_req = 50

A = np.array([[cos(o)*cos(u) - sin(o)*cos(i)*sin(u), -sin(u)*cos(o) - sin(o)*cos(i)*cos(u), sin(i)*sin(o)],
              [sin(o)*cos(u) + cos(o)*cos(i)*sin(u), -sin(o)*sin(u) + cos(o)*cos(i)*cos(u), -cos(o)*sin(i)],
              [sin(i)*sin(u)                       , sin(i)*cos(u)                        , cos(i)]])

x0_0[:3] = np.dot(A, x0_0[:3])
x0_0[3:] = np.dot(A, x0_0[3:])
t, x0 = rk4(RightPart_u, t0, tf, x0_0, n, 0)



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

x0_arr = np.zeros((n, 6))
for i in range(N):
    x0_arr[i, :] = Set_x0(X0_arr[i, :], x0_0)



#коэффициенты управления
k1 = np.array([1e-5, 1e-3, 1e0, 1e1, 1e20, 1, 1e-3])
k2 = np.array([1e-5, 1e-3, 1e0, 1e0, 1e20, 1, 1e-3])
k3 = np.array([1e-5, 1e-3, 1e0, 1e0, 1e20, 1, 1e-3])
k4 = np.array([1e-5, 1e-3, 1e0, 1e0, 1e20, 1, 1e-3])
k5 = np.array([1e-5, 1e-3, 1e0, 1e0, 1e20, 1, 1e-3])
k = np.array([k1, k2, k3, k4, k5])


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


t, x, X, Ac = Global_rk4(N, RightPart_u, t0, tf, x0_arr, X0_arr, x0, n, k, B_req)

show_B(t, Ac, N)
show_trajectories_LVLH(t, X, N)
show_Lyapunov(t, Ac, N)
show_control_(t, Ac, N)

plt.show()




