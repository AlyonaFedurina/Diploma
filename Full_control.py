import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import cos
from numpy import sin
from numpy import pi


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


def q_current(x1, x2):
    X = IF2BF(x2, x1)
    rho = norm(x2[:3]) - norm(x1[:3])
    a0 = norm(x1[:3])
    phi = np.arctan2(X[0], (X[2] + a0))
    theta = np.arcsin(X[1] / norm(x2[:3]))
    phi_dot = (X[3] * cos(phi) - X[5] * sin(phi)) / (cos(theta) * (rho + a0))
    theta_dot = (X[4] * cos(theta) - X[3] * sin(phi) * sin(theta) - X[5] * cos(phi) * sin(theta)) / (rho + a0)
    rho_dot = X[3] * sin(phi) / cos(theta) + X[5] * cos(phi) / cos(theta) + (rho + a0) * theta_dot * np.tan(theta)
    q = np.array([phi, theta, rho, phi_dot, theta_dot, rho_dot])
    return q


def calculate_a_u(k, B2_req, B3_req, B4_req, omega, q, C_q, G):
    ka = k[0]
    kb = k[1]
    k_phi = k[2]
    k_theta = k[3]
    k_rho = k[4]
    k2 = k[5]
    k4 = k[6]
    a0 = norm(x1[:3])

    phi_DoubleDot = -(C_q[0] / omega + k2*(C_q[1] - B2_req) * 2 * (2*C_q[0] - q[2]/a0) / (C_q[1] * omega)) * k_phi * C_q[1]**2
    theta_DoubleDot = -q[4] * (C_q[2] - B3_req) / (omega ** 2) * k_theta * C_q[2]
    rho_DoubleDot = -(k2*(C_q[1] - B2_req) * q[5] / (C_q[1]*omega ** 2 * a0 ** 2) - k4*2 * (C_q[3] - B4_req) / (omega * a0)) * k_rho * C_q[1]**2


    if (3*omega * C_q[0] * (C_q[3] - B4_req)*C_q[1]**2 * C_q[2]**2 * k4 <= -phi_DoubleDot**2 * C_q[2]**2/( k_phi) - theta_DoubleDot**2 *C_q[1]**2 / (k_theta) - rho_DoubleDot**2 * C_q[2]**2/ (k_rho)):
        phi_DoubleDot = -ka * omega * C_q[0]
        theta_DoubleDot = 0
        rho_DoubleDot = kb * omega * a0 * 0.5 * (C_q[3] - B4_req) - 3 * (omega ** 2) * a0 * C_q[0] * 0.5
        print("blink")
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


def rk4_u(fnc, t0, tf, x0, X0, x1, n):
    t = np.linspace(t0, tf, n)
    h = t[1] - t[0]

    x2 = np.zeros((x0.shape[0], n))
    x2[:, 0] = x0
    X2 = np.zeros(np.shape(x2))
    X2[:, 0] = X0
    a0 = norm(x0[:3])
    B = np.zeros((7, n))
    for i in range(1, n):
        G_i = get_G(x1[:, i-1])
        omega_i = get_omega(x1[:, i-1])
        q = q_current(x1[:, i - 1], x2[:, i - 1])  #[phi, theta, rho, phi_dot, theta_dot, rho_dot]

        if (np.isnan(q[1])):
            print(i, '***')

        B1 = q[3] / norm(omega_i) + 2 * q[2] / a0
        B2 = np.sqrt((q[5] / (norm(omega_i) * a0)) ** 2 + (-2 * q[3] / norm(omega_i) - 3 * q[2] / a0) ** 2)
        B3 = np.sqrt((q[4] / norm(omega_i)) ** 2 + q[1] ** 2)
        B4 = -2 * q[5] / (norm(omega_i) * a0) + q[0]

        C_q = np.array([B1, B2, B3, B4])
        k = np.array([1e-5, 1e-3, 1e0, 1e0, 1e20, 1, 1e-3])
        a_u, V = calculate_a_u(k, 0 ,0, 2e-5, norm(omega_i), q, C_q, G_i) # уже в ИСК л

        B[:7, i] = np.array([V, q[4], q[5], B1, B2, B3, B4])

        k1 = fnc(t[i-1], x2[:, i-1], a_u)
        k2 = fnc(t[i-1] + h * 0.5, x2[:, i-1] + h * 0.5 * k1, a_u)
        k3 = fnc(t[i-1] + h * 0.5, x2[:, i-1] + h * 0.5 * k2, a_u)
        k4 = fnc(t[i-1] + h, x2[:, i-1] + h * k3, a_u)
        x2[:, i] = x2[:, i - 1] + (1/6) * (k1 + 2*k2 + 2*k3 + k4) * h
        X2[:, i] = IF2BF(x2[:, i], x1[:, i])


    return t, x2, X2, B


# параметры
t0 = 0.0                            #start time
tf =60*1000                         #end time
r_0 = (6778)*(10**3)
mu = 398600.4415 * (10**9)
v_1 = np.sqrt(mu/r_0)
x0_1 = np.array([0.0, r_0, 0.0, -v_1, 0.0, 0.0])
n = 20000
i = pi/180 * 0.0
u = 0
o = 0


# матрица перехода из сск в иск
A = np.array([[cos(o)*cos(u) - sin(o)*cos(i)*sin(u), -sin(u)*cos(o) - sin(o)*cos(i)*cos(u), sin(i)*sin(o)],
              [sin(o)*cos(u) + cos(o)*cos(i)*sin(u), -sin(o)*sin(u) + cos(o)*cos(i)*cos(u), -cos(o)*sin(i)],
              [sin(i)*sin(u)                       , sin(i)*cos(u)                        , cos(i)]])

x0_1[:3] = np.dot(A, x0_1[:3])
x0_1[3:] = np.dot(A, x0_1[3:])
t, x1 = rk4(RightPart_u, t0, tf, x0_1, n, 0)


# 2_ой спутник
w = np.sqrt(mu / (r_0**3))
c = np.array([0.0, 10.0, 50.0, 0.0, 100.0, 10.0]) #в оск для 2-го спутника
X0_2 = np.array([c[3] + 2*c[1], c[5], 2*c[0] + c[2], -3*w*c[0] - 2*w*c[2], w*c[4], w*c[1]]) #2-ой спутник в оск


#2-ой спутник в иск + оск с управлением
x0_2 = BF2IF(X0_2, x0_1)
t, x2, X2, B = rk4_u(RightPart_u, t0, tf, x0_2, X0_2, x1, n) #больше 5 нулей не ставить


# построение графиков
plt.figure()
plt.plot(t[1:], B[3, 1:], 'm')
plt.plot(t[1:], B[4, 1:], 'y')
plt.plot(t[1:], B[5, 1:], 'b')
plt.plot(t[1:], B[6, 1:], 'g')
plt.legend(['B1', 'B2', 'B3', 'B4'])
plt.grid(True, linestyle=":", alpha=0.5)

plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(t[1:], B[0, 1:])
# plt.xlabel('t')
# plt.ylabel('phi')
# plt.grid(True, linestyle=":", alpha=0.5)
# plt.subplot(3, 1, 2)
# plt.plot(t[1:], B[1, 1:])
# plt.subplot(3, 1, 3)
plt.plot(t[1:], B[0, 1:])
plt.title('Функция Ляпунова')
plt.xlabel('t')
plt.ylabel('V')
plt.grid(True, linestyle=":", alpha=0.5)

# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.plot(x2[0, :], x2[1, :], x2[2, :], label='parametric curve')
# #ax1.plot(x1[0, :], x1[1, :], x1[2, :], label='parametric curve')
# #ax1.plot(x2[0, :10], x2[1, :10], x2[2, :10], 'g', lw=3)
# ax1.set_xlabel('x, м', fontsize=15)
# ax1.set_ylabel('y, м', fontsize=15)
# ax1.set_zlabel('z, м', fontsize=15)
# lim1 = abs(x2).max()
# ax1.set_xlim(-lim1, lim1)
# ax1.set_ylim(-lim1, lim1)
# ax1.set_zlim(-lim1, lim1)
# ax1.set_title('Траектория 2-го спутника в ИСК')

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot(X2[0, :], X2[1, :], X2[2, :], label='parametric curve')
ax2.plot(X2[0, :10], X2[1, :10], X2[2, :10], 'g', lw=3)
ax2.set_xlabel('x', fontsize=15)
ax2.set_ylabel('y', fontsize=15)
ax2.set_zlabel('z', fontsize=15)
lim2 = abs(X2).max()
ax2.set_xlim(-lim2, lim2)
ax2.set_ylim(-lim2, lim2)
ax2.set_zlim(-lim2, lim2)
ax2.set_title('Траектория 2-го спутника в ОСК')
plt.show()