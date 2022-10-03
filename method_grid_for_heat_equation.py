import math
import numpy as np
import matplotlib.pyplot as plt


def vector_x(h, N):
    "the function calculates the values of the vector x"
    x = []
    for i in range(N + 1):
        x.append(np.around(i * h, 2))
    return x


def vector_t(l, M):
    "the function calculates the values of the vector t"
    t = []
    for j in range(M + 1):
        t.append(np.around(j * l, 4))
    return t


def function_u(h, l, N, M):
    "the function calculates the value of the grid nodes"
    u = []
    for i in range(M + 1):
        u.append([float(0)] * (N + 1))

    for i in range(N + 1):
        u[M][i] = np.around((1 * vector_x(h, N)[i] ** 2 + 1) * math.sin(math.pi * vector_x(h, N)[i]), 5)

    for i in range(M - 1, -1, -1):
        alpha = [0]
        beta = [0]
        for j in range(0, N):
            s = h ** 2 / l
            alpha.append(np.around(1 / (2 + s - alpha[j]), 5))
            beta.append(np.around((-s * u[i + 1][j] - beta[j]) / (alpha[j] - 2 - s), 5))
        for k in range(N, 1, -1):
            u[i][k - 1] = np.around((alpha[k] * u[i][k] + beta[k]), 5)

    return np.array(u)


def draw_graphics():
    # plt.subplot(2, 2, 1)
    # plt.title(f'При h={h[0]}, l={l[0]}')
    # plt.xlim([0, 1])
    # #plt.xlabel('x')
    # plt.ylim([0, 1.5])
    # plt.ylabel('U(x,t)')
    # plt.grid(color='green')
    # for i in range(M[0] + 1):
    #     plt.plot(vector_x(h[0], N[0]), function_u(h[0], l[0], N[0], M[0])[M[0] - i], '--',
    #              label=f'U{i}x (t={vector_t(l[0], M[0])[i]})')
    # plt.legend(bbox_to_anchor=(-0.07, 1.2))
    # plt.subplot(2, 2, 2)
    # plt.title(f'При h={h[1]}, l={l[1]}')
    # plt.xlim([0, 1])
    # plt.xlabel('x')
    # plt.ylim([0, 1.5])
    # plt.ylabel('U(x,t)')
    # plt.grid(color='green')
    # for i in range(M[0] + 1):
    #     plt.plot(vector_x(h[0], N[0]), function_u(h[1], l[1], N[1], M[1])[::4, ::2][M[0]-i], '-',
    #              label=f'U{i}x (t={vector_t(l[0], M[0])[i]})')
    # plt.legend(bbox_to_anchor=(1.28, 1.2))
    # plt.subplot(2, 2, 3)
    # plt.title(f'При h={h[2]}, l={l[2]}')
    # plt.xlim([0, 1])
    # plt.xlabel('x')
    # plt.ylim([0, 1.5])
    # plt.ylabel('U(x,t)')
    # plt.grid(color='green')
    # for i in range(M[0] + 1):
    #     plt.plot(vector_x(h[0], N[0]), function_u(h[2], l[2], N[2], M[2])[::20, ::4][M[0] - i], '-.',
    #              label=f'U{i}x (t={vector_t(l[0], M[0])[i]})')
    # plt.legend(bbox_to_anchor=(1.28, 1))

    fig = plt.figure()
    count = 2
    xv, yv = np.meshgrid(vector_x(h[count], N[count]), vector_t(l[count], M[count]))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xv, yv, function_u(h[count], l[count], N[count], M[count]), cmap='inferno')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    plt.show()


CONST_X = 1
CONST_T = 0.2

h = [0.2, 0.1, 0.05]
l = [0.04, 0.01, 0.002]

N = []
for i in h:
    N.append(int(CONST_X / i))

M = []
for j in l:
    M.append(int(CONST_T / j))

for k in range(3):
    match k:
        case 0:
            print('x\t', end='')
            for i in range(N[0] + 1):
                print(f"|{vector_x(h[0], N[0])[i]:^7}", end='')
            print('\nt\t', f'|{"U0k":^7}|{"U1k":^7}|{"U2k":^7}|{"U3k":^7}|{"U4k":^7}|{"U5k":^7}', sep='')
            for i in range(M[0] + 1):
                print(f"{vector_t(l[0], M[0])[-1 - i]:<4}", end='')
                for j in range(N[0] + 1):
                    print(f"|{function_u(h[k], l[k], N[k], M[k])[i][j]:^7}", end='')
                print()
            print()
        case 1:
            print('x\t', end='')
            for i in range(N[0] + 1):
                print(f"|{vector_x(h[0], N[0])[i]:^7}", end='')
            print('\nt\t', f'|{"U0k":^7}|{"U1k":^7}|{"U2k":^7}|{"U3k":^7}|{"U4k":^7}|{"U5k":^7}', sep='')
            for i in range(M[0] + 1):
                print(f"{vector_t(l[0], M[0])[-1 - i]:<4}", end='')
                for j in range(N[0] + 1):
                    print(f"|{function_u(h[k], l[k], N[k], M[k])[::4, ::2][i][j]:^7}", end='')
                print()
            print()
        case 2:
            print('x\t', end='')
            for i in range(N[0] + 1):
                print(f"|{vector_x(h[0], N[0])[i]:^7}", end='')
            print('\nt\t', f'|{"U0k":^7}|{"U1k":^7}|{"U2k":^7}|{"U3k":^7}|{"U4k":^7}|{"U5k":^7}', sep='')
            for i in range(M[0] + 1):
                print(f"{vector_t(l[0], M[0])[-1 - i]:<4}", end='')
                for j in range(N[0] + 1):
                    print(f"|{function_u(h[k], l[k], N[k], M[k])[::20, ::4][i][j]:^7}", end='')
                print()

draw_graphics()
