from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import math


def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

clear_all()
np.random.seed(11)

'''Functions'''


def penalty_to_mvno(beta_local, r_m_local, R_local, q_m_old, b_local):
    # Compute penalty value for each MVNO m
    # input:
    # beta: virtual price of InP
    # r_m: allocated RBs to MVNO m
    # R: total number of RBs of InP
    # q_m_old: penalty parameter at previous time slot.
    # b: bidding value vector getting from MVNO
    q_m_new = 1 / beta_local * (1 - r_m_local / R_local ) * q_m_old / (R_local - r_m_local) * np.sum(b_local)
    return q_m_new

    # # testing:
    # beta = 5;
    # r = 4;
    # R = 20;
    # b = np.array([3,4,5])
    # q_old = 3.4;
    #
    # q_new = penalty_to_MVNO(beta,r,R,q_old,b)
    # print(q_new)


def virtual_price(b_local, R_local):
    # b: bidding value vector getting from MVNO
    # R: total number of RBs of InP
    # beta: virtual price of InP
    # function of virtual price
    # print(sum(b), R)
    beta_local = np.sum(b_local) / R_local
    # print(beta)
    return beta_local
    # b1 = np.array([1, 2,3, 4, 5])
    # R1 = 20 ;

    # beta = virtual_price(b,R)
    # print(beta)


def bidding_value(RATE_m):
    # RATE_m: data rate vector of the MVNO m
    b_m = sum(RATE_m)
    return b_m


def cal_number_of_rbs(b_m, b, R):
    # Calcuating number RBs for each MVNO
    # b: bidding value vector getting from MVNO
    # R: total number of RBs of InP
    # b_m: bidding value of MVNO m
    r_m = b_m / (np.sum(b)) * R  # Question: Do we need to round this number to an Integer value ???
    # This one will be considered after finishing ...
    return r_m

    # # Test
    # b_m = 4;
    # b = np.array([4,5,10]);
    # R = 20;

    # r_m = cal_number_of_RBs(b_m, b,R)

    # print(r_m)


'''Module design'''


def create_system(rad, num, dist_min=20):
    """This function to create random UEs inside one BS
    *Input:
    - rad: radius of the BS
    - num: number of UEs
    - dist_min: minimum distance between BS and UEs
    * Output:
    - Coor_UE, Coor_BS, distance from BS to UEs
    """
    # Testing:
    #   rad = 100
    #   num = 20

    # Update coordinator of BS
    x_BS = rad + 50
    y_BS = rad + 50
    Coor_BS = np.array([[x_BS, y_BS]])
    Coor_UE = np.zeros(shape=(num, 2))
    exit = False;

    while (exit == False):
        t = np.random.uniform(0.0, 2.0 * np.pi, num)
        r = rad * np.sqrt(np.random.uniform(0.0, 1.0, num))
        x = Coor_BS[0][0] + r * np.cos(t)
        y = Coor_BS[0][1] + r * np.sin(t)

        # Update coordinator of UEs
        Coor_UE[:, 0] = x;
        Coor_UE[:, 1] = y;

        # calculate distance UE and BS
        distance = distanceUEtoMBS(Coor_BS, Coor_UE)

        # Satisfy condition?
        if (min(distance[:, 0]) < dist_min):
            exit = False
        else:
            exit = True

    return Coor_BS, Coor_UE, distance

    # Coor_BS, Coor_UE = create_system(200, 20)


def display_network_model(Coor_BS, Coor_UE, M, S, rad):
    #   """
    #   * Input:

    #   - Coor_BS: coor of BSs
    #   - Coor_UE: coor of UEs
    #   - M: number of MVNO
    #   - S = users vector: number of UEs at each MVNO
    #   - rad: radius of MBS

    #   * Output:
    #   - Display network information

    #   """

    # Plot MBS location
    circle1 = plt.Circle((Coor_BS[0][0], Coor_BS[0][1]), 500, color='y')
    fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
    # (or if you have an existing figure)
    # fig = plt.gcf()
    # ax = fig.gca()
    ax.add_artist(circle1)
    plt.figure(1)
    plt.plot(Coor_BS[0][0], Coor_BS[0][1], "r^", ms=10, label='MBS')

    # Plot UEs of each MVNO
    for m in range(M):
        if (m == 0):
            plt.plot(Coor_UE[:S[m], 0], Coor_UE[:S[m], 1], "go", ms=5, label='MVNO-1')
        elif (m == 1):
            plt.plot(Coor_UE[sum(S[:m]):(sum(S[:m]) + S[m]), 0], Coor_UE[sum(S[:m]):(sum(S[:m]) + S[m]), 1], "b*", ms=5,
                     label='MVNO-2')
        elif (m == 2):
            plt.plot(Coor_UE[sum(S[:m]):(sum(S[:m]) + S[m]), 0], Coor_UE[sum(S[:m]):(sum(S[:m]) + S[m]), 1], "m>", ms=5,
                     label='MVNO-3')
        else:
            ValueError('M must be less than 3!')
    plt.axis([0, (2 * rad + rad / 5), 0, (2 * rad + rad / 5)])
    plt.title('Network model')
    plt.legend()
    plt.show()

def plot_RATE(rate, M, iter):
    plt.figure(2)
    t = range(iter)
    y_max = rate.max() + rate.max()/10
    for m in range(M):
        if (m == 0):
            plt.plot(t, rate[0,:], "-go", ms=5, label='MVNO-1')
        elif (m == 1):
            plt.plot(t, rate[1,:], "-b*", ms=5,
                     label='MVNO-2')
        elif (m == 2):
            plt.plot(t, rate[2,:], "-m>", ms=5,
                     label='MVNO-3')
        else:
            ValueError('M must be less than 3!')
    plt.axis([0, iter, 0, y_max])
    plt.title('Data Rate of MVNOs with Proportional Allocation')
    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Data rate (Mbps)')
    plt.show()

def plot_RA(rate, M, iter):
    plt.figure(3)
    t = range(iter)
    y_max = rate.max() + rate.max()/10
    for m in range(M):
        if (m == 0):
            plt.plot(t, rate[0,:], "-go", ms=5, label='MVNO-1')
        elif (m == 1):
            plt.plot(t, rate[1,:], "-b*", ms=5,
                     label='MVNO-2')
        elif (m == 2):
            plt.plot(t, rate[2,:], "-m>", ms=5,
                     label='MVNO-3')
        else:
            ValueError('M must be less than 3!')
    plt.axis([0, iter, 0, y_max])
    plt.title('Proportional Allocation')
    plt.legend()
    plt.grid()
    plt.ylabel('Number of Resource Blocks')
    plt.xlabel('Iteration')
    plt.show()


def distanceUEtoMBS(cBS, cUE):
    """
    Input:
    - cUE: coordinators of UEs
    - cBS: coordinators of BSs
    * Ouput:
    - array 2D
    - distance between UEs and cBSs
    """

    # Getting number of UEs from cUE
    nUE = cUE.shape[0]
    # Getting number of BSs from cBS
    nBS = cBS.shape[0]
    distance = np.zeros(shape=(nUE, nBS))
    for ue in range(nUE):
        for bs in range(nBS):
            # print('ue=',ue,'bs=',bs)
            distance[ue][bs] = math.sqrt((cUE[ue][0] - cBS[bs][0]) ** 2
                                         + (cUE[ue][1] - cBS[bs][1]) ** 2)
    return distance


def calculateNoise(self, bandwidth=20):
    k = 1.3806488 * math.pow(10, -23)
    T = 293.0
    BW = bandwidth * 1000 * 1000
    N = 10 * math.log10(k * T) + 10 * np.log10(BW)  # dB
    return N


def calculateReceivedPower(pSend, distance):
    R = distance
    lambda_val = 0.142758313333
    a = 4.0
    b = 0.0065
    c = 17.1
    d = 10.8
    s = 15.8

    ht = 40
    hr = 1.5
    f = 1.9
    gamma = a - b * ht + c / ht
    Xf = 6 * np.log10(f / 2)
    Xh = -d * np.log10(hr / 2)

    R0 = 100.0
    R0p = R0 * pow(10.0, -((Xf + Xh) / (10 * gamma)))

    if (R > R0p):
        alpha = 20 * np.log10((4 * np.pi * R0p) / lambda_val)
        PL = alpha + 10 * gamma * np.log10(R / R0) + Xf + Xh + s
    else:
        PL = 20 * np.log10((4 * np.pi * R) / lambda_val) + s

    pRec = pSend - PL
    if (pRec > pSend):
        pRec = pSend
    return pRec


def cal_XR_min(r_m, m, S, rate_UE, rate_m_min):
    # This function to calculate minimum the number of RBs for each UE of the MVNO m.
    # Input:
    # rate_m_min: minimum data rate requirement for UEs at MVNO m
    # r_m  : total number of RBs for MVNO m
    # S: Number of UEs of MVNO m
    # rate_UE: Data rate of the UEs with one RB
    # m: index of the MVNO
    # Output:
    # X_m_min: minimum fraction of number of RBs of MVNO m
    # R_m: maximum data rate of users in MVNO m with r_m to the MVNO m
    s_m = S[m];
    if m == 0:
        # print(rate_UE[0:s_m][0])
        X_m_min = rate_m_min / (r_m * rate_UE[0:s_m, 0]);
        R_m = r_m * rate_UE[0:s_m, 0]
    else:
        X_m_min = rate_m_min / (r_m * rate_UE[sum(S[:m]):(sum(S[:m]) + S[m]), 0]);
        R_m = r_m * rate_UE[sum(S[:m]):(sum(S[:m]) + S[m]), 0]

    return X_m_min, R_m


    # X_m_min = cal_R_min(20, 0, S, rate_UE, 2)
    # # print(S)
    # print (X_m_min)


# Test: OK

def DD_method(S, Z_m_max, m, r_m, rate_UE, rate_m_min):
    # Input parameter: S, m
    # Input:
    # S = np.array([5, 15, 25]) # Number of users at each MVNO
    # Z_m_max =  np.array([20, 40, 100]) # Capacity limitation
    # m = 0;
    # r_m = 20;
    # rate_UE;
    # rate_m_min = 2; # minimum data rate requirement of UEs at MVNO m

    # Output parameter: X_m (vector of users of the service provider m)
    # vector data rate of the MVNO m R_m * X_mR_m*X_m

    # Starting function
    z_m = Z_m_max[m];  # Backhaul capacity of the MVNO m
    s_m = S[m];  # Number of UEs at MVNO m

    # algorithm parameter

    iter_m = 100;  # Number of iterations for MVNO m
    X_m = np.zeros(iter_m)  # Matrix of resource allocation of all UE
    # at MVNO m following iteration iter_m

    lambda_m = np.zeros(iter_m)

    mu_m = np.zeros(iter_m)

    sum_rate_m = np.zeros(iter_m)

    sum_X_m = np.zeros(iter_m)

    # Initiate variable:
    lambda_m[0] = 7
    mu_m[0] = 0.2
    sum_rate_m[0] = 0;

    if s_m <= 9:
	    theta_1 = 1
	    theta_3 = 0.001
    elif 9 < s_m <= 15:
	    theta_1 = 1sẩng2012
	    theta_3 = 0.001
    elif s_m > 15:
	    theta_1 = 1
	    theta_3 = 0.0001

    # Getting minimumfraction of RB of number of RBs of each MVNO
    # Getting R_m (see in the paper or problem analysis):
    X_m_min, R_m = cal_XR_min(r_m, m, S, rate_UE, rate_m_min)

    # print(X_m_min, R_m)
    # Starting algorithm of DD method

    for i in range(1, iter_m):
        # Update vector X_m at all UE

        # print ('lambda_m =', lambda_m[i-1])
        # print ('R_m*mu_m =', R_m*mu_m[i-1])

        X_m = 1/(lambda_m[i - 1] + R_m * mu_m[i - 1])
        # print (X_m)
        # print (lambda_m[999-1])
        # Update vector lambda_m
        lambda_m[i] = max(0, lambda_m[i - 1] - theta_1 * (1 - np.sum(X_m)))

        # Update vecotr mu_m
        mu_m[i] = max(0, mu_m[i - 1] - theta_3 * (z_m - np.sum(R_m * X_m)))
        # print (R_m*X_m, X_m)

        # Update total rate of MVNO m
        sum_rate_m[i] = np.sum(R_m * X_m)
        sum_X_m[i] = np.sum(X_m)



    # t1 = np.arange(iter_m)

    # print(lambda_m)

    #   plt.figure(1)
    #   plt.subplot(211)
    #   plt.plot(t1, lambda_m, 'bo')

    #   plt.subplot(212)
    #   plt.plot(t1, mu_m, 'k')
    #   plt.show()

    #   plt.figure(2)
    #   plt.plot(t1, sum_rate_m, '--k')
    #   plt.show()

    #   plt.figure(3)
    #   plt.plot(t1, sum_X_m, '-g')

    # plt.show()
    # print ('sum(X_m) = ', np.sum(X_m))

    # print ('np.sum(R_m*X_m) =', np.sum(R_m*X_m))

    return np.sum(X_m), np.sum(R_m * X_m), R_m*X_m

# # Input:
# S = np.array([5, 15, 25]) # Number of users at each MVNO
# Z_m_max =  np.array([20, 40, 100]) # Capacity limitation
# m = 1;
# # z_m = Z_m_max[m]; # Backhaul capacity of the MVNO m
# # s_m = S[m]; # Number of UEs at MVNO m
# r_m =100;
# rate_UE;
# rate_m_min = 2; # minimum data rate requirement of UEs at MVNO m

# sum_X_m, sum_rate_m = DD_method(S,Z_m_max, m, r_m, rate_UE, rate_m_min)

# print (sum_X_m, sum_rate_m)


"""Main function"""

# Initializate network model

I = 1;  # Number of InP
M = 3;  # number of MVNO
S = np.array([5, 10, 25])  # Number of users at each MVNO

# Getting network model (using class)

# Number of MBS
n_MBS = I;
n_UE = sum(S);
radius_MBS = 500;  # m
bandwidth = 20;  # Mhz
bw = 0.18  # MHz, bandwidth of each RB
power_MBS = 43  # dBm
n_RB = 100 # Number of RBs

# Z_m_max = np.array([100, 40, 70])

Z_m_max = np.array([100, 100, 100])

rate_m_min = 0.5
# Calculate noise
Noise = 10 ** (calculateNoise(bandwidth) / 10)  # ~~ -100dBW
Noise = 10 ** (-11)  #
# Noise ~ 10**(-14); # Wats
# print(Noise)

# network_model.update(n_MBS,n_UE)
# H = network_model.channel_gain();
# P = network_model.power_downlink();

# Create network topology and display

# Getting Coors of BSs and UEs
(Coor_BS, Coor_UE, distance_BS_to_UE) = create_system(radius_MBS, n_UE)

# compute receivedSINRatUE 2D matrix
rate_UE = np.zeros(shape=(n_UE, n_MBS))  # Matrix to update penalty to MVNO

for i in range(n_UE):
    for j in range(n_MBS):
        # SINR_Rx[i][j] = 10*math.log10(10**(calculateReceivedPower(power_MBS, distance_BS_to_UE[i][j])/10)/Noise)
        rate_UE[i][j] = 0.18 * np.log2(
            1 + 10 ** (calculateReceivedPower(power_MBS, distance_BS_to_UE[i][j]) / 10) / Noise)

# print(rate_UE)

iter = 100# Number of Interations

Q = np.zeros(shape=(M, iter))  # Matrix to update penalty to MVNO
Q[:, 0] = 0.1

B = np.zeros(shape=(M, iter))  # Matrix to update biding value to InP
B[:, 0] = 0.1

BETA = np.zeros(shape=(I, iter))  # Matrix to update biding value to InP
BETA[:, 0] = 0.1

RATE = np.zeros(shape=(M, iter))  # Sum data rate matrix of MVNOs
RATE[:, 0] = 10 # s


X_M = np.zeros(shape=(M, iter))  # Sum data rate matrix of MVNOs
X_M[:, 0] = 0.8 # Mbps

RM = np.zeros(shape=(M, iter))  # Number of RBs allocate to MVNOs
RM[:, 0] = n_RB/M

RV = np.zeros(shape=(np.sum(S), iter))  # Vector data rate of users in all MVNOs
RV[:, 0] = 0.5 # Mbps

# Stage I: Resource Competition Game


for i in range(1, iter):
    print (i)

    # PHASE I

    # Step 1: Updating penalty value to MNVO --> Q matrix
        # Using function: penalty_to_mvno(beta, r_m, R, q_m_old, b)
    for m in range(M):
        Q[m][i] = penalty_to_mvno(BETA[0][i-1], RM[m][i-1], n_RB, Q[m][i-1], B[:, i-1])
    #print('Q[:, i] =', Q[:, i])

    # Step 2: Updating bidding value to InP --> B
    # Using function:
    for m in range(M):

        s_m = S[m]
        if m == 0:
            # print(rate_UE[0:s_m][0])
            RV_m = RV[0:s_m, i - 1]
        else:
            RV_m = RV[sum(S[:m]):(sum(S[:m]) + S[m]), i - 1]

        B[m][i] = bidding_value(RV_m)
    #print('B[:, i] =', B[:, i])

    # Step 3: Broadcast virtual price (how much for each resource block)
    # using virtual_price(b, R)
    BETA[0][i] = virtual_price(B[:, i], n_RB)

    #print('BETA[0][i] =', BETA[0][i] )

    # Step 4: InP calculates number of RBs to allocate to each MVNO according to
    # bidding value
    # Using function: cal_number_of_rbs(b_m, b, R)
    for m in range(M):
        RM[m][i] = cal_number_of_rbs(B[m][i], B[:, i], n_RB)
        RM[m][i] = int(RM[m][i])
    #print('RM[:, i] =', RM[:, i])

    # PHASE II
    # Step 5: Updating resource allocation at each MVNO
    for m in range(M):
        # np.sum(X_m), np.sum(R_m * X_m), R_m*X_m = DD_method(S, Z_m_max, m, r_m, rate_UE, rate_m_min)
        X_M_A, RATE_A, V_m_A = DD_method(S, Z_m_max, m, RM[m][i], rate_UE, rate_m_min)
        s_m = S[m]
        if m == 0:
            # print(rate_UE[0:s_m][0])
            RV[0:s_m, i] = V_m_A
        else:
            RV[sum(S[:m]):(sum(S[:m]) + S[m]), i] = V_m_A
        X_M[m][i] = X_M_A
        RATE[m][i] = RATE_A

    #print('RV[:, i] =', RV[:, i])
    #print('X_M[:, i] =', X_M[:, i])
    #print('RATE[:, i] =', RATE[:, i])




# Stage II: Output display

# 0. Network model

# plot network model
# display_network_model(Coor_BS, Coor_UE, M, S, radius_MBS)

# 1. Data rate of each MVNO following time

plot_RATE(RATE, M, iter)


plot_RA(RM, M, iter)

# 2. Plot proportional allocation:
