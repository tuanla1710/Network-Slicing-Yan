from __future__ import division

print 'Hello, Colaboratory!'
# !pip install cvxpy
# !pip install ncvx
# !pip install numpy
# !pip
# install
# matplotlib

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
    q_m_new = 1 / beta_local * (1 - r_m_local / R_local) * q_m_old / (R_local - r_m_local) * np.sum(b_local)
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
    r_m = (b_m / (np.sum(b))) * R  # Question: Do we need to round this number to an Integer value ???
    # This one will be considered after finishing ...
    return r_m

    # # Test
    # b_m = 4;
    # b = np.array([4,5,10]);
    # R = 20;

    # r_m = cal_number_of_RBs(b_m, b,R)

    # print(r_m)


'''Module design'''


def create_system(rad, num, dist_min=5):
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
    plt.title('Network topology')
    plt.legend()
    plt.show()


def plot_RATE(rate, M, iter):
    plt.figure(2)
    t = range(iter)
    y_max = rate.max() + rate.max() / 10
    for m in range(M):
        if (m == 0):
            plt.plot(t, rate[0, :], "-go", ms=5, label='MVNO-1')
        elif (m == 1):
            plt.plot(t, rate[1, :], "-b*", ms=5,
                     label='MVNO-2')
        elif (m == 2):
            plt.plot(t, rate[2, :], "-m>", ms=5,
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
    y_max = rate.max() + rate.max() / 10
    for m in range(M):
        if (m == 0):
            plt.plot(t, rate[0, :], "-go", ms=5, label='MVNO-1')
        elif (m == 1):
            plt.plot(t, rate[1, :], "-b*", ms=5,
                     label='MVNO-2')
        elif (m == 2):
            plt.plot(t, rate[2, :], "-m>", ms=5,
                     label='MVNO-3')
        else:
            ValueError('M must be less than 3!')
    plt.axis([0, iter, 0, y_max])
    plt.title('Generalized Kelly-based bandwidth allocation')
    plt.legend()
    plt.grid()
    plt.ylabel('Valuation function')
    plt.xlabel('Iteration')
    plt.show()


def plot_utility(rate, M, iter):
    plt.figure(4)
    t = range(iter)
    y_max = rate.max() + rate.max() / 10
    for m in range(M):
        if (m == 0):
            plt.plot(t, rate[0, :], "-go", ms=5, label='MVNO-1')
        elif (m == 1):
            plt.plot(t, rate[1, :], "-b*", ms=5,
                     label='MVNO-2')
        elif (m == 2):
            plt.plot(t, rate[2, :], "-m>", ms=5,
                     label='MVNO-3')
        else:
            ValueError('M must be less than 3!')
    plt.axis([0, iter, 0, y_max])
    # plt.title('Generalized Kelly-based bandwidth allocation')
    plt.legend()
    plt.grid()
    plt.ylabel('Utility of MVNOs')
    plt.xlabel('Iteration')
    plt.show()

def plot_fraction_bandwidth(rate, M, iter):
    plt.figure(5)
    t = range(iter)
    y_max = rate.max() + rate.max() / 10
    for m in range(M):
        if (m == 0):
            plt.plot(t, rate[0, :], "-go", ms=5, label='MVNO-1')
        elif (m == 1):
            plt.plot(t, rate[1, :], "-b*", ms=5,
                     label='MVNO-2')
        elif (m == 2):
            plt.plot(t, rate[2, :], "-m>", ms=5,
                     label='MVNO-3')
        else:
            ValueError('M must be less than 3!')
    plt.axis([0, iter, 0, y_max])
    # plt.title('Generalized Kelly-based bandwidth allocation')
    plt.legend()
    plt.grid()
    plt.ylabel('Bandwidth allocation (MHz)')
    plt.xlabel('Iteration')
    plt.show()


def plot_FN(rate, M, iter):
    plt.figure(6)
    t = range(iter)
    y_max = rate.max() + rate.max() / 10
    for m in range(1):
        if (m == 0):
            plt.plot(t, rate, "-go", ms=5, label='Our proposal')
        else:
            ValueError('M must be less than 3!')
    plt.axis([0, iter, 0, y_max])
    # plt.title('Generalized Kelly-based bandwidth allocation')
    plt.legend()
    plt.grid()
    plt.ylabel('Fairness index')
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

def KKT_method(S, m, r_m, rate_UE, rate_m_min):
    # r_m: fraction of bandwidth allocating to MVNO m
    # S = np.array([10, 20, 30]) # Number of users at each MVNO
    # rate_UE: SINR of all users.
    # rate_m_min: array of minimum requirement for each user in each MVNO.

    # Output parameter: X_m (vector of users of the service provider m)
    # vector data rate of the MVNO m R_m * X_mR_m*X_m

    s_m = S[m];  # Number of UEs at MVNO m
    s_m_nonQoS = int(s_m / 2);  # Number of non-QoS users
    s_m_QoS = int(s_m / 2);  # Number of QoS users

    # calculate the fraction of bandwidth for each QoS user.
    # print ("s_m = ", s_m)
    X_m = np.zeros(s_m)  # Matrix of resource allocation of all UEs

    # print ("s_m_QoS -1 = ", s_m_QoS-1 )
    # print('rate_UE = ', rate_UE[:,0]);
    for i in range(0, s_m_QoS):
        if m == 0:
            # print(rate_UE[0:s_m][0])
            X_m[i] = rate_m_min[m] / (rate_UE[i][0]);
            # R_m = r_m * rate_UE[0:s_m, 0]
        elif m == 1:
            X_m[i] = rate_m_min[m] / rate_UE[i + S[m - 1]][0];

        elif m == 2:
            X_m[i] = rate_m_min[m] / rate_UE[i + S[m - 1] + S[m - 2]][0];
        else:
            print("Number of MVNOs = 3")

            # R_m = r_m * rate_UE[sum(S[:m]):(sum(S[:m]) + S[m]), 0]

    # assign fraction of bandwidth to non-QoS users:
    # print("X_m = ", X_m);

    sum_X_m_nonQoS = r_m - np.sum(X_m)  # total fraction of bandwidth allocate to non_QoS users

    # print("np.sum(X_m)  = ",np.sum(X_m) );
    #print("r_m  = ", r_m);
    #print("sum_X_m_nonQoS = ", sum_X_m_nonQoS);



    x_bandwidth_each_non_QoS_user = sum_X_m_nonQoS / s_m_nonQoS;

    X_m[s_m_QoS:s_m] = x_bandwidth_each_non_QoS_user;

    # print("Xm = ", X_m);

    # Calculate valuation for each MVNO:
    V_m = np.zeros(s_m);
    R_m = np.zeros(s_m);

    for i in range(0, s_m):
        if m == 0:
            # print(rate_UE[0:s_m][0])
            V_m[i] = np.log10(X_m[i] * rate_UE[i][0]);
            R_m[i] = X_m[i] * rate_UE[i][0];
            # R_m = r_m * rate_UE[0:s_m, 0]
        elif m == 1:
            V_m[i] = np.log10(X_m[i] * rate_UE[i + S[m - 1]][0]);
            R_m[i] = X_m[i] * rate_UE[i + S[m - 1]][0];

        elif m == 2:
            V_m[i] = np.log10(X_m[i] * rate_UE[i + S[m - 1] + S[m - 2]][0]);
            R_m[i] = X_m[i] * rate_UE[i + S[m - 1] + S[m - 2]][0];
        else:
            print("Number of MVNOs = 3")

    # print("V_m =", V_m);

    return np.sum(X_m), np.sum(V_m), V_m, X_m


def plot_bandwidth_allocation(X1):
    import pandas as pd
    X = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
    df = pd.DataFrame(np.transpose(X1), index=X, columns=['MVNO-1', 'MVNO-2', 'MVNO-3'])
    ax = df.plot.bar(width=0.8);
    ax.set_xlabel("User index")
    ax.set_ylabel("Bandwidth (MHz)")
    plt.show()


print ("Starting ...")

"""Main function"""

# Initializate network model

# plt.close(all);

I = 1;  # Number of InP
M = 3;  # number of MVNO
S = np.array([10, 14, 6])  # Number of users at each MVNO

# Getting network model (using class)

# Number of MBS
n_MBS = I;
n_UE = sum(S);
radius_MBS = 500;  # m
bandwidth = 20;  # Mhz
# bw = 0.18  # MHz, bandwidth of each RB
power_MBS = 43  # dBm
# n_RB = 100 # Mbps Number of RBs

# Z_m_max = np.array([100, 40, 70])

# Z_m_max = np.array([100, 100, 100])
rate_m_min = np.array([2, 4, 6]);  # Mbps

# Calculate noise
Noise = 10 ** (calculateNoise(bandwidth) / 10)  # ~~ -100dBW
Noise = 10 ** (-11)  #
# Noise ~ 10**(-14); # Wats
# print(Noise)

# network_model.update(n_MBS,n_UE)
# H = network_model.channel_gain();
# P = network_model.power_downlink();

for k in range(1000):


# Create network topology and display

# Getting Coors of BSs and UEs
(Coor_BS, Coor_UE, distance_BS_to_UE) = create_system(radius_MBS, n_UE)

# compute receivedSINRatUE 2D matrix
rate_UE = np.zeros(shape=(n_UE, n_MBS))  # Matrix to update penalty to MVNO

for i in range(n_UE):
    for j in range(n_MBS):
        # SINR_Rx[i][j] = 10*math.log10(10**(calculateReceivedPower(power_MBS, distance_BS_to_UE[i][j])/10)/Noise)
        rate_UE[i][j] = np.log2(
            1 + 10 ** (calculateReceivedPower(power_MBS, distance_BS_to_UE[i][j]) / 10) / Noise)

# print(rate_UE)

# print(test)

iter = 25  # Number of Interations

Q = np.zeros(shape=(M, iter))  # Matrix to update penalty to MVNO
Q[:, 0] = 0.1

B = np.zeros(shape=(M, iter))  # Matrix to update biding value to InP
B[:, 0] = 0.1

BETA = np.zeros(shape=(I, iter))  # Matrix to update biding value to InP
BETA[:, 0] = 0.1

RATE = np.zeros(shape=(M, iter))  # Sum data rate matrix of MVNOs
RATE[:, 0] =  0 # s

RATE[0, 0] =  1 # s
RATE[1, 0] =  6 # s
RATE[2, 0] =  14 # s


S = np.array([10, 14, 6])  # Number of users at each MVNO


X_M = np.zeros(shape=(M, iter))  # Sum fractionh of bandwidth to each MVNO
X_M[:, 0] = 0  # Mbps

# RM = np.zeros(shape=(M, iter))  # Number of RBs allocate to MVNOs
# RM[:, 0] = n_RB/M

RM = np.zeros(shape=(M, iter))  # The fraction of bandwidth to MVNOs
RM[:, 0] = bandwidth / M

UM = np.zeros(shape=(M, iter))  # The fraction of bandwidth to MVNOs

RV = np.zeros(shape=(np.sum(S), iter))  # Vector data rate of users in all MVNOs
RV[:, 0] = 0.5  # Mbps


max_S = max(S);
X_M_user_MVNO = np.zeros(shape=(M, max_S)) ; # fraction of bandwidth allocating to each users at each MVNO.

# Stage I: Resource Competition Game


for i in range(1, iter):
    print (i)

    # PHASE I

    # Step 1: Updating penalty value to MNVO --> Q matrix
    # Using function: penalty_to_mvno(beta, r_m, R, q_m_old, b)
    for m in range(M):
        Q[m][i] = penalty_to_mvno(BETA[0][i - 1], RM[m][i - 1], bandwidth, Q[m][i - 1], B[:, i - 1])
    # print('Q[:, i] =', Q[:, i])
    # print(test);
    # Step 2: Updating bidding value to InP --> B
    # Using function:
    for m in range(M):

        s_m = S[m]
        if m == 0:
            # print(rate_UE[0:s_m][0])
            RV_m = RV[0:s_m, i - 1]
        else:
            RV_m = RV[sum(S[:m]):(sum(S[:m]) + S[m]), i - 1]

        # print("RV=", RV)

        B[m][i] = bidding_value(RV_m)
    # print('B[:, i] =', B[:, i])

    # Step 3: Broadcast virtual price (how much for each resource block)
    # using virtual_price(b, R)
    BETA[0][i] = virtual_price(B[:, i], bandwidth)

    # print('BETA[0][i] =', BETA[0][i] )

    # Step 4: InP calculates number of RBs to allocate to each MVNO according to
    # bidding value
    # Using function: cal_number_of_rbs(b_m, b, R)
    # print('RM[:, i] =', RM[:, i-1])
    # print(test);

    for m in range(M):
        print "B[m][i-1] =", B[m][i-1];
        print "B[:, i-1] =", B[:, i-1];


        RM[m][i] = cal_number_of_rbs(B[m][i], B[:, i], bandwidth) # --> actually, this function will return fraction of bandwidth we will allocate.
        # RM[m][i] = RM[m][i]
    # print('RM[:, i] =', RM[:, i])
    # print(test);



    # PHASE II
    # Step 5: Updating resource allocation at each MVNO
    for m in range(M):
        # np.sum(X_m), np.sum(R_m * X_m), R_m*X_m = DD_method(S, Z_m_max, m, r_m, rate_UE, rate_m_min)
        # X_M_A, RATE_A, V_m_A = DD_method(S, Z_m_max, m, RM[m][i], rate_UE, rate_m_min)
        s_m = S[m];
        print("RM[m][i]", RM[m][i]);

        X_M_A, RATE_A, V_m_A, X_m_user_MVNO = KKT_method(S, m, RM[m][i], rate_UE, rate_m_min)
        # rate_UE is log2(1+SNR) in the Globecom paper).

        # Update data rate of all UEs
        if m == 0:
            # print(rate_UE[0:s_m][0])
            # print(np.shape(V_m_A), np.shape(RV[0:s_m, i]));
            # print (s_m)
            RV[0:s_m, i] = V_m_A
            # V_m_A: vector data rate of USERS in the MVNO m
        else:
            RV[sum(S[:m]):(sum(S[:m]) + S[m]), i] = V_m_A

        X_M[m][i] = X_M_A
        RATE[m][i] = RATE_A  # RATE_A is exactly the valuation of each MVNO. log(rate)...


        # Updating the fraction of bandwidth allocating to each user at each MVNO
        X_M_user_MVNO[m,0:s_m] = X_m_user_MVNO;


        # Update utility value of each MVNO
        UM[m][i] =  RATE_A - Q[m][i]*B[m][i];
        UM[m][0] = RATE_A - Q[m][0] * B[m][0];


    # print('RV[:, i] =', RV[:, i])
    # print('X_M[:, i] =', X_M[:, i])
    # print('RATE[:, i] =', RATE[:, i])

# Stage II: Output display

# 0. Network model

# 1. plot network model
#display_network_model(Coor_BS, Coor_UE, M, S, radius_MBS)

# 2. Data rate of each MVNO following time

print("RATE = ", RATE, RM);
# plot_RA(RATE, M, iter)

# 3. Plot bandwidth allocation to each user:
print (X_M_user_MVNO)
# plot_bandwidth_allocation(X_M_user_MVNO);

# 4. Plot utility of each MVNO (u = v - q*b):
# plot_utility(UM, M, iter);


# 5. Plot totol the fraction of bandwidth to each MVNO

# plot_fraction_bandwidth(X_M, M, iter);



# 6. Ploting Jain's fairness index of among all MVNO



Fairness = np.zeros(shape=(1, iter))


Fair_x = (np.sum(RATE,0))**2;


Fair_2 = (np.sum(RATE**2,0))# **2;

Fairness = Fair_x/(3*Fair_2);


print("RATE", RATE)

print("Fair_x =",Fair_x);
print("Fair_x_2 =",Fair_2)
print("Fairness =",Fairness)

plot_FN(Fairness, M, iter)


# plot_fraction_bandwidth(X_M, M, iter);

