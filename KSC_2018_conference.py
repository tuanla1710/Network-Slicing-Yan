from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pdb # to debug

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


clear_all()


def distancesFromAtoB(coorA, coorB):
    """
    Input:
    - coorA: coordinators of As (EV)
    - coorB: coordinators of Bs (CS)
    * Ouput:
    - array 2D
    - distance between A and B

    """

    # Getting number of As
    nA = coorA.shape[0]
    # Getting number of Bs
    nB = coorB.shape[0]

    D = np.zeros(shape=(nA, nB))

    for a in range(nA):
        for b in range(nB):
            D[a][b] = math.sqrt((coorA[a][0] - coorB[b][0]) ** 2
                                + (coorA[a][1] - coorB[b][1]) ** 2)
    return D


def simulation_setting(number_of_EV=10, number_of_CS=3, radius=2000, d_min=200):
    # In this setting:
    # The number of RSU is one
    # d_min : mnimum distance between EVs and CSs , EVs and EVs, CSs and CSs.

    # output:

    cost = np.array([[number_of_EV, number_of_CS]])

    num = number_of_EV + number_of_CS  # total ES in the system

    x_RSU = radius
    y_RSU = radius

    Coor_RSU = np.array([[x_RSU, y_RSU]])
    Coor_EV = np.zeros(shape=(number_of_EV, 2))  # number of EVs in the system
    Coor_CS = np.zeros(shape=(number_of_CS, 2))  # number of EVs in the system

    check = False

    while (check == False):
        t = np.random.uniform(0.0, 2.0 * np.pi, num)  # possitions of EV and CS
        r = radius * np.sqrt(np.random.uniform(0.0, 1.0, num))
        x = Coor_RSU[0][0] + r * np.cos(t)
        y = Coor_RSU[0][1] + r * np.sin(t)

        # Update coordinator of EVs

        # print(Coor_CS,number_of_CS)
        # print x[0:(number_of_CS)]

        # print('num', num)

        Coor_CS[:, 0] = x[0:(number_of_CS)]
        Coor_CS[:, 1] = y[0:(number_of_CS)]
        Coor_EV[:, 0] = x[number_of_CS:num]
        Coor_EV[:, 1] = y[number_of_CS:num]

        # calculate distance UEs and CSs

        D = distancesFromAtoB(Coor_EV, Coor_CS)

        # print(D, D.min())

        if D.min() >= d_min:
            check = True
            # compute cost

            # https://www.energuide.be/en/questions-answers/how-much-power-does-an-electric-car-use/212/
            # 4500 in electricity charges to travel 100km. 6500/100000
            C = D * 9500 / 100000

    return (D, C, Coor_EV, Coor_CS)


# testing: x =   simulation_setting(number_of_EV = 10, number_of_CS = 3, radius = 2000)
# distance_EV_CS, cost_EV_CS, coor_EV, coor_CS = simulation_setting(number_of_EV = 10, number_of_CS = 3, radius = 2000, d_min = 200)


def display_network_model(coor_CS, coor_EV, number_of_EV=10, number_of_CS=3, radius=2000):
    #   * Output:
    #   - Display network information

    #   """

    # Plot CS location
    # plotting points as a scatter plot

    plt.figure(1)
    plt.scatter(coor_CS[:, 0], coor_CS[:, 1], label='Mobile Charging Station', color="red",
                marker="o", s=200)

    plt.scatter(coor_EV[:, 0], coor_EV[:, 1], label='Electric Vehicle', color="blue",
                marker="s", s=25)

    # x-axis label
    plt.xlabel('x - axis (m)')
    # frequency label
    plt.ylabel('y - axis (m)')
    # plot title
    plt.title('Simulation setting')
    # showing legend
    plt.legend()

    # function to show the plot
    plt.show()


def one_to_many_matching(Weight, q, number_of_EV, number_of_CS):  # output is connection matrix following t
    # input: here, the Weight = number_of_EV x number_of_CS


    # output initialization
    output = np.zeros(shape=(number_of_EV, number_of_CS), dtype=int)

    # sorting
    sorted_index = np.argsort(-Weight, axis=1)  # Due to indexing MCS starting from 1 to 3.
    # our simulation this part is only for the EVs where can observe whole MCSs in the system. Why? Because
    # when we perform sorting, it always appear in the system the index.

    # print("Weight", Weight)
    # print("sorted_index" ,sorted_index)
    # pdb.set_trace()

    # store the reward following convergence of the matching algoirhtm
    reward = np.zeros(number_of_EV * number_of_CS)

    # print("reward.shape = ", reward)
    index = 0 # indexing of steps
    stop = 0  # to stop algorihtm
    while (stop < 3): # until the matching are the same in tree cosequence timestep
        # print('index',index)
        MCS_number_of_EV = np.zeros(number_of_CS, dtype=int)  # reset the number of EV in each MCS
        MCS_getting_EV = np.zeros(shape=(number_of_EV, number_of_CS), dtype=int) -1 # EV index from the EV request
        MCS_data = np.zeros(shape=(number_of_EV, number_of_CS))  # data geeting from EVs

        # EV sending requests and MCS receive requests
        for n in range(number_of_EV):
            # print("n", n)
            # pdb.set_trace()
            if sorted_index[n, 0]!=-1: # it has no MCS in the preference list
                MCS_number_of_EV[sorted_index[n, 0]] = MCS_number_of_EV[sorted_index[n, 0]] + 1  # MCS increases 1 EV

                # getting index
                MCS_getting_EV[MCS_number_of_EV[sorted_index[n, 0]] - 1, sorted_index[n, 0]] = n # exactly indexing of n in MCS
                # MCS_number_of_EV[sorted_index[n, 0]] - 1 is the index in the MCS list of EVs
                # sorted_index[n, 0] is MCS index

                # updating which EV belongs to which MCS
                # -1 to address index of the EV getting from 0
                # sorted_index[n, 0] is the MCS indexing
                # here, actuall EV index will be minus 1 due to complexity of the algorithm design.

                # getting data
                MCS_data[MCS_number_of_EV[sorted_index[n, 0]] - 1, sorted_index[n, 0]] = Weight[n, sorted_index[n, 0]]  # value of utility function from EV n
                # sorted_index[n, 0] is MCS index

        # sorting EVs in each MCS
        MCS_sorted_index = np.argsort(-MCS_data, axis=0)  # index of posstion, sorting EVs in each MCS

        # print("debug-2", MCS_data, MCS_sorted_index, MCS_number_of_EV)

        # pdb.set_trace()

        # from MCS_sorted_index, we can find EV index from MCS_getting_EV and its value: MCS_data

        # Data test --> OK
        # print("MCS_sorted_index", MCS_sorted_index, "MCS_number_of_EV", MCS_number_of_EV, "MCS_data", MCS_data)
        # print("MCS_sorted_index", MCS_sorted_index)
        # pdb.set_trace()
        # print("debug-1")
        # pdb.set_trace()
        for m in range(number_of_CS):
            # print(q[m])
            # pdb.set_trace()
            if (MCS_number_of_EV[m] > 0):  # MCS_number_of_EV: number of EVs in m
                if (MCS_number_of_EV[m] <= q[m]):
                    # find EV indexes
                    list_EV_index_in_MCS_data = MCS_sorted_index[:MCS_number_of_EV[m], m]
                    list_EV_index_in_Weight = MCS_getting_EV[list_EV_index_in_MCS_data, m]
                    # print("list_EV_index_in_Weight", list_EV_index_in_Weight)
                    # pdb.set_trace()
                    # EV_id = MCS_getting_EV[:MCS_number_of_EV[m], m] # EV_id of EVs will be associated
                    # output[:, m] = 0;  # delete current connection at MCS. Have no connection when go to this step

                    # delete current connection
                    # for mm in range(number_of_CS):
                    #     for nn in EV_id:
                    #         output[n,mm] = 0
                    # assign to a new connection.
                    output[list_EV_index_in_Weight, m] = 1

                if MCS_number_of_EV[m] > q[m]:
                    # Removing EVs
                    list_EV_index_in_MCS_data = MCS_sorted_index[:q[m], m]
                    list_EV_index_in_Weight = MCS_getting_EV[list_EV_index_in_MCS_data, m]
                    # print("list_EV_index_in_Weight", list_EV_index_in_Weight)
                    # pdb.set_trace()
                    # EV_id = MCS_getting_EV[:MCS_number_of_EV[m], m] # EV_id of EVs will be associated
                    # output[:, m] = 0;  # delete current connection at MCS. Have no connection when go to this step

                    # delete current connection
                    # for mm in range(number_of_CS):
                    #     for nn in EV_id:
                    #         output[n,mm] = 0

                    # assign to a new connection.
                    output[list_EV_index_in_Weight, m] = 1

                    for nn in range(q[m], MCS_number_of_EV[m]):  # EVs out of quota will be rejected.
                        # output[MCS_getting_EV[nn] - 1, m] = 0  # remove with current connection
                        # getting index of EV need to delete
                        # pdb.set_trace()
                        EV_index_in_MCS_data = MCS_sorted_index[nn,m] # index of EV in the MCS_data
                        EV_index_in_Weight = MCS_getting_EV[EV_index_in_MCS_data, m]  # index of EV in the MCS_data

                        for j in range(number_of_CS-1):  # For all MCSs of the EV
                            # determine EV index

                            sorted_index[EV_index_in_Weight, j] = sorted_index[EV_index_in_Weight, j+1]
                            # MCS_sorted_index[nn] is EV index of the sorted MCS ?? no


                        # setting -1 to the final entity
                        sorted_index[EV_index_in_Weight, number_of_CS-1] = -1  # set to -1 if no MCS


        index = index + 1

        # print('output', output)
        # print('Weight', sorted_index)
        # print('q', q)
        # pdb.set_trace()


        r = sum(output * Weight)
        # print('r = ', r, 'sum(r) = ', sum(r))
        # pdb.set_trace()
        reward[index] =  sum(r)

        if reward[index] == reward[index - 1]:
            stop = stop + 1
        # print('reward', reward)
        # pdb.set_trace()
    return output



def one_to_many_matching_greedy(Weight, q, number_of_EV, number_of_CS):  # output is connection matrix following t
    # input: here, the Weight = number_of_EV x number_of_CS


    # output initialization
    output = np.zeros(shape=(number_of_EV, number_of_CS), dtype=int)

    # sorting
    sorted_index = np.argsort(-Weight, axis=1)  # Due to indexing MCS starting from 1 to 3.
    # our simulation this part is only for the EVs where can observe whole MCSs in the system. Why? Because
    # when we perform sorting, it always appear in the system the index.

    # print("Weight", Weight)
    # print("sorted_index" ,sorted_index)
    # pdb.set_trace()

    # store the reward following convergence of the matching algoirhtm
    reward = np.zeros(number_of_EV * number_of_CS)

    # print("reward.shape = ", reward)
    index = 0 # indexing of steps
    stop = 0  # to stop algorihtm
    while (stop < 3): # until the matching are the same in tree cosequence timestep
        # print('index',index)
        MCS_number_of_EV = np.zeros(number_of_CS, dtype=int)  # reset the number of EV in each MCS
        MCS_getting_EV = np.zeros(shape=(number_of_EV, number_of_CS), dtype=int) -1 # EV index from the EV request
        MCS_data = np.zeros(shape=(number_of_EV, number_of_CS))  # data geeting from EVs

        # EV sending requests and MCS receive requests
        for n in range(number_of_EV):
            # print("n", n)
            # pdb.set_trace()
            if sorted_index[n, 0]!=-1: # it has no MCS in the preference list
                MCS_number_of_EV[sorted_index[n, 0]] = MCS_number_of_EV[sorted_index[n, 0]] + 1  # MCS increases 1 EV

                # getting index
                MCS_getting_EV[MCS_number_of_EV[sorted_index[n, 0]] - 1, sorted_index[n, 0]] = n # exactly indexing of n in MCS
                # MCS_number_of_EV[sorted_index[n, 0]] - 1 is the index in the MCS list of EVs
                # sorted_index[n, 0] is MCS index

                # updating which EV belongs to which MCS
                # -1 to address index of the EV getting from 0
                # sorted_index[n, 0] is the MCS indexing
                # here, actuall EV index will be minus 1 due to complexity of the algorithm design.

                # getting data
                MCS_data[MCS_number_of_EV[sorted_index[n, 0]] - 1, sorted_index[n, 0]] = Weight[n, sorted_index[n, 0]]  # value of utility function from EV n
                # sorted_index[n, 0] is MCS index

        # sorting EVs in each MCS
        MCS_sorted_index = np.argsort(-MCS_data, axis=0)  # index of posstion, sorting EVs in each MCS

        # print("debug-2", MCS_data, MCS_sorted_index, MCS_number_of_EV)

        # pdb.set_trace()

        # from MCS_sorted_index, we can find EV index from MCS_getting_EV and its value: MCS_data

        # Data test --> OK
        # print("MCS_sorted_index", MCS_sorted_index, "MCS_number_of_EV", MCS_number_of_EV, "MCS_data", MCS_data)
        # print("MCS_sorted_index", MCS_sorted_index)
        # pdb.set_trace()
        # print("debug-1")
        # pdb.set_trace()
        for m in range(number_of_CS):
            # print(q[m])
            # pdb.set_trace()
            if (MCS_number_of_EV[m] > 0):  # MCS_number_of_EV: number of EVs in m
                if (MCS_number_of_EV[m] <= q[m]):
                    # find EV indexes
                    list_EV_index_in_MCS_data = MCS_sorted_index[:MCS_number_of_EV[m], m]
                    list_EV_index_in_Weight = MCS_getting_EV[list_EV_index_in_MCS_data, m]
                    # print("list_EV_index_in_Weight", list_EV_index_in_Weight)
                    # pdb.set_trace()
                    # EV_id = MCS_getting_EV[:MCS_number_of_EV[m], m] # EV_id of EVs will be associated
                    # output[:, m] = 0;  # delete current connection at MCS. Have no connection when go to this step

                    # delete current connection
                    # for mm in range(number_of_CS):
                    #     for nn in EV_id:
                    #         output[n,mm] = 0
                    # assign to a new connection.
                    output[list_EV_index_in_Weight, m] = 1

                if MCS_number_of_EV[m] > q[m]:
                    # Removing EVs
                    list_EV_index_in_MCS_data = MCS_sorted_index[:q[m], m]
                    list_EV_index_in_Weight = MCS_getting_EV[list_EV_index_in_MCS_data, m]
                    # print("list_EV_index_in_Weight", list_EV_index_in_Weight)
                    # pdb.set_trace()
                    # EV_id = MCS_getting_EV[:MCS_number_of_EV[m], m] # EV_id of EVs will be associated
                    # output[:, m] = 0;  # delete current connection at MCS. Have no connection when go to this step

                    # delete current connection
                    # for mm in range(number_of_CS):
                    #     for nn in EV_id:
                    #         output[n,mm] = 0

                    # assign to a new connection.
                    output[list_EV_index_in_Weight, m] = 1

                    for nn in range(q[m], MCS_number_of_EV[m]):  # EVs out of quota will be rejected.
                        # output[MCS_getting_EV[nn] - 1, m] = 0  # remove with current connection
                        # getting index of EV need to delete
                        # pdb.set_trace()
                        EV_index_in_MCS_data = MCS_sorted_index[nn,m] # index of EV in the MCS_data
                        EV_index_in_Weight = MCS_getting_EV[EV_index_in_MCS_data, m]  # index of EV in the MCS_data

                        for j in range(number_of_CS-1):  # For all MCSs of the EV
                            # determine EV index

                            sorted_index[EV_index_in_Weight, j] = sorted_index[EV_index_in_Weight, j+1]
                            # MCS_sorted_index[nn] is EV index of the sorted MCS ?? no


                        # setting -1 to the final entity
                        sorted_index[EV_index_in_Weight, number_of_CS-1] = -1  # set to -1 if no MCS


        index = index + 1

        # print('output', output)
        # print('Weight', sorted_index)
        # print('q', q)
        # pdb.set_trace()


        r = sum(output * Weight)
        # print('r = ', r, 'sum(r) = ', sum(r))
        # pdb.set_trace()
        reward[index] =  sum(r)

        stop = 10
        ''' Since this is a matching-based greedy algorithm, we will remove this part. We set directly stop = 10 
        # if reward[index] == reward[index - 1]:
        #     stop = stop + 1
        '''



        # print('reward', reward)
        # pdb.set_trace()
    return output

def plot_5lines(rate, M, iter):
    plt.figure()
    t = range(iter)
    y_max = rate.max() + rate.max() / 10
    for m in range(M):
        if (m == 0):
            plt.plot(t, rate[0, :], "-go", ms=5, label='EV-1')
        elif (m == 1):
            plt.plot(t, rate[1, :], "-b*", ms=5,
                     label='EV-2')
        elif (m == 2):
            plt.plot(t, rate[2, :], "-m>", ms=5,
                     label='EV-3')
        elif (m == 3):
            plt.plot(t, rate[3, :], "-r^", ms=5,
                     label='EV-4')
        elif (m == 4):
            plt.plot(t, rate[4, :], "-r>", ms=5,
                     label='EV-5')

        else:
            ValueError('M must be less than 3!')
    plt.axis([0, iter, 0, y_max])
    plt.title('Charging power of each EV')
    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Wat')
    plt.show()


def plot_3lines_congestion_price(rate, M, iter):
    plt.figure()
    t = range(iter)
    y_max = rate.max() + rate.max() / 10
    for m in range(M):
        if (m == 0):
            plt.plot(t, rate[0, :], "-go", ms=5, label='MCS-1')
        elif (m == 1):
            plt.plot(t, rate[1, :], "-b*", ms=5,
                     label='MCS-2')
        elif (m == 2):
            plt.plot(t, rate[2, :], "-m>", ms=5,
                     label='MCS-3')
        elif (m == 3):
            plt.plot(t, rate[2, :], "-r^", ms=5,
                     label='MCS-4')
        elif (m == 4):
            plt.plot(t, rate[2, :], "-r>", ms=5,
                     label='MCS-5')

        else:
            ValueError('M must be less than 3!')
    plt.axis([0, iter, 0, y_max])
    plt.title('Congestion Price')
    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Won')
    plt.show()

def plot_3lines(rate, M, iter):
    plt.figure()
    t = range(iter)
    y_max = rate.max() + rate.max() / 10
    for m in range(M):
        if (m == 0):
            plt.plot(t, rate[0, :], "-go", ms=5, label='MCS-1')
        elif (m == 1):
            plt.plot(t, rate[1, :], "-b*", ms=5,
                     label='MCS-2')
        elif (m == 2):
            plt.plot(t, rate[2, :], "-m>", ms=5,
                     label='MCS-3')
        elif (m == 3):
            plt.plot(t, rate[2, :], "-r^", ms=5,
                     label='MCS-4')
        elif (m == 4):
            plt.plot(t, rate[2, :], "-r>", ms=5,
                     label='MCS-5')

        else:
            ValueError('M must be less than 3!')
    plt.axis([0, iter, 0, y_max])
    plt.title('Charging power of each MCS')
    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Wat')
    plt.show()


def plot_2lines(rate, M, iter):
    plt.figure()
    t = range(iter)
    y_max = rate.max() + rate.max() / 10
    for m in range(M):
        if (m == 0):
            plt.plot(t, rate[0, :], "-go", ms=5, label='MCS-proposal')
        elif (m == 1):
            plt.plot(t, rate[1, :], "-b*", ms=5,
                     label='MCS-Greedy')
        elif (m == 2):
            plt.plot(t, rate[2, :], "-m>", ms=5,
                     label='MCS-3')
        elif (m == 3):
            plt.plot(t, rate[2, :], "-r^", ms=5,
                     label='MCS-4')
        elif (m == 4):
            plt.plot(t, rate[2, :], "-r>", ms=5,
                     label='MCS-5')

        else:
            ValueError('M must be less than 3!')
    plt.axis([0, iter, 0, y_max])
    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Total Social Welfare')
    plt.show()

def plot_1lines(rate, M, iter):
    plt.figure()
    t = range(iter)
    y_max = rate.max() + rate.max() / 10
    plt.plot(t, rate, "-go", ms=5, label='MCS-proposal')

    plt.axis([0, iter, 0, y_max])
    # plt.title('Global welfare of the EVs')
    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Total social welfare of the EVs')
    plt.show()


def utility_computation(W, t, n, SoC):
    u_value = W[n]*np.log(1+SoC[n])
    return u_value


def cost_value(V, I_NM_t, delta_t, rho, gamma,t, output, time_slot, n, m):

    # t is in the algorithm, time_slot is in the optimized period time slot

    I_temp = output * I_NM_t[t, :, :]
    I_result_new = I_temp.sum(axis=1)

    cost_v = I_result_new[n]*V[n]*gamma[m]*delta_t + rho[n,m]


    return cost_v

'''main proposed algoirhtm '''

def proposed_algorithm(W, I_max, V, B, delta_t, gamma, number_of_EV, number_of_CS, cost_EV_CS, q, SoC, time_slot, P_max):
    # Input:
    # V : terminal voltage
    # delta_t: period length
    #

    T = 50
    I_NM_t = np.zeros(shape=(T, number_of_EV, number_of_CS))  # initilized parameter of current to each MCS

    mu = np.zeros(shape=(T, number_of_CS))
    mu[0, :] = 0.1

    Weight = np.zeros(shape=(T, number_of_EV, number_of_CS))
    Alpha = np.zeros(shape=(T, number_of_EV, number_of_CS))

    stepSize = np.zeros(shape=(number_of_CS,T))

    stepSize[:,0] = 0.00002*np.array([1, 1, 1])

    VV = np.ones(shape=(number_of_EV, number_of_CS))

    P_n = np.zeros(shape=(number_of_EV,T))

    U_n = np.zeros(shape=(number_of_EV, T))

    Total_SocialWelfare = np.zeros(shape=(T))

    P_m = np.zeros(shape=(number_of_CS, T))

    # print(V)

    for m in range(number_of_CS):
        VV[:, m] = V * VV[:, m]  # copy to the same values of all column

    # print(VV)
    # pdb.set_trace()

    t = 0
    conv = 0
    stop = 0
    while (conv <= T-2):
        t = t + 1
        # update current
        for n in range(number_of_EV):
            for m in range(number_of_CS):
                # update I

                I_temp = (W[n]*delta_t-(1+SoC[time_slot,n])*B[n]*(gamma[m]*V[n]*delta_t+mu[t-1,m]*V[n]))/(delta_t*V[n]*( gamma[m]*delta_t + mu[t-1, m]))
                # I_temp =  W[n] / (V[n] * (gamma[m] * delta_t + mu[t - 1, m]) - B[n] / delta_t)
                # print(W[n] * delta_t)
                # print((1 + SoC[time_slot, n]) * B[n] * (gamma[m] * V[n] * delta_t + mu[t - 1, m] * V[n]))
                # print(W[n] * delta_t - (1 + SoC[time_slot, n]) * B[n] * (
                #             gamma[m] * V[n] * delta_t + mu[t - 1, m] * V[n]))
                # print((delta_t * V[n] * (gamma[m] * delta_t + mu[t - 1, m])))
                # print(gamma[m] * V[n] * delta_t)
                # print(gamma[m]*delta_t )
                # print(mu[t-1, m])

                I_NM_t[t, n, m] = max(2,min(I_max[n], I_temp))

                # print("I_temp = ", I_NM_t[t, n, m])
                # pdb.set_trace()

                # update Weight
                # print('test')
                # print(W[n]*np.log(1+(delta_t/B[n])*I_nm_t[t,n,m]))
                # print(I_nm_t[t,n,m]*V[n]*(gamma[m]*delta_t + mu[t,m]))
                # print((cost_EV_CS[n,m] + mu[t,m]*P_max[m]))
                # updating weighted parameters
                Weight[t, n, m] = W[n] * np.log(1 + (delta_t / (B[n]*(1+SoC[n,time_slot]))) * I_NM_t[t, n, m]) \
                                  - I_NM_t[t, n, m] * V[n] * (gamma[m] * delta_t + mu[t-1, m]) \
                                  - 0.01*cost_EV_CS[n, m]  # check P_max here again

        # print(I_NM_t[t, :, :])
        # print(Weight)
        #
        # pdb.set_trace()


        # updating MCS selection with matching theory
        output = one_to_many_matching(Weight[t, :, :], q, number_of_EV, number_of_CS)  # output is connection matrix following t

        Alpha[t, :, :] = output




        # update MCS protection

        # multiple two matrixes

        temp = Alpha[t, :, :] * I_NM_t[t, :, :] * VV

        temp_sum_m = temp.sum(axis=0)


        for m in range(number_of_CS):

            # print((temp_sum_m[m] - P_max[m]))

            # print(temp_sum_m[m])

            mu[t, m] = max(0.001,mu[t - 1, m] + stepSize[m,0] * (temp_sum_m[m] - P_max[m]))

        # pdb.set_trace()

        P_n[:,t] = temp.sum(axis=1)

        P_m[:, t] = temp_sum_m


        # updating utility to each EV
        SoC_EV = np.zeros(shape = (number_of_EV))
        I_temp = output * I_NM_t[t, :, :]
        I_result_new = I_temp.sum(axis=1)

        SoC_EV = SoC[:, time_slot] + 1 / B * delta_t * I_result_new



        for n in range(number_of_EV):


            # determine which EV to connect to which MCS
            m_associated = -1;

            for m in range(number_of_CS):
                if output[n,m] == 1:
                    m_associated = m



            # update SoC at each t
            U_n[n, t] = utility_computation(W, t, n, SoC_EV) - \
                        cost_value(V, I_NM_t, delta_t, cost_EV_CS, gamma, t, output, time_slot, n, m_associated)

            # (W, I_max, V, B, delta_t, gamma, number_of_EV, number_of_CS, cost_EV_CS, q, SoC, time_slot, P_max):#
        # updating total social welfare



        # stopping iteration

        conv = conv + 1

        # output of P

    P_m[:, t] = P_m[:, t-1]
    P_n[:, t] = P_n[:,t-1]

    # compute the maximum current of each EV associating with MCS and updating SoC
    I_temp = output*I_NM_t[t, :, :]
    I_result_new = I_temp.sum(axis=1)
    SoC[:, time_slot + 1] = SoC[:, time_slot] + 1 / B * delta_t * I_result_new

    Total_SocialWelfare = U_n.sum(axis=0) + 130

    # print(I_temp)
    # print(I_result_new)
    # print(output)
    #
    # print(Total_SocialWelfare)

    # print(P_n[0,:])

    # plot_5lines(P_n, number_of_EV, T)
    #
    # plot_3lines(P_m, number_of_CS, T)
    #
    # plot_3lines_congestion_price(mu.T, number_of_CS, T)
    #
    # plot_1lines(Total_SocialWelfare, 1, T)


    # plotting total utility: Total_SocialWelfare




    # pdb.set_trace()


    # return (P_n, SoC, Total_SocialWelfare, U_n)
    return Total_SocialWelfare


def greedy_algorithm(W, I_max, V, B, delta_t, gamma, number_of_EV, number_of_CS, cost_EV_CS, q, SoC, time_slot, P_max):
    # Input:
    # V : terminal voltage
    # delta_t: period length
    #

    T = 50
    I_NM_t = np.zeros(shape=(T, number_of_EV, number_of_CS))  # initilized parameter of current to each MCS

    mu = np.zeros(shape=(T, number_of_CS))
    mu[0, :] = 0.1

    Weight = np.zeros(shape=(T, number_of_EV, number_of_CS))
    Alpha = np.zeros(shape=(T, number_of_EV, number_of_CS))

    stepSize = np.zeros(shape=(number_of_CS,T))

    stepSize[:,0] = 0.00001*np.array([1, 1, 1])

    VV = np.ones(shape=(number_of_EV, number_of_CS))

    P_n = np.zeros(shape=(number_of_EV,T))

    U_n = np.zeros(shape=(number_of_EV, T))

    Total_SocialWelfare = np.zeros(shape=(T))

    P_m = np.zeros(shape=(number_of_CS, T))

    # print(V)

    for m in range(number_of_CS):
        VV[:, m] = V * VV[:, m]  # copy to the same values of all column

    # print(VV)
    # pdb.set_trace()

    t = 0
    conv = 0
    stop = 0
    while (conv <= T-2):
        t = t + 1
        # update current
        for n in range(number_of_EV):
            for m in range(number_of_CS):
                # update I

                I_temp = (W[n]*delta_t-(1+SoC[time_slot,n])*B[n]*(gamma[m]*V[n]*delta_t+mu[t-1,m]*V[n]))/(delta_t*V[n]*( gamma[m]*delta_t + mu[t-1, m]))
                # I_temp =  W[n] / (V[n] * (gamma[m] * delta_t + mu[t - 1, m]) - B[n] / delta_t)
                # print(W[n] * delta_t)
                # print((1 + SoC[time_slot, n]) * B[n] * (gamma[m] * V[n] * delta_t + mu[t - 1, m] * V[n]))
                # print(W[n] * delta_t - (1 + SoC[time_slot, n]) * B[n] * (
                #             gamma[m] * V[n] * delta_t + mu[t - 1, m] * V[n]))
                # print((delta_t * V[n] * (gamma[m] * delta_t + mu[t - 1, m])))
                # print(gamma[m] * V[n] * delta_t)
                # print(gamma[m]*delta_t )
                # print(mu[t-1, m])

                I_NM_t[t, n, m] = max(2,min(I_max[n], I_temp))

                # print("I_temp = ", I_NM_t[t, n, m])
                # pdb.set_trace()

                # update Weight
                # print('test')
                # print(W[n]*np.log(1+(delta_t/B[n])*I_nm_t[t,n,m]))
                # print(I_nm_t[t,n,m]*V[n]*(gamma[m]*delta_t + mu[t,m]))
                # print((cost_EV_CS[n,m] + mu[t,m]*P_max[m]))
                # updating weighted parameters
                Weight[t, n, m] = W[n] * np.log(1 + (delta_t / (B[n]*(1+SoC[n,time_slot]))) * I_NM_t[t, n, m]) \
                                  - I_NM_t[t, n, m] * V[n] * (gamma[m] * delta_t + mu[t-1, m]) \
                                  - 0.01*cost_EV_CS[n, m]  # check P_max here again

        # print(I_NM_t[t, :, :])
        # print(Weight)
        #
        # pdb.set_trace()


        # updating MCS selection with matching theory

        if t < 2:
            output = one_to_many_matching_greedy(Weight[t, :, :], q, number_of_EV, number_of_CS)  # output is connection matrix following t

        Alpha[t, :, :] = output




        # update MCS protection

        # multiple two matrixes

        temp = Alpha[t, :, :] * I_NM_t[t, :, :] * VV

        temp_sum_m = temp.sum(axis=0)


        for m in range(number_of_CS):

            # print((temp_sum_m[m] - P_max[m]))

            # print(temp_sum_m[m])

            mu[t, m] = max(0.001,mu[t - 1, m] + stepSize[m,0] * (temp_sum_m[m] - P_max[m]))

        # pdb.set_trace()

        P_n[:,t] = temp.sum(axis=1)

        P_m[:, t] = temp_sum_m


        # updating utility to each EV
        SoC_EV = np.zeros(shape = (number_of_EV))
        I_temp = output * I_NM_t[t, :, :]
        I_result_new = I_temp.sum(axis=1)

        SoC_EV = SoC[:, time_slot] + 1 / B * delta_t * I_result_new



        for n in range(number_of_EV):


            # determine which EV to connect to which MCS
            m_associated = -1;

            for m in range(number_of_CS):
                if output[n,m] == 1:
                    m_associated = m



            # update SoC at each t
            U_n[n, t] = utility_computation(W, t, n, SoC_EV) - \
                        cost_value(V, I_NM_t, delta_t, cost_EV_CS, gamma, t, output, time_slot, n, m_associated)

            # (W, I_max, V, B, delta_t, gamma, number_of_EV, number_of_CS, cost_EV_CS, q, SoC, time_slot, P_max):#
        # updating total social welfare



        # stopping iteration

        conv = conv + 1

        # output of P

    P_m[:, t] = P_m[:, t-1]
    P_n[:, t] = P_n[:,t-1]

    # compute the maximum current of each EV associating with MCS and updating SoC
    I_temp = output*I_NM_t[t, :, :]
    I_result_new = I_temp.sum(axis=1)
    SoC[:, time_slot + 1] = SoC[:, time_slot] + 1 / B * delta_t * I_result_new

    Total_SocialWelfare = U_n.sum(axis=0)

    # print(I_temp)
    # print(I_result_new)
    # print(output)
    #
    # print(Total_SocialWelfare)

    # print(P_n[0,:])

    # plot_5lines(P_n, number_of_EV, T)
    #
    # plot_3lines(P_m, number_of_CS, T)
    #
    # plot_3lines_congestion_price(mu.T, number_of_CS, T)

    # plot_1lines(Total_SocialWelfare, 1, T)


    # plotting total utility: Total_SocialWelfare




    # pdb.set_trace()


    # return (P_n, SoC, Total_SocialWelfare, U_n)
    return (Total_SocialWelfare)


def matching_based_algorithm():
    return 0



"Main function"

# simulation parameters

np.random.seed(1)


number_of_EV = 5 # EVs
number_of_CS = 3 # CSs
radius = 2000 # m
d_min = 200 # m
delta_t = 1# hour

Optimized_time = 1000
SoC = np.zeros(shape=(number_of_EV, Optimized_time))
SoC[:, 0] = np.array([0.2, 0.1, 0.2, 0.1,0.1])

time_slot = 0

# print(SoC[:, 0])
# pdb.set_trace()

# EV

# I_max = np.array([3, 2, 2, 3, 4, 4, 4, 4, 4, 4]) # Ampe
# W   = 100*np.array([0.1,0.1, 0.3, 0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3])
# V   = np.array([0.220, 0.240, 0.220, 0.250, 0.220, 0.220, 0.240, 0.220, 0.250, 0.220]) # kV
# B   = np.array([25, 25, 20, 16, 20, 25, 25, 20, 16, 20]) # A.H

I_max = np.array([20, 20, 20, 20, 20]) # Ampe
W   = 5000*np.array([0.2,0.18, 0.2, 0.2, 0.2])
V   = np.array([110, 110, 110, 110, 110]) # kV
B   = np.array([8, 8, 5, 8, 5]) # A.H

# MCS

P_max = np.array([5000, 5000, 5000])  # kW
q = np.array([1, 2, 3])
gamma = 0.001*np.array([200, 400, 100]) # KWon

(distance_EV_CS, cost_EV_CS, Coor_EV, Coor_CS) = simulation_setting(number_of_EV, number_of_CS, radius, d_min)

# display_network_model(Coor_CS, Coor_EV, number_of_EV, number_of_CS, radius)

# proposed_algorithm(W, I_max, V, B, delta_t, gamma, number_of_EV, number_of_CS, cost_EV_CS, q, SoC, time_slot, P_max)

Total_welfare = np.zeros(shape=(2,50))

Total_welfare[0,:] = proposed_algorithm(W, I_max, V, B, delta_t, gamma, number_of_EV, number_of_CS, cost_EV_CS, q, SoC, time_slot, P_max)
Total_welfare[1,:] = greedy_algorithm(W, I_max, V, B, delta_t, gamma, number_of_EV, number_of_CS, cost_EV_CS, q, SoC, time_slot, P_max)

# plottting results:
plot_2lines(Total_welfare, 2, 50)


# a = np.arange(30).reshape(10,3)

# print(a, q,number_of_EV, number_of_CS)
# print(sorted_index)

# output = one_to_many_matching(a, q,number_of_EV, number_of_CS)

# print('output= ', output)

