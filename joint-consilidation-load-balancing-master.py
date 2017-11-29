

class Datacenter():

    # initialize the object
    def __init__(self, lamb):
        # array of servers in the data-center
        self.servers = []
        # input request rate of data-center
        self.lamb = lamb

    # add a server to the data-center
    def add_server(self, server):
        self.servers.append(server)

    # get array of active servers in the data-center
    def active_servers(self):
        active_servers = []

        for server in self.servers:
            if server.active == 1:
                active_servers.append(server)

        return active_servers

    # number of servers in data-center
    def size(self):
        return len(self.servers)

    # TODO: calculate cost of data-center
    def cost(self):
        return self.power() + self.delay()

    # TODO: calculate power consumption of data-center
    def power(self):
        power = 0

        for server in self.servers:
            if server.active == 1:
                power += server.e_i + (server.e_p - server.e_i) * server.utilization()

        return self.alpha * power

    # TODO: calculate delay of data-center
    def delay(self):
        delay = 0

        for server in self.servers:
            if server.active == 1:
                delay += server.lamb / (server.mu - (server.lamb / server.k))

        return self.teta * delay
  
  
  import sys
import math
import random
from Solver import Solver


class Jcl():

    # initialize the object
    def __init__(self, datacenter):
        # number of servers in data-center
        self.m = len(datacenter.servers)
        self.datacenter = datacenter

        # converts delay to monetary
        self.teta = 0.01
        # converts power to monetary
        self.alpha = 0.01
        # tunable smoothing parameter
        self.delta = 10
        # stopping criteria
        self.epsilon = 0.01

        # best answer
        self.c = sys.maxint - 1
        # next step's best answer
        self.c_prime = sys.maxint

        # number of iterations that algorithm iterates
        self.iterations = 10

        # these variables used for percentage of servers that should be active
        self.initial_percentage = 0
        self.percentage_step = 0.001

    # main body of JCL algorithm
    def optimize(self):
        iterations = self.iterations

        while (abs(self.c - self.c_prime) > self.epsilon) and iterations != 0:
            # 1. choose m_prime randomly
            percentage = self.initial_percentage

            while True:
                self.random_activate_servers(self.datacenter.servers, percentage)

                if self.feasible(self.datacenter):
                    break
                else:
                    percentage += self.percentage_step

            # 2. find lambda_i for minimum c_prime
            solver = Solver(self.datacenter.servers, self.alpha, self.teta, self.datacenter.lamb)
            lambs = solver.solve(False)

            # print(lambs)
            # print("C = " + str(solver.c_prime))
            # print(lambs)

            iterations -= 1

            # 3. if we don't have a better answer, stay in current state
            if solver.c_prime > self.c or solver.c_prime < 0:
                continue

            # 4. with probability 1-p stay in current state
            if random.uniform(0, 1) < 1 - self.transition_probability():
                continue

            # 5. with probability p go to next step
            self.c_prime = self.c
            self.c = solver.c_prime

            for index, server in enumerate(self.datacenter.servers):
                server.lamb = lambs[index]

    # find random subset of servers and make them active
    # make others inactive
    def random_activate_servers(self, servers, percentage):
        for server in servers:
            if random.uniform(0, 1) < percentage:
                server.activate()
            else:
                server.de_active()

    # probability that JCL algorithm moves to next step (p)
    def transition_probability(self):
        delta = self.delta

        try:
            return math.exp(delta * self.c_prime) / (math.exp(delta * self.c_prime) + math.exp(delta * self.c))
        except:
            return 1

    # check if with this set of activated servers
    # all input traffic can be processed
    def feasible(self, datacenter):
        total_capacity = 0

        for server in datacenter.servers:
            if server.active == 1:
                total_capacity += server.mu * server.k

        return datacenter.lamb <= total_capacity
 
 import random


class Server(object):

    # initialize the object
    def __init__(self):
        # number of virtual machines in server
        self.k = random.randint(4, 10)
        # service rate of server
        self.mu = random.randint(3, 10)
        # idle power of server
        self.e_i = random.randint(200, 250)
        # peak power of server
        self.e_p = random.randint(400, 500)
        # is server activated or not
        self.active = 0
        # input request rate of server
        self.lamb = 0

    # utilization of server
    def utilization(self):
        return self.lamb / (self.k * self.mu)

    # activate the server
    def activate(self):
        self.active = 1

    # de-activate the server
    def de_active(self):
        self.active = 0

    # activate the server
    def set_active(self, active):
        self.active = active

    # set input request rate of server
    def set_lambda(self, lamb):
        self.lamb = lamb
        
from scipy.optimize import minimize


class Solver(object):
    # initialize the object
    def __init__(self, servers, alpha, teta, lambda_s):
        self.c_prime = 0

        self.teta = teta
        self.alpha = alpha
        self.lambda_s = lambda_s

        # fill servers specification in arrays for solver use
        self.k = []
        self.a = []
        self.mu = []
        self.lambs = []
        self.bound = []
        self.active = []

        for server in servers:
            self.a.append((server.e_p - server.e_i) / server.mu * server.k)
            self.k.append(server.k)
            self.mu.append(server.mu)
            self.lambs.append(server.lamb)
            self.active.append(server.active)

    # solve sub-problem
    def solve(self, linear):
        if linear:
            return self.linear_solver()
        else:
            return self.non_linear_solver()

    # implementation of linear solver
    # this method uses cvxopt solver
    # visit http://cvxopt.org for more information
    def linear_solver(self):
        print("Initiating linear solver")

    # implementation of non-linear solver
    # this method uses scipy solver
    # visit http://docs.scipy.org for more information
    def non_linear_solver(self):
        print("Initiating non-linear solver")

        for active in self.active:
            print(active)

        cons = [{'type': 'eq', 'fun': self.workload_preserving_constraint}]
        cons += self.inactive_server_constraint(self.lambs)
        cons += self.positive_variables(self.lambs)

        res = minimize(self.objective_function, self.lambs, method='SLSQP', bounds=None, constraints=tuple(cons))

        print(res.x)

        self.c_prime = res.fun

        return res.x

    # definition of objective function
    def objective_function(self, x):
        objective = 0

        for index, lamb in enumerate(x):
            if lamb == 0:
                continue
            objective += self.teta * (lamb / (self.mu[index] - lamb / self.k[index])) + self.alpha * (
                self.a[index] * lamb)

        return objective

    # definition of workload preserving constraint
    def workload_preserving_constraint(self, x):
        constraint = 0

        for index, lamb in enumerate(x):
            constraint += lamb

        constraint -= self.lambda_s

        return constraint

    # definition of inactive server constraint using "Big M" method
    def inactive_server_constraint(self, x):
        constraints = []

        for index, lamb in enumerate(x):
            if self.active[index] == 0:
                continue
            constraints.append({'type': 'ineq', 'fun': lambda x: self.lambda_s - x[index]})

        return constraints

    # all variables must be positive
    def positive_variables(self, x):
        constraints = []

        for index, lamb in enumerate(x):
            if self.active[index] == 0:
                constraints.append({'type': 'eq', 'fun': lambda x: x[index]})
            else:
                constraints.append({'type': 'ineq', 'fun': lambda x: x[index]})

        return constraints

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize

fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2

cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

bnds = ((0, None), (0, None))

res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds, constraints=cons)

print(res)



import random
from Jcl import Jcl
from Datacenter import Datacenter
from Server import Server
import time


start = time.time()
print("Algorithm started.")


m = 10
lamb = random.randint(50, 150)

datacenter = Datacenter(lamb)

# make some servers
for i in range(0, m):
    server = Server()
    # server.set_active(random.randint(0, 1))
    server.set_active(1)
    datacenter.add_server(server)

# turn on some of the servers
for server in datacenter.servers:
    if server.active:
        server.set_lambda(datacenter.lamb / len(datacenter.active_servers()))

jcl = Jcl(datacenter)

jcl.optimize()


end = time.time()

print("Algorithm ended. Time spent: " + str(end - start) + " (ms)")
        
        
        
        
