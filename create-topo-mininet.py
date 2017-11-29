
import numpy as np

import random

# imput procedure for liner topology  
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.util import irange,dumpNodeConnections
from mininet.log import setLogLevel

# input for controller architecture. 


from mininet.node import CPULimitedHost


from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import Link, Intf, TCLink
from mininet.util import dumpNodeConnections
import logging
import os 

import pandas as pd

# input for full mesh architecture

from mininet.node import OVSSwitch

# 1. functions to create full mesh architecture: 

"""
Ref. : https://gist.github.com/lantz/7853026
OVS Bridge with Spanning Tree Protocol
Note: STP bridges don't start forwarding until
after STP has converged, which can take a while!
See below for a command to wait until STP is up.
Example:
mn -v output --custom torus2.py --switch ovs-stp --topo torus,3,3
mininet> sh time bash -c 'while ! ovs-ofctl show s1x1 | grep FORWARD; do sleep 1; done'
"""

class randomTopo(Topo):
   "Full topology of k switches, with one host per switch."
   def __init__(self, n=2,  alpha = 0.5, maxCPU = 90, minCPU = 10, maxBW = 1000, minBW = 10, **opts):	
		# k: number of switches (and hosts). Each switch has a host		    
        df = pd.read_csv('mininet_link.csv')
	df = df.set_index(['index'])		

        dfnode = pd.read_csv('mininet_node.csv')
	dfnode = dfnode.set_index(['index'])		


   	super(randomTopo, self).__init__(**opts)
        self.n = n 
        self.maxCPU = maxCPU 
        self.minCPU = minCPU 
        self.maxBW = maxBW 
        self.minBW = minBW    
        switches = n   # total switchs

        self.alpha = alpha

        cons = n        # connections with next switch
        if cons >= switches:
               cons = switches - 1 # do we need change? No. Because for fullmesh, this will be activated. 
        hosts = 1      # nodes per switch

        # Create host and Switch
        # Add link :: host to switch
        for s_num in range(1,switches+1): # switch index 
               switch = self.addSwitch("s%s" %(s_num))  
	       print 'adding a new row' 
	       dfnode.loc[dfnode.index[-1] + 1] = [switch, 'NaN']
	       print 'continue to update the file '
	     
	       
	       #cmd = "ovs-vsctl set Bridge %s stp_enable=true" %str(s_num)
	       #os.system(cmd)

               for h_num in range(1,hosts+1):
    		       cpu_array =  np.random.uniform(minCPU, maxCPU,1)
                       cpu_percent = int(cpu_array[0]) # % cpu standard 
                       host = self.addHost("h%s" %(h_num + ((s_num - 1) * hosts)), cpu = cpu_percent)
		       print 'adding a new row' 
		       dfnode.loc[dfnode.index[-1] + 1] = ["h%s" %(h_num + ((s_num - 1) * hosts)), cpu_percent]
		       print 'continue to update the file '
		       
  
                       # 10 Mbps, 5ms delay, 2% loss, 1000 packet queue                      
                       bw_array =  np.random.uniform(minBW,maxBW,1)
                       bw_Mbps = int(bw_array[0])
                       print host, switch, bw_Mbps

                       self.addLink(host,switch, bw = bw_Mbps, delay='5ms', loss=2, max_queue_size=1000, use_htb=True)
		       print 'adding a new row' 
		       df.loc[df.index[-1] + 1] = [host, switch, bw_Mbps]
		       print 'continue to update the file '
		       


        # Add link :: switch to switch
        for src in range(1,switches+1):
               for c_num in range(1,cons+1):
                       dst = src + c_num
                       if dst <= switches:
                               flip_state = self.flip(alpha)
                               
                               if flip_state =='H':
                               	       print("Adding link: ", "s%s" %src,"s%s" %dst)
				       # 10 Mbps, 5ms delay, 2% loss, 1000 packet queue                      
				       bw_array =  np.random.uniform(minBW,maxBW,1)
				       bw_Mpbs = int(bw_array[0])
				       self.addLink("s%s" %src,"s%s" %dst,bw = bw_Mpbs, delay='5ms', loss=2, max_queue_size=1000, use_htb=True)
				       df.loc[df.index[-1] + 1] = ["s%s" %src, "s%s" %dst, bw_Mpbs]
                               	       # self.addLink("s%s" %src,"s%s" %dst)
                               else:
                               	       print("flip = T. This edge link is not added. Go to next candidated link.")

                       elif dst > switches:
                               dst = dst - switches
                               if src - dst > cons:
		                       if flip_state =='H':
		                       	       print("Adding link: ", "s%s" %src,"s%s" %dst)
					       # 10 Mbps, 5ms delay, 2% loss, 1000 packet queue                      
					       bw_array =  np.random.uniform(minBW,maxBW,1)
					       bw_Mpbs = int(bw_array[0])
					       self.addLink("s%s" %src,"s%s" %dst, bw = bw_Mpbs, delay='5ms', loss=2, max_queue_size=1000, use_htb=True)
					       df.loc[df.index[-1] + 1] = ["s%s" %src, "s%s" %dst, bw_Mpbs]
		                       	       # self.addLink("s%s" %src,"s%s" %dst)
		                       else:
		                       	       print("flip = T. This edge link is not added. Go to next candidated link.")
        df.to_csv('mininet_link.csv')
        dfnode.to_csv('mininet_node.csv')
        



   def flip(self, p):
   	return 'H' if random.random() < p else 'T'

class FullTopo(Topo):
    "Single switch connected to sigle host."
    "n switches are connected following StarTopo."
    def build( self, n=2, maxCPU = 90, minCPU = 10, maxBW = 1000, minBW = 10):
  	df = pd.read_csv('mininet_link.csv')
  	df = df.set_index(['index'])
	
  	dfnode = pd.read_csv('mininet_node.csv')
  	dfnode = dfnode.set_index(['index'])


        self.n = n 
        self.maxCPU = maxCPU 
        self.minCPU = minCPU 
        self.maxBW = maxBW 
        self.minBW = minBW      
        
        
        cons = n # connections with next switch
        if cons >=n: 
        	cons = n-1 
	hosts = 1 # hosts per switch 
	"Adding Switches"   
	for s_num in range(1,n+1):
		switch = self.addSwitch("s%s" %(s_num))
		dfnode.loc[dfnode.index[-1] + 1] = [switch, 'NaN']
		cmd = "ovs-vsctl set Bridge %s stp_enable=true" %str(s_num)
		os.system(cmd)
		print cmd 
               	for h_num in range(1,hosts+1):
			cpu_array =  np.random.uniform(minCPU, maxCPU,1)
			cpu_percent = int(cpu_array[0]) # % cpu standard 
                        ihost = (h_num + ((s_num - 1) * hosts))
                       	host = self.addHost("h%s" %ihost, cpu = cpu_percent/100)
                        dfnode.loc[dfnode.index[-1] + 1] = ["h%s" %ihost, cpu_percent]
			"Adding links"
			# bw Mbps, 5ms delay, 0.01% loss, 1000 packet queue
			bw_array =  np.random.uniform(minBW,maxBW,1)
			bw_Mpbs = int(bw_array[0])
			self.addLink(host,switch, bw=bw_Mpbs, delay='5ms',loss=0.01,max_queue_size=1000, use_htb=True )
		   	df.loc[df.index[-1] + 1] = [host, switch, bw_Mpbs]
		   	

	"Add link :: switch to switch"
       	for src in range(1,n+1):
		for c_num in range(1,cons+1):
		       	dst = src + c_num
		       	if dst <= n:
				bw_array =  np.random.uniform(minBW,maxBW,1)
				bw_Mpbs = int(bw_array[0])
				print("s%s" %src,"s%s" %dst)
				self.addLink("s%s" %src,"s%s" %dst, bw = bw_Mpbs, delay='5ms',loss=0.01,max_queue_size=1000, use_htb=True )
				df.loc[df.index[-1] + 1] = ["s%s" %src, "s%s" %dst, bw_Mpbs]

		       	else:
		               	dst = dst - n
		               	if src - dst > cons:
					bw_array =  np.random.uniform(minBW,maxBW,1)
					bw_Mpbs = int(bw_array[0])
					print("s%s" %src,"s%s" %dst)
		                       	self.addLink("s%s" %src,"s%s" %dst, bw = bw_Mpbs, delay='5ms',loss=0.01,max_queue_size=1000, use_htb=True )
				        df.loc[df.index[-1] + 1] = ["s%s" %src, "s%s" %dst, bw_Mpbs]


       	df.to_csv('mininet_link.csv')
	dfnode.to_csv('mininet_node.csv')




class StarTopo( Topo ):
    	"Single switch connected to n hosts."
	def build( self, n=2, maxCPU = 90, minCPU = 10, maxBW = 1000, minBW = 10):
		df = pd.read_csv('mininet_link.csv')
		df = df.set_index(['index'])
	  	dfnode = pd.read_csv('mininet_node.csv')
	  	dfnode = dfnode.set_index(['index'])

		#super(LinearTopo, self).__init__(**opts)

		switch = self.addSwitch( 's1' )
		dfnode.loc[dfnode.index[-1] + 1] = [switch, 'NaN']
                self.n = n 
                self.maxCPU = maxCPU 
                self.minCPU = minCPU 
                self.maxBW = maxBW 
                self.minBW = minBW      
  
		for h in range(n):
			# Each host h gets h% of system CPU
		        cpu_array =  np.random.uniform(minCPU, maxCPU,1)
		        cpu_percent = int(cpu_array[0]) # % cpu standard 

			host = self.addHost( 'h%s' % (h + 1),
				         cpu= cpu_percent/100)
                        dfnode.loc[dfnode.index[-1] + 1] = [host, cpu_percent]

                        logger.debug("Start writing host data to file")
                        print 'host = ', host

			# bw Mbps, 5ms delay, 0.01% loss, 1000 packet queue
		        bw_array =  np.random.uniform(minBW,maxBW,1)
		        bw_Mpbs = int(bw_array[0])
		    
			self.addLink( host, switch, bw=bw_Mpbs, delay='5ms', loss=0.01,
		                  max_queue_size=1000, use_htb=True )
                           
			df.loc[df.index[-1] + 1] = [host, switch, bw_Mpbs]

		df.to_csv('mininet_link.csv')
		dfnode.to_csv('mininet_node.csv')

class LinearTopo(Topo):
   "Linear topology of k switches, with one host per switch."

   def __init__(self, k=1, maxCPU = 90, minCPU = 10, maxBW = 1000, minBW = 10, **opts):
       """Init.
           k: number of switches
           h: number of hosts 
           hconf: host configuration options
           lconf: link configuration options"""
       df = pd.read_csv('mininet_link.csv')
       df = df.set_index(['index'])
       dfnode = pd.read_csv('mininet_node.csv')
       dfnode = dfnode.set_index(['index'])

       super(LinearTopo, self).__init__(**opts)

       self.k = k
       self.maxCPU = maxCPU 
       self.minCPU = minCPU 
       self.maxBW = maxBW 
       self.minBW = minBW       

       lastSwitch = None
       for i in irange(1, k):
	   # Each host h gets h% of system CPU
	   cpu_array =  np.random.uniform(minCPU, maxCPU,1)
	   cpu_percent = int(cpu_array[0]) # % cpu standard 


           host = self.addHost( 'h%s' % i, cpu= cpu_percent/100)
           dfnode.loc[dfnode.index[-1] + 1] = ['h%s' % i, cpu_percent]
           switch = self.addSwitch('s%s' % i)
           dfnode.loc[dfnode.index[-1] + 1] = [switch, 'NaN']
      	   bw_array =  np.random.uniform(minBW,maxBW,1)
	   bw_Mpbs = int(bw_array[0])
	   self.addLink( host, switch,  bw=bw_Mpbs, delay='5ms', loss=0.01,
		                  max_queue_size=1000, use_htb=True )
	   df.loc[df.index[-1] + 1] = [host, switch, bw_Mpbs]

           if lastSwitch:
      	   	bw_array =  np.random.uniform(minBW,maxBW,1)
	   	bw_Mpbs = int(bw_array[0])
               	self.addLink(switch,lastSwitch, bw=bw_Mpbs, delay='5ms', loss=0.01,
		                  max_queue_size=1000, use_htb=True )
		print 'continue to update the file '
		df.loc[df.index[-1] + 1] = [switch, lastSwitch, bw_Mpbs]




           lastSwitch = switch

       df.to_csv('mininet_link.csv')
       dfnode.to_csv('mininet_node.csv')

# CREATE CONTROLLER Topology

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger( __name__ )




class ControllerTopo(Topo):
    logger.debug("ControllerTopo with Flat Tree Topology")
    CoreSwitchList = []
    AggSwitchList = []
    EdgeSwitchList = []
    HostList = []
    def __init__(self, iNUMBER = 4, **opts):
        """Init.
           k: number of switches
           h: number of hosts 
           hconf: host configuration options
           lconf: link configuration options"""

        super(ControllerTopo, self).__init__(**opts)

        self.iNUMBER = iNUMBER
        self.iCoreLayerSwitch = iNUMBER
        self.iAggLayerSwitch = iNUMBER * 2
        self.iEdgeLayerSwitch = iNUMBER * 2
        self.iHost = self.iEdgeLayerSwitch * 2 
 
    
        #Init Topo
        Topo.__init__(self)

    def createTopo(self):    
        logger.debug("Start create Core Layer Swich")
        self.createCoreLayerSwitch(self.iCoreLayerSwitch)
        logger.debug("Start create Agg Layer Swich ")
        self.createAggLayerSwitch(self.iAggLayerSwitch)
        logger.debug("Start create Edge Layer Swich ")
        self.createEdgeLayerSwitch(self.iEdgeLayerSwitch)
        logger.debug("Start create Host")
        self.createHost(self.iHost)

    """
    Create Switch and Host
    """

    def createCoreLayerSwitch(self, NUMBER):
        logger.debug("Create Core Layer")
        for x in range(1, NUMBER+1):
            PREFIX = "100"
            if x >= int(10):
                PREFIX = "10"
            self.CoreSwitchList.append(self.addSwitch(PREFIX + str(x)))

    def createAggLayerSwitch(self, NUMBER):
        logger.debug( "Create Agg Layer")
        for x in range(1, NUMBER+1):
            PREFIX = "200"
            if x >= int(10):
                PREFIX = "20"
            self.AggSwitchList.append(self.addSwitch(PREFIX + str(x)))

    def createEdgeLayerSwitch(self, NUMBER):
        logger.debug("Create Edge Layer")
        for x in range(1, NUMBER+1):
            PREFIX = "300"
            if x >= int(10):
                PREFIX = "30"
            self.EdgeSwitchList.append(self.addSwitch(PREFIX + str(x)))
    
    def createHost(self, NUMBER):
        logger.debug("Create Host")
        for x in range(1, NUMBER+1):
            PREFIX = "400"
            if x >= int(10):
                PREFIX = "40"
            self.HostList.append(self.addHost(PREFIX + str(x))) 

    """
    Create Link 
    """
    def createLink(self):
        df = pd.read_csv('mininet_link.csv')
        df = df.set_index(['index'])

        logger.debug("Create Core to Agg")
        for x in range(0, self.iAggLayerSwitch, 2):
   	    bw_array =  np.random.uniform(100,1000,1)
	    bw_Mpbs = int(bw_array[0])

            # self.addLink(self.CoreSwitchList[0], self.AggSwitchList[x], bw=bw_Mpbs, loss=5)

            self.addLink(self.CoreSwitchList[0],self.AggSwitchList[x], bw=bw_Mpbs, delay='5ms', loss=0.01,
		                  max_queue_size=1000, use_htb=True )	    
            df.loc[df.index[-1] + 1] = [self.CoreSwitchList[0], self.AggSwitchList[x], bw_Mpbs]


   	    bw_array =  np.random.uniform(100,1000,1)
	    bw_Mpbs = int(bw_array[0])
            self.addLink(self.CoreSwitchList[1], self.AggSwitchList[x], bw_Mpbs=1000, loss=5)
            df.loc[df.index[-1] + 1] = [self.CoreSwitchList[1], self.AggSwitchList[x], bw_Mpbs]



        for x in range(1, self.iAggLayerSwitch, 2):
   	    bw_array =  np.random.uniform(100,1000,1)
	    bw_Mpbs = int(bw_array[0])
            self.addLink(self.CoreSwitchList[2], self.AggSwitchList[x], bw_Mpbs=1000, loss=5)           
	    df.loc[df.index[-1] + 1] = [self.CoreSwitchList[2], self.AggSwitchList[x], bw_Mpbs] 


   	    bw_array =  np.random.uniform(100,1000,1)
	    bw_Mpbs = int(bw_array[0])
            self.addLink(self.CoreSwitchList[3], self.AggSwitchList[x], bw=bw_Mpbs, loss=5)
	    df.loc[df.index[-1] + 1] = [self.CoreSwitchList[3], self.AggSwitchList[x], bw_Mpbs] 


        
        logger.debug("Create Agg to Edge")
        for x in range(0, self.iAggLayerSwitch, 2):
   	    bw_array =  np.random.uniform(100,1000,1)
	    bw_Mpbs = int(bw_array[0])
            self.addLink(self.AggSwitchList[x], self.EdgeSwitchList[x], bw=bw_Mpbs, loss=5)
	    df.loc[df.index[-1] + 1] = [self.AggSwitchList[x], self.EdgeSwitchList[x], bw_Mpbs] 

   	    bw_array =  np.random.uniform(100,1000,1)
	    bw_Mpbs = int(bw_array[0])
            self.addLink(self.AggSwitchList[x], self.EdgeSwitchList[x+1], bw=bw_Mpbs)
	    df.loc[df.index[-1] + 1] = [self.AggSwitchList[x], self.EdgeSwitchList[x+1], bw_Mpbs] 

   	    bw_array =  np.random.uniform(100,1000,1)
	    bw_Mpbs = int(bw_array[0])
            self.addLink(self.AggSwitchList[x+1], self.EdgeSwitchList[x], bw=bw_Mpbs)
	    df.loc[df.index[-1] + 1] = [self.AggSwitchList[x+1], self.EdgeSwitchList[x], bw_Mpbs] 


   	    bw_array =  np.random.uniform(100,1000,1)
	    bw_Mpbs = int(bw_array[0])
            self.addLink(self.AggSwitchList[x+1], self.EdgeSwitchList[x+1], bw=bw_Mpbs)
	    df.loc[df.index[-1] + 1] = [self.AggSwitchList[x+1], self.EdgeSwitchList[x+1], bw_Mpbs] 

        logger.debug("Create Edge to Host")
        for x in range(0, self.iEdgeLayerSwitch):
            ## limit = 2 * x + 1 
   	    bw_array =  np.random.uniform(10,100,1)
	    bw_Mpbs = int(bw_array[0])
            self.addLink(self.EdgeSwitchList[x], self.HostList[2 * x], bw = bw_Mpbs)
            df.loc[df.index[-1] + 1] = [self.EdgeSwitchList[x], self.HostList[2*x], bw_Mpbs] 

   	    bw_array =  np.random.uniform(10,100,1)
	    bw_Mpbs = int(bw_array[0])
            self.addLink(self.EdgeSwitchList[x], self.HostList[2 * x + 1])
	    df.loc[df.index[-1] + 1] = [self.EdgeSwitchList[x], self.HostList[2*x+1], bw_Mpbs] 


        df.to_csv('mininet_link.csv')
        dfnode.to_csv('mininet_node.csv')

def enableSTP():
    """
    //HATE: Dirty Code
    """
    for x in range(1,5):
        cmd = "ovs-vsctl set Bridge %s stp_enable=true" % ("100" + str(x))
        os.system(cmd)
        print cmd 

    for x in range(1, 9):
        cmd = "ovs-vsctl set Bridge %s stp_enable=true" % ("200" + str(x))
        os.system(cmd)  
        print cmd 
        cmd = "ovs-vsctl set Bridge %s stp_enable=true" % ("300" + str(x))
        os.system(cmd)
        print cmd

def iperfTest(net, topo):
    logger.debug("Start iperfTEST")
    h1000, h1015, h1016 = net.get(topo.HostList[0], topo.HostList[14], topo.HostList[15])
    
    #iperf Server
    h1000.popen('iperf -s -u -i 1 > iperf_server_differentPod_result', shell=True)

    #iperf Server
    h1015.popen('iperf -s -u -i 1 > iperf_server_samePod_result', shell=True)

    #iperf Client
    h1016.cmdPrint('iperf -c ' + h1000.IP() + ' -u -t 10 -i 1 -b 100m')
    h1016.cmdPrint('iperf -c ' + h1015.IP() + ' -u -t 10 -i 1 -b 100m')


def pingTest(net):
    logger.debug("Start Test all network")
    net.pingAll()

def createControllerTopo(iNumber):
    logging.debug("LV1 Create HugeTopo")
    topo = ControllerTopo()
    topo.createTopo() 
    topo.createLink() 
    
    logging.debug("LV1 Start Mininet")
    CONTROLLER_IP = "127.0.0.1"
    CONTROLLER_PORT = 6633
    net = Mininet(topo=topo, link=TCLink, controller=NaN)
    net.addController( 'controller',controller=RemoteController,ip=CONTROLLER_IP,port=CONTROLLER_PORT)
    net.start()

    logger.debug("LV1 dumpNode")
    enableSTP()
    dumpNodeConnections(net.hosts)
    
    pingTest(net)
    # iperfTest(net, topo)
    

    CLI(net)
    net.stop()

def returnmaxmin():

	df = pd.read_csv('mininet_link.csv')
        df = df.set_index(['index'])
        print df

        # max link weight: 
	
        
        df = df.drop(df.index[[0]])

        print df

        maxlw = df['Weight'].max()
        print 'max link weight = ', maxlw
        minlw = df['Weight'].min()
        print 'min link weight = ', minlw

        dfnode = pd.read_csv('mininet_node.csv')
        dfnode = dfnode.set_index(['index'])
        print dfnode        

        dfnode = dfnode.drop(dfnode.index[[0]])
        dfnode = dfnode.dropna()

        print dfnode
    
        maxnw = dfnode['Weight'].max()
        print 'max link weight = ', maxnw
        minnw = dfnode['Weight'].min()
        print 'min link weight = ', minnw

        return maxlw, minlw, maxnw, minnw



def choosingTopo():
   "Create and test a network in Question 3"
   
   # choosing max_CPU of host: 
   #max_CPU = raw_input("Enter max_CPU value (%): ")   
   #print "you entered", s
   #s = int(s)

   # choosing min_CPU of host: 

   maxCPU = 90 # %
   minCPU = 10 # %
   maxBW = 1000 # Mbps
   minBW = 10 # Mbps 
    

   topology = 'NaN'
   nodes = 0;   
   alpha = -1; 
   nodeMin = 0
   nodeMax = 0
   linkMin = 0
   linkMax = 0
 
   # choosing topo here: 
   s = raw_input("Choosing topo type index to create: 1.datacenter 2.linear 3.full 4.star 5.random: ")   
   print "you entered", s
   s = int(s)
 
   if s==1: 
	print "\n You selected topo type: datacenter"
        logger.debug("Meaning of iNumber: ")
        logger.debug("self.iNUMBER = iNUMBER") 
        logger.debug("self.iAggLayerSwitch = iNUMBER * 2") 
        logger.debug("self.iEdgeLayerSwitch = iNUMBER * 2")
        logger.debug("self.iHost = self.iEdgeLayerSwitch * 2") 
 
        iNUMBER = raw_input("\n Enter Core iNUMBER: iNumber = ")
        iNUMBER = int(iNUMBER); 
                 
        if iNUMBER < 4:
	    print 'Error!!! n_core >= 4'
	    exit()		      
       
        createControllerTopo(iNUMBER)

        nodes = iNUMBER + iNUMBER*2 + iNUMBER*2 + iNUMBER*4
        topology = 'datacenter'

        linkMax, linkMin, nodeMax, nodeMin = returnmaxmin()




 
   elif s==2: 
	print "\n You selected topo type: linear"
        topology = 'linear'
        print "\n In this topo type, number of hosts = number of swiths = k" 
	k = raw_input("\n How many nodes you want to create? k = ") 
        k = int(k)
        if k>=4: 
	    topo = LinearTopo( k, maxCPU, minCPU, maxBW, minBW)
            nodes = 2*k # number of nodes in the virtual network. 
        else: 
	    print 'Error!!! K >= 4'
	    exit()
        net = Mininet( topo=topo,
	           host=CPULimitedHost, link=TCLink )
	net.start()
	print "Dumping host connections"
	dumpNodeConnections(net.hosts)
	print "Testing network connectivity"
	net.pingAll()
        print "Testing bandwidth between h1 and h4"
        h1, h4 = net.get( 'h1', 'h4' )
        net.iperf( (h1, h4) )

        #print "Testing bandwidth between h1 and controller c0"
        #h1, c0 = net.get( 'h1', 'c0' )
        #net.iperf( (h1, c0) )

	net.stop()
	linkMax, linkMin, nodeMax, nodeMin = returnmaxmin()


   elif s==3: 
	print "\n You selected topo type: full"
        topology = 'full'
	k = raw_input("\n How many nodes you want to create? k = ") 
        k = int(k)
        if k>=4: 
            topo = FullTopo(k, maxCPU, minCPU, maxBW, minBW)
	    # topo = FullTopo(k)
            nodes = 2*k # number of nodes in the virtual network. 
        else: 
	    print 'Error!!! k >= 4'
	    exit()
        net = Mininet( topo=topo,
	           host=CPULimitedHost, link=TCLink )
	net.start()
	print "Dumping host connections"
	dumpNodeConnections(net.hosts)
	print "Testing network connectivity"
	# net.pingAll()
        print "Testing bandwidth between h1 and h4"
        # h1, h4 = net.get( 'h1', 'h4' )
        # net.iperf( (h1, h4) )

        #print "Testing bandwidth between h1 and controller c0"
        #h1, c0 = net.get( 'h1', 'c0' )
        #net.iperf( (h1, c0) )

	net.stop()
	linkMax, linkMin, nodeMax, nodeMin = returnmaxmin()


   elif s==4: 
	print "\n You selected topo type: star"
        topology = 'star'

	k = raw_input("\n In this topo, hosts connect to a switch. How many hosts you want to create? h = ") 
        k = int(k)
        if k>=4: 
            topo = StarTopo(k, maxCPU, minCPU, maxBW, minBW)
	    # topo = FullTopo(k)
            nodes = k+1 # number of nodes in the virtual network. 
        else: 
	    print 'Error!!! k >= 4'
	    exit()
        net = Mininet( topo=topo,
	           host=CPULimitedHost, link=TCLink)
	net.start()
	print "Dumping host connections"
	dumpNodeConnections(net.hosts)
	print "Testing network connectivity"
	net.pingAll()
        print "Testing bandwidth between h1 and h4"
        h1, h4 = net.get( 'h1', 'h4' )
        net.iperf( (h1, h4) )

        #print "Testing bandwidth between h1 and controller c0"
        #h1, c0 = net.get( 'h1', 'c0' )
        #net.iperf( (h1, c0) )

	net.stop()
	linkMax, linkMin, nodeMax, nodeMin = returnmaxmin()


   elif s==5: 
	print "\n You selected topo type: random"
        topology = 'random'
     

	k = raw_input("\n In this topo, a host connects to a switch. Switches are connected random topology. How many switchs you want to create? k = ") 
        k = int(k)
        if k>=4: 
     	     alpha = raw_input("\n Enter the bias of the random coin. Alpha value in range [0,1]. Alpha = ") 
	     alpha = float(alpha)
	     if  alpha == 0:
                topology = 'random, alpha = 0.' 
	     elif alpha == 1:
		print 'This topology converted to a fullmesh topology.'
                topology = '(random, alpha = 1) --> full'    
	     elif alpha == 0.5: 
                print 'Perfect parameter.'	 

	     elif (alpha > 0) & (alpha < 1) & (alpha != 0.5):
		print 'Perfect parameter.'

	     elif alpha < 0 | alpha > 1: 
		print 'Error!!! range(k) = [0,1]'
		exit()    

             topo = randomTopo(k, alpha, maxCPU, minCPU, maxBW, minBW) 
             nodes = 2*k # number of nodes in the virtual network. 
        else: 
	     print 'Error!!! k >= 4'
	     exit()

        net = Mininet( topo=topo,
	           host=CPULimitedHost, link=TCLink)
        
	net.start()
	print "Dumping host connections"
	dumpNodeConnections(net.hosts)
	print "Testing network connectivity"
	net.pingAll()
        print "Testing bandwidth between h1 and h4"
        h1, h4 = net.get( 'h1', 'h4' )
        #net.iperf( (h1, h4) )

        #print "Testing bandwidth between h1 and controller c0"
        #h1, c0 = net.get( 'h1', 'c0' )
        #net.iperf( (h1, c0) )

	net.stop()
	linkMax, linkMin, nodeMax, nodeMin = returnmaxmin()
   else:
	print "\n Error!!. Out of scope!"
        exit()
   
   # return information
   print " nodes: ", nodes
   print " topology: ", topology
   if alpha != -1: 
	print " alpha: ", alpha
   else: 
        print " alpha: xxx (irrelevant)"
   print " node-min: ",nodeMin
   print " node-max: ",nodeMax
   print " link-min: ",linkMin
   print " link-max: ",linkMax


if __name__ == '__main__':
   # Tell mininet to print useful information
	setLogLevel('info')

	data = [('Soure', ['init']),
		 ('Destination', ['init']),
		 ('Weight', ['init']),
		 ]

	datanode = [('Node', ['init']),
		    ('Weight', ['init']),		 		 
		  ]


    
	df = pd.DataFrame.from_items(data)
        df.index.name = 'index'

        # os.remove('mininet_link.csv')
       
        print ("write to the file")
        out_csv = 'mininet_link.csv'
        df.to_csv(out_csv)


	dfnode = pd.DataFrame.from_items(datanode)
        dfnode.index.name = 'index'
	# os.remove('mininet_node.csv')
	print ("write to the file")
	out_csv_node = 'mininet_node.csv'
        dfnode.to_csv(out_csv_node)
        

	choosingTopo()

        df = pd.read_csv('mininet_link.csv')
        df = df.set_index(['index'])
        #print df

        # max link weight: 
	
        
        df = df.drop(df.index[[0]])

        print df

        #maxlw = df['Weight'].max()
        #print 'max link weight = ', maxlw
        #minlw = df['Weight'].min()
        #print 'min link weight = ', minlw

        dfnode = pd.read_csv('mininet_node.csv')
        dfnode = dfnode.set_index(['index'])
        # print dfnode        

        dfnode = dfnode.drop(dfnode.index[[0]])
        dfnode = dfnode.dropna()

        print dfnode
    
        #maxnw = dfnode['Weight'].max()
        #print 'max link weight = ', maxnw
        #minnw = dfnode['Weight'].min()
        #print 'min link weight = ', minnw
