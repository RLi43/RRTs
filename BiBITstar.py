#!/usr/bin/env python3
#coding:utf-8

### ---------------
# BIBITSTAR - biBiT* - bidirectional Batch Informed Tree 
# Author: RLi
### ---------------

import copy
import math
import platform
import random
import time
import numpy as np
import queue

import matplotlib.pyplot as plt

show_animation = True

def _dis(x1,x2):
    return np.linalg.norm(np.array(x1)-np.array(x2))
    

def radius(q):
    # zeta = math.pi**(self.dimension/2)/GAMMA_N[self.dimension-1]
    # lamb = math.pi**(self.dimension/2)/GAMMA_N[self.dimension-1]*cMax/2*(math.sqrt(cMax**2-self.cMin**2)/2)**(self.dimension-1)
    # return self.eita*(math.log(q)/q*lamb/zeta)**(1/self.dimension)
    return 30.0 * math.sqrt((math.log(q) / q))

# map
class Map:
    def __init__(self,dim=2,obs_num=20,obs_size_max=3,xinit=[0,0],xgoal=[23,23],randMax=[29,29],randMin=[-3,-3]):
        self.dimension = dim
        self.xinit = xinit
        self.xgoal = xgoal
        self.randMax = randMax
        self.randMin = randMin
        self.obstacles = []
        self.DISCRETE = 0.05

        # obstacles
        for i in range(obs_num):
            #TODO
            ob = []
            for j in range(dim):
                ob.append(random.random()*20+1.5)
            ob.append(random.random()*obs_size_max+0.2)
            self.obstacles.append(ob)

        # informed        
        self.cMin = _dis(self.xinit,self.xgoal)
        self.xCenter = (np.array(xinit)+np.array(xgoal))/2
        a1 = np.transpose([(np.array(xgoal)-np.array(xinit))/self.cMin])        
        # first column of idenity matrix transposed
        id1_t = np.array([1.0]+[0.0,]*(self.dimension-1)).reshape(1,self.dimension)
        M = a1 @ id1_t
        U, S, Vh = np.linalg.svd(M, 1, 1)
        self.C = np.dot(np.dot(U, 
            np.diag([1.0,]*(self.dimension-1)+[np.linalg.det(U) * np.linalg.det(np.transpose(Vh))]))
            , Vh)

    def collision(self,x):
        for ob in self.obstacles:
            if _dis(x,ob[:-1])<=ob[-1]:
                return True
        return False

    def collisionLine(self,x1,x2):
        dis = _dis(x1,x2)
        if dis<self.DISCRETE:
            return False
        nums = int(dis/self.DISCRETE)
        direction = (np.array(x2)-np.array(x1))/_dis(x1,x2)
        for i in range(nums+1):
            x = np.add(x1 , i*self.DISCRETE*direction)
            if self.collision(x): return True
        if self.collision(x2): return True
        return False


    def randomSample(self):
        x = []
        for j in range(self.dimension):
            x.append(random.random()*(self.randMax[j]-self.randMin[j])+self.randMin[j])
        return x
    def freeSample(self):
        x = self.randomSample()
        while self.collision(x):
            x = self.randomSample()
        return x
    def informedSample(self,cMax):
        L = np.diag([cMax/2]+[math.sqrt(cMax**2-self.cMin**2)/2,]*(self.dimension-1))
        cl = np.dot(self.C,L)
        x = np.dot(cl,self.ballSample())+self.xCenter
        while self.collision(x):
            x = np.dot(cl,self.ballSample())+self.xCenter
        return list(x)
    def ballSample(self):
        ret = []
        for i in range(self.dimension):
            ret.append(random.random()*2-1)
        ret = np.array(ret)
        return ret/np.linalg.norm(ret)*random.random()

    
    def drawMap(self):
        if self.dimension==2:
            plt.clf()
            sysstr = platform.system()
            if(sysstr =="Windows"):
                scale = 18
            elif(sysstr == "Linux"):
                scale = 24
            else: scale = 24
            for (ox, oy, size) in self.obstacles:
                plt.plot(ox, oy, "ok", ms=scale * size)
            
            plt.plot(self.xinit[0],self.xinit[1], "xr")
            plt.plot(self.xgoal[0],self.xgoal[1], "xr")
            plt.axis([self.randMin[0],self.randMax[0],self.randMin[1],self.randMax[1]])
            plt.grid(True)

### main algorithm

## main Class
class BiBITstar(object):
    def __init__(self,_map,maxIter =300, bn=10,connFactor=0.0):
        self.map = _map
        self.batchSize = bn
        self.maxIter = maxIter
        self.bestCost = float('inf')
        self.bestConn = None

        self.x = [_map.xinit,_map.xgoal] # store all the point(samples, vertices)
        self.r = float('inf')

        self.qe = queue.PriorityQueue() # [vertex, xvertex],ecost
        self.qv = queue.PriorityQueue()
        self.vold = []
        self.Gtree = [0]
        self.Htree = [1]
        self.X_sample = []
        self.isGtree = {0:True,1:False} 
        self.cost = {0:0,1:0}
        self.conn = {}
        self.parent = {}
        # self.children = {}
        self.depth ={0:0,1:0}

        self.qv.put((self.distance(0,1),0))
        self.qv.put((self.distance(0,1),1))

        # if show_animation:
        #     self.map.drawMap()
    
    def vAdd(self,x,h=False):
        if h:
            self.Htree.append(x)
        else:
            self.Gtree.append(x)
        self.X_sample.remove(x)
        self.qv.put((self.vertexQueueValue(x),x))
    def qeAdd(self,v,x):
        self.qe.put((self.edgeQueueValue(v,x),[v,x]))

    def bestInQv(self):
        vc,vm = self.qv.get()
        self.qv.put((vc,vm))
        return (vc,vm)
    def bestInQe(self):
        ec,[wm,xm] = self.qe.get()
        self.qe.put((ec,[wm,xm]))
        return (ec,[wm,xm])

    def getCost(self,ind):
        try:
            cost = self.cost[ind]
            return cost
        except:
            return float('inf')

    def vertexQueueValue(self,v):
        # vn = self.nearest(v,not self.isGtree[v])
        # return self.distance(v,vn)
        return self.cost[v] + self.costTjHeuristicVertex(v)
    def edgeQueueValue(self,w,x):
        # vn = self.nearest(x,self.isGtree[w])
        # return self.distance(w,x) + self.distance(x,vn)
        return self.cost[w] + self.distance(w,x) + self.costTgHeuristic(x,self.isGtree[w])

    def lowerBoundHeuristicEdge(self,v,x):
        return self.costFgHeuristic(v,not self.isGtree[v]) + \
                    self.costFgHeuristic(x, self.isGtree[v]) + \
                        self.distance(v,x)
    def lowerBoundHeuristicVertex(self,x):
        return self.costFgHeuristic(x,True) + self.costFgHeuristic(x,False)        
    def costFgHeuristic(self,x,h=False):
        if h: target = 1
        else: target = 0
        return self.distance(target,x)

    def costTgHeuristic(self,ind,h=False):
        # if h:
        #     Vnearest = self.nearest(ind,False)
        # else:
        #     Vnearest = self.nearest(ind,True)
        Vnearest = self.nearest(ind,not h)
        return self.cost[Vnearest] + self.distance(Vnearest, ind) 

    def costTjHeuristicVertex(self,vertex,i=False):
        if i:
            return self.getCost(vertex)
        else:
            return self.costTgHeuristic(vertex,self.isGtree[vertex])
    def costTjVertex(self,vertex,i=False):
        if i:
            return self.getCost(vertex)
        else:
            try:
                vcon = self.conn[vertex]
                return self.cost[vcon] + self.distance(vertex,vcon)
            except:
                return float('inf')

    def nearest(self,indx, inGtree):
        if inGtree:
            theTree = self.Gtree
        else:
            theTree = self.Htree

        nearestDis = float('inf')
        vn = None
        for v in theTree:
            dis = self.distance(v,indx)
            if dis < nearestDis:
                vn = v
                nearestDis = dis
        return vn

    def distance(self,ind1,ind2):
        return np.linalg.norm(np.array(self.x[ind1]) - np.array(self.x[ind2]))

    def solve(self):
        for iterateNum in range(self.maxIter):
            print("iter: ",iterateNum)
            if show_animation:
                self.drawGraph()
            if self.isEmpty():
                print("newBatch")
                self.newBatch()
            else:
                ec,[wm,xm] = self.qe.get()
                if self.lowerBoundHeuristicEdge(wm,xm) > self.bestCost:
                    # end it.
                    while not self.qe.empty():
                        self.qe.get()
                    while not self.qv.empty():
                        vc,v = self.qv.get()
                        self.vold.append(v)
                    continue
                
                ## ExpandEdge
                if self.collisionEdge(wm,xm):
                    continue
                # it's a simple demo, we don't care too much about time-cost
                # if there's no collision, we add this edge.
                trueEdgeCost = self.distance(wm,xm)
                # if wm.inGtree:
                #     theTree = self.Gtree
                # else: theTree = self.Htree
                try:
                    isG = self.isGtree[xm]                    
                    # same tree
                    if isG == self.isGtree[wm]:
                        if self.cost[wm] + trueEdgeCost >= self.cost[xm]:
                            continue
                        self.parent[xm] = wm
                        self.cost[xm] = self.cost[wm] + trueEdgeCost # has not update the children TODO
                        self.depth[xm] = self.depth[wm] + 1

                    # another tree
                    else:
                        try:
                            wcon = self.conn[wm]
                            if self.cost[wcon] + self.distance(wm,wcon) <= \
                                self.cost[xm] + self.distance(wm,xm):
                                continue
                            self.conn.pop(wcon)
                        except:
                            pass
                        try:
                            xcon = self.conn[xm]                        
                            if self.cost[xcon] + self.distance(xm,xcon) <= \
                                self.cost[wm] + self.distance(xm,wm):
                                continue
                            self.conn.pop(xcon)
                        except:
                            pass
                        # update or create one                        
                        self.conn[wm] = xm
                        self.conn[xm] = wm
                        newCost = self.cost[wm] + self.cost[xm] + self.distance(wm,xm)
                        if newCost < self.bestCost:
                            self.bestCost = newCost
                            # report?
                            self.bestConn = [wm,xm]
                            if self.bestCost == self.map.cMin:
                                break
                # v->sample
                except:
                    self.parent[xm] = wm
                    self.isGtree[xm] = self.isGtree[wm]
                    self.cost[xm] = self.cost[wm] + trueEdgeCost
                    self.depth[xm] = self.depth[wm] + 1
                    self.vAdd(xm,not self.isGtree[xm])

        if self.bestCost == float('inf'):
            print("plan failed")
        else:
            if show_animation:
                path = self.getPath()       
                plt.plot([self.x[ind][0] for ind in path], [self.x[ind][1] for ind in path], '-o') 
                plt.show() 
            print("plan finished with cost: ",self.bestCost)
                           
    ## ---
    # while BestQueueValue(Qv) <= BestQueueValue(Qe):
    #     ExpandVertex(BestValueIn(Qv))
    def isEmpty(self):
        while not self.qv.empty():
            if self.qe.empty():
                self.expandVertex()
            else:
                vcost,vm = self.bestInQv()
                ecost,[wm,xm] = self.bestInQe()
                if(ecost>=vcost):
                    self.expandVertex()
                else:
                    break
        while self.qe.empty() and not self.qv.empty():
            self.expandVertex()

        return self.qe.empty()

    # f_hat(v,x) < bestCost
    def edgeInsertConditionSample(self,v,xind):
        return  self.lowerBoundHeuristicEdge(v,xind) < self.bestCost
    # f_hat(v,x) < bestCost AND (better solution)
    # Ti_hat(v) + c(v,x) < Ti(x) (optimal tree)
    def edgeInsertConditionSameTree(self,v,iv):
        if (v!=0 and v!=1):
            if self.parent[v] == iv:
                return False
        if (iv!=0 and iv!=1):
            if self.parent[iv] == v:
                return False

        costTargetHeuristic = self.costFgHeuristic(v,not self.isGtree[v]) + \
                                self.distance(v,iv)
        return costTargetHeuristic < self.cost[iv] and \
                self.costFgHeuristic(iv, self.isGtree[v]) + \
                    costTargetHeuristic < self.bestCost 
    def edgeInsertConditionAnotherTree(self,v,jv):
        cvx = self.distance(v,jv)
        if self.costFgHeuristic(v,not self.isGtree[v]) + \
                self.costFgHeuristic(jv, self.isGtree[v]) + \
                    cvx >= self.bestCost:
            return False
        # if is better than current connEdge
        try:
            vcon = self.conn[v]
            if vcon == jv or self.cost[jv] + cvx > self.cost[vcon] + self.distance(vcon,v):
                return False
        except:
            pass
        try:
            jcon = self.conn[jv]
            if jcon == v or self.cost[v] + cvx > self.cost[jcon] + self.distance(jcon,jv):
                return False
        except:
            pass
        return True



    def expandVertex(self):
        (vcost,v) = self.qv.get()
        self.vold.append(v)
        if self.lowerBoundHeuristicVertex(v) > self.bestCost:
            while not self.qv.empty():
                vc,v = self.qv.get()
                self.vold.append(v)
        else:
            ## expand vertex
            # expand to free sample
            xnearby = self.nearby(v,self.X_sample)
            for xind in xnearby:
                if self.edgeInsertConditionSample(v,xind):
                    self.qeAdd(v,xind)
            # expand to tree
            if self.isGtree[v]:
                iTree = self.Gtree
                jTree = self.Htree
            else:
                iTree = self.Htree
                jTree = self.Gtree
            # expand to the same tree
            ## TODO delay rewire?
            inear = self.nearby(v,iTree)
            for iv in inear:
                if self.edgeInsertConditionSameTree(v,iv):
                    self.qeAdd(v,iv)
            # expand to another tree
            jnear = self.nearby(v,jTree)
            for jv in jnear:
                if self.edgeInsertConditionAnotherTree(v,jv):
                    # TODO if there's no solution, should we give some award?
                    self.qeAdd(v,jv)

    """
    return nearby(self.r) x in thelist
    """
    def nearby(self,vind,thelist):
        near = []
        for ind in thelist: # 太暴力…… 下次试试r近邻……
            if self.distance(ind,vind) < self.r:
                near.append(ind)
        return near

    def sample(self,c):
        if c == float('inf'):
            for i in range(self.batchSize):
                self.X_sample.append(len(self.x))
                self.x.append(self.map.freeSample())
        else:
            for i in range(self.batchSize):
                self.X_sample.append(len(self.x))
                self.x.append(self.map.informedSample(c))
    def collisionEdge(self,vind,xind):
        return self.map.collisionLine(self.x[vind],self.x[xind])

    def newBatch(self):
        self.prune()
        self.sample(self.bestCost)
        self.r = radius(len(self.x))
        while len(self.vold) > 0:
            v = self.vold.pop()
            self.qv.put((self.vertexQueueValue(v),v))
    
    # # update the cost of vertex (might be out of date because of rewire)
    # def updateCost(self,prune = False):


    def prune(self):
        pass
        # # if prune ...
        # if self.bestCost < float('inf'):
        #     self.updateCost(prune=True)
        #     for x in self.X_sample:
        #         if self.lowerBoundHeuristicVertex(x) > self.bestCost:
        #             self.X_sample.remove(x)


    def getPath(self):
        reversePath = []
        if self.isGtree[self.bestConn[0]]:
            vg = self.bestConn[0]
            vh = self.bestConn[1]
        else:
            vg = self.bestConn[1]
            vh = self.bestConn[0]
        curV = vg
        if vg != 0:
            while self.parent[curV] != 0:
                reversePath.append(curV)
                curV = self.parent[curV]
            reversePath.append(curV)
        
        # reverse
        path = [0]
        while len(reversePath)>0:
            path.append(reversePath.pop())

        curV = vh
        if vh != 1:
            while self.parent[curV] != 1:
                path.append(curV)
                curV = self.parent[curV]
            path.append(curV)
        path.append(1)

        return path

    def drawGraph(self):
        plt.clf()
        if self.map.dimension == 2:
            self.map.drawMap()
            for xind in self.X_sample:
                plt.plot(self.x[xind][0],self.x[xind][1],'ob')            

            for v in self.Gtree:
                plt.plot(self.x[v][0],self.x[v][1],'og')   
                if v != 0:          
                    plt.plot([self.x[v][0], self.x[self.parent[v]][0]], 
                        [self.x[v][1], self.x[self.parent[v]][1]], '-g')
            
            for v in self.Htree:
                plt.plot(self.x[v][0],self.x[v][1],'or')  
                if v != 1:        
                    plt.plot([self.x[v][0], self.x[self.parent[v]][0]], 
                        [self.x[v][1], self.x[self.parent[v]][1]], '-r')
            
            for vconn in self.conn.keys():
                plt.plot([self.x[vconn][0], self.x[self.conn[vconn]][0]], 
                    [self.x[vconn][1], self.x[self.conn[vconn]][1]], '-y')
                
        plt.pause(0.01)
        #plt.show()


if __name__ == '__main__':
    map2Drand = Map()
    bit = BiBITstar(map2Drand)
    # show map
    if show_animation:
        bit.map.drawMap()
        # plt.pause(10)
    start_time = time.time()
    bit.solve()
    print("time_use: ",time.time()-start_time)


        