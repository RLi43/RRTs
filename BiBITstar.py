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

        self.qe = queue.PriorityQueue() # ecost,[vtind, xind]
        self.qv = queue.PriorityQueue()
        self.vold = []
        self.Tree = [0,1]
        # self.Gtree = [0]
        # self.Htree = [1]
        self.X_sample = []
        # because python alway copy a vertex for me ... :(
        self.isGtree = [True,False] # accroding to the order in Tree
        self.cost = [0,0]           # accroding to the order in Tree
        self.parent = [None,None]   # accroding to the order in tree
        self.children = [[],[]]     # accroding to the order in Tree
        self.depth = [0,0]          # accroding to the order in Tree
        self.conn = {}

        self.qv.put((self.distance(0,1),0))
        self.qv.put((self.distance(0,1),1))

        # if show_animation:
        #     self.map.drawMap()
    
    def qeAdd(self,vt,x):
        self.qe.put((self.edgeQueueValue(vt,x),[vt,x]))

    def bestInQv(self):
        vc,vmt = self.qv.get()
        self.qv.put((vc,vmt))
        return (vc,vmt)
    def bestInQe(self):
        ec,[wmt,xm] = self.qe.get()
        self.qe.put((ec,[wmt,xm]))
        return (ec,[wmt,xm])

    def getCost(self,ind):
        try:
            tind = self.Tree.index(ind)
            cost = self.cost[tind]
            return cost
        except:
            return float('inf')

    def vertexQueueValue(self,v):
        # vn = self.nearest(v,not self.isGtree[v])
        # return self.distance(v,vn)
        return self.cost[v] + self.costTjHeuristicVertex(v)
    def edgeQueueValue(self,wt,x):
        # vn = self.nearest(x,self.isGtree[w]) # problem: can't update during a batch
        # return self.distance(w,x) + self.distance(x,vn)
        return self.cost[wt] + self.distance(self.Tree[wt],x) + self.costTgHeuristic(x,self.isGtree[wt])

    def lowerBoundHeuristicEdge(self,vt,x):
        return self.costFgHeuristic(self.Tree[vt],not self.isGtree[vt]) + \
                    self.costFgHeuristic(x, self.isGtree[vt]) + \
                        self.distance(self.Tree[vt],x)
    def lowerBoundHeuristicVertex(self,x):
        x = self.Tree[x]
        return self.costFgHeuristic(x,True) + self.costFgHeuristic(x,False) 
    def lowerBoundHeuristic(self,x):
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
        Vnearest,nearDis = self.nearest(ind,not h)
        return self.cost[Vnearest] + nearDis

    def costTjHeuristicVertex(self,vt,i=False):
        if i:
            return self.cost(vt)
        else:
            return self.costTgHeuristic(self.Tree[vt],self.isGtree[vt])
    def costTjVertex(self,vertex,i=False):
        if i:
            return self.getCost(vertex)
        else:
            try:
                vcon = self.conn[vertex]
                return self.cost[vcon] + self.distance(vertex,vcon)
            except:
                return float('inf')

    """return treeIndx"""
    def nearest(self,indx, inGtree):
        nearestDis = float('inf')
        vn = None
        for vt in range(len(self.Tree)):
            if(self.isGtree[vt] == inGtree):
                dis = self.distance(self.Tree[vt],indx)
                if dis < nearestDis:
                    vn = vt
                    nearestDis = dis
        return vn,nearestDis

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
                ec,[wmt,xm] = self.qe.get()
                wm = self.Tree[wmt]
                if self.lowerBoundHeuristicEdge(wmt,xm) > self.bestCost:
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
                try:
                    xmt = self.Tree.index(xm)
                    try:
                        isG = self.isGtree[xmt]                    
                        # same tree
                        ## delay rewire?
                        if isG == self.isGtree[wmt]:
                            if self.cost[wmt] + trueEdgeCost >= self.cost[xmt]:
                                continue
                            oldparent = self.parent[xmt]
                            self.children[oldparent].remove(xmt)
                            self.parent[xmt] = wmt
                            self.cost[xmt] = self.cost[wmt] + trueEdgeCost # has not update the children TODO
                            self.depth[xmt] = self.depth[wmt] + 1

                        # another tree
                        else:
                            try:
                                wcont = self.conn[wmt]
                                if self.cost[wcont] + self.distance(wm,self.Tree[wcont]) <= \
                                    self.cost[xmt] + self.distance(wm,xm):
                                    continue
                                self.conn.pop(wcont)
                            except:
                                pass
                            try:
                                xcont = self.conn[xmt]                        
                                if self.cost[xcont] + self.distance(xm,self.Tree[xcont]) <= \
                                    self.cost[wmt] + self.distance(xm,wm):
                                    continue
                                self.conn.pop(xcont)
                            except:
                                pass
                            # update or create one                        
                            self.conn[wmt] = xmt
                            self.conn[xmt] = wmt
                            newCost = self.cost[wmt] + self.cost[xmt] + self.distance(wm,xm)
                            if newCost < self.bestCost:
                                self.bestCost = newCost
                                # report?
                                self.bestConn = [wmt,xmt]
                                if self.bestCost == self.map.cMin:
                                    break
                    except:
                        raise Exception("unknow exception")
                # v->sample
                except:
                    xmt = len(self.Tree)
                    self.Tree.append(xm)
                    self.isGtree.append(self.isGtree[wmt])
                    self.parent.append(wmt)
                    self.children[wmt].append(xmt)
                    self.children.append([])
                    self.cost.append(self.cost[wmt] + trueEdgeCost)
                    self.depth.append(self.depth[wmt]+1)
                    self.X_sample.remove(xm)
                    self.qv.put((self.vertexQueueValue(xmt),xmt))

        if self.bestCost == float('inf'):
            print("plan failed")
        else:
            if show_animation:
                path = self.getPath()       
                plt.plot([self.x[ind][0] for ind in path], [self.x[ind][1] for ind in path], '-o') 
                plt.show() 
            print("plan finished with cost: ",self.bestCost)
        # print plan information
        gnum = 0
        for v in range(len(self.Tree)):
            if self.isGtree[v]:
                gnum += 1
        print("Plan Info:")
        print("total samples:",len(self.x),"Gtree:",gnum,"Htree:",len(self.Tree)-gnum)
        print("edge num:",len(self.parent),"pruned:",len(self.x)-len(self.X_sample)-len(self.Tree))
        ## TODO more informations?

    ## ---
    # while BestQueueValue(Qv) <= BestQueueValue(Qe):
    #     ExpandVertex(BestValueIn(Qv))
    def isEmpty(self):
        while not self.qv.empty():
            if self.qe.empty():
                self.expandVertex()
            else:
                vcost,vmt = self.bestInQv()
                ecost,[wmt,xm] = self.bestInQe()
                if(ecost>=vcost):
                    self.expandVertex()
                else:
                    break
        while self.qe.empty() and not self.qv.empty():
            self.expandVertex()

        return self.qe.empty()

    # f_hat(v,x) < bestCost
    def edgeInsertConditionSample(self,vt,xind):
        return  self.lowerBoundHeuristicEdge(vt,xind) < self.bestCost
    # f_hat(v,x) < bestCost AND (better solution)
    # Ti_hat(v) + c(v,x) < Ti(x) (optimal tree)
    def edgeInsertConditionSameTree(self,vt,ivt):
        if self.parent[vt] == ivt:
            return False
        if self.parent[ivt] == vt:
            return False
        v = self.Tree[vt]
        iv = self.Tree[ivt]
        costTargetHeuristic = self.costFgHeuristic(v,not self.isGtree[vt]) + \
                                self.distance(v,iv)
        return costTargetHeuristic < self.cost[ivt] and \
                self.costFgHeuristic(iv, self.isGtree[vt]) + \
                    costTargetHeuristic < self.bestCost 
    def edgeInsertConditionAnotherTree(self,vt,jvt):
        v = self.Tree[vt]
        jv = self.Tree[jvt]
        cvx = self.distance(v,jv)
        if self.costFgHeuristic(v,not self.isGtree[vt]) + \
                self.costFgHeuristic(jv, self.isGtree[vt]) + \
                    cvx >= self.bestCost:
            return False
        # if is better than current connEdge
        try:
            vcont = self.conn[vt]
            if vcont == jvt or self.cost[jvt] + cvx > self.cost[vcont] + self.distance(self.Tree[vcont],v):
                return False
        except:
            pass
        try:
            jcont = self.conn[jvt]
            if jcont == vt or self.cost[vt] + cvx > self.cost[jcont] + self.distance(self.Tree[jcont],jv):
                return False
        except:
            pass
        return True

    def expandVertex(self):
        (vcost,vt) = self.qv.get()
        self.vold.append(vt)
        if self.lowerBoundHeuristicVertex(vt) > self.bestCost:
            while not self.qv.empty():
                vc,vt = self.qv.get()
                self.vold.append(vt)
        else:
            ## expand vertex
            # expand to free sample
            v = self.Tree[vt]
            xnearby = self.nearby(v,self.X_sample)
            for xind in xnearby:
                if self.edgeInsertConditionSample(vt,xind):
                    self.qeAdd(vt,xind)

            ## expand to tree
            # expand to the same tree
            ## TODO delay rewire?
            inear = self.nearbyT(v,self.isGtree[vt])
            for ivt in inear:
                if self.edgeInsertConditionSameTree(vt,ivt):
                    self.qeAdd(vt,self.Tree[ivt])
            # expand to another tree
            jnear = self.nearbyT(v,not self.isGtree[vt])
            for jvt in jnear:
                if self.edgeInsertConditionAnotherTree(vt,jvt):
                    # TODO if there's no solution, should we give some reward?
                    self.qeAdd(vt,self.Tree[jvt])

    """
    return nearby(self.r) x in thelist
    """
    def nearby(self,vind,thelist):
        near = []
        for ind in thelist: # 太暴力…… 下次试试r近邻……
            if self.distance(ind,vind) < self.r:
                near.append(ind)
        return near
    def nearbyT(self,vind,isG):
        near = []
        for ti in range(len(self.Tree)):
            if self.isGtree[ti] == isG:
                if self.distance(vind, self.Tree[ti]) < self.r:
                    near.append(ti)
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
        self.updateCost()
        self.prune()
        self.sample(self.bestCost)
        self.r = radius(len(self.x))
        while len(self.vold) > 0:
            v = self.vold.pop()
            self.qv.put((self.vertexQueueValue(v),v))
    
    # update the cost of vertex (might be out-of-date because of rewire)
    def updateCost(self,prune = False):
        waitingToUpdate = queue.Queue()
        for cd in self.children[0]:
            waitingToUpdate.put(cd)
        for cd in self.children[1]:
            waitingToUpdate.put(cd)
        while not waitingToUpdate.empty():
            curV = waitingToUpdate.get()
            self.cost[curV] = self.cost[self.parent[curV]] + self.distance(self.Tree[curV],self.Tree[self.parent[curV]])
            for cd in self.children[curV]:
                waitingToUpdate.put(cd)
        if self.bestCost < float('inf'):
            self.bestCost = self.cost[self.bestConn[0]] + self.cost[self.bestConn[1]]\
                + self.distance(self.Tree[self.bestConn[0]],self.Tree[self.bestConn[1]])


    def prune(self):
        # if prune ...
        if self.bestCost < float('inf'):
            # self.updateCost(prune=True)
            for x in self.X_sample:
                if self.lowerBoundHeuristic(x) > self.bestCost:
                    self.X_sample.remove(x)


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
                reversePath.append(self.Tree[curV])
                curV = self.parent[curV]
            reversePath.append(self.Tree[curV])
        
        # reverse
        path = [0]
        while len(reversePath)>0:
            path.append(reversePath.pop())

        curV = vh
        if vh != 1:
            while self.parent[curV] != 1:
                path.append(self.Tree[curV])
                curV = self.parent[curV]
            path.append(self.Tree[curV])
        path.append(1)

        return path

    def drawGraph(self):
        plt.clf()
        if self.map.dimension == 2:
            self.map.drawMap()
            for xind in self.X_sample:
                plt.plot(self.x[xind][0],self.x[xind][1],'ob')            

            for vt in range(len(self.Tree)):
                v = self.Tree[vt]
                cl = 'r'
                if self.isGtree[vt]:
                    cl = 'g'
                plt.plot(self.x[v][0],self.x[v][1],'o'+cl)   
                if self.parent[vt]!=None:          
                    plt.plot([self.x[v][0], self.x[self.Tree[self.parent[vt]]][0]], 
                        [self.x[v][1], self.x[self.Tree[self.parent[vt]]][1]], '-'+cl)
            
            for vconnt in self.conn.keys():
                vconn = self.Tree[vconnt]
                vcon2 = self.Tree[self.conn[vconnt]]
                plt.plot([self.x[vconn][0], self.x[vcon2][0]], 
                    [self.x[vconn][1], self.x[vcon2][1]], '-y')
                
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


        