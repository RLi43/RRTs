#!/usr/bin/env python3
#coding:utf-8

import copy
import math
import platform
import random
import time
import numpy as np
import queue

import matplotlib.pyplot as plt

show_animation = False

class BITstar(object):
    def __init__(self,_map,batchSize=30,maxIter=10):
        self.map = _map
        self.dimension = _map.dimension

        self.x = [_map.xinit,_map.xgoal]
        self.V = [0]
        self.E = [[],[]]
        self.gT = [0,float('inf')]
        self.X_samples = [1]
        self.qv = queue.PriorityQueue() # index of X
        self.qe = queue.PriorityQueue() # indexs of X [vind,xind/vind2]
        self.V_old = []
        self.r = float('inf')

        self.maxIter = maxIter
        self.batchSize = batchSize
        self.DISCRETE = self.map.DISCRETE
        self.pathCost = float('inf')
        
        self.eita = 1.2*2*(1+1/self.dimension)**(1/self.dimension) #?

        # Computing the sampling space
        xinit = self.map.xinit
        xgoal = self.map.xgoal
        self.cMin = self.distance(0,1)
        self.xCenter = (np.array(xinit)+np.array(xgoal))/2
        a1 = np.transpose([(np.array(xgoal)-np.array(xinit))/self.cMin])        
        # first column of idenity matrix transposed
        id1_t = np.array([1.0]+[0.0,]*(self.dimension-1)).reshape(1,self.dimension)
        M = a1 @ id1_t
        U, S, Vh = np.linalg.svd(M, 1, 1)
        self.C = np.dot(np.dot(U, 
            np.diag([1.0,]*(self.dimension-1)+[np.linalg.det(U) * np.linalg.det(np.transpose(Vh))]))
            , Vh)
    
    def search(self):
        ret = False
        i = 0
        while True:

            # batch creation
            if self.qe.empty() and self.qv.empty():
                i += 1
                if i == self.maxIter: break
                # self.Prune(self.gT[1])
                self.Sample()
                self.V_old = self.V.copy()
                for vind in self.V:
                    self.qvAdd(vind)
                self.r = self.radius(len(self.x))
                print("Batch ",i," r:",self.r,"V:",len(self.V),"E:",len(self.E))
                self.drawGraph()
                plt.pause(0.01)
            
            # Edge Selection
            bothEmpty = False
            while True:
                if self.qv.empty():
                    if self.qe.empty():
                        print('error! 73 - qe and qv are both empty')
                        bothEmpty = True
                    else:
                        (evalue,[vind,xind]) = self.qe.get()
                    break
                (vvalue,vin) = self.qv.get()
                if self.qe.empty():
                    self.ExpandVertex(vin)
                else:
                    (evalue, [vind,xind]) = self.qe.get()
                    if evalue < vvalue:
                        self.qv.put((vvalue,vin))
                        break
                    else:
                        self.qe.put((evalue,[vind,xind]))
                        self.ExpandVertex(vin)
            if bothEmpty:
                continue
            # Edge process
            # have a feasible path
            if self.gT[1] < float('inf'):
                ret = True
                if self.pathCost!= self.gT[1]:
                    print('pathCost:',self.gT[1])
                    self.pathCost = self.gT[1]
                # # debug
                # print(self.E)
                # return True
                hhxm = self.distance(1,xind)
                if self.gT[vind] + self.distance(vind,xind) + hhxm >= self.gT[1]:
                    # 不知道python需不需要清指针……
                    self.qe = queue.PriorityQueue()
                    self.qv = queue.PriorityQueue()
                    continue
                cost = self.cost(vind,xind)
                if self.distance(0,vind) + cost + hhxm >= self.gT[1]:
                    continue
            else:
                cost = self.cost(vind,xind)
                if cost == float('inf'):
                    continue

            try:
                self.V.index(xind)
                if self.gT[vind] + cost >= self.gT[xind]:
                    continue
                self.gT[xind] = self.gT[vind] + cost
                # prune (v,xm)
                for vin in self.E[xind]:
                    self.E[vin].remove(xind)
                self.E[xind] = []
            except:
                self.X_samples.remove(xind)
                self.V.append(xind)
                self.gT[xind] = self.gT[vind] + cost
                self.qvAdd(xind)
                
            self.E[xind] = [vind]
            self.E[vind].append(xind)
            # prune qe
            newqe = queue.PriorityQueue()
            while not self.qe.empty():
                (evalue,[vi,xi]) = self.qe.get()
                if xi != xind or self.gT[vi] + self.distance(vi,xind) >= self.gT[xind]:
                    newqe.put((evalue,[vi,xi]))
            self.qe = newqe
            if show_animation:
                self.drawGraph()
                plt.pause(0.01)
        
        return ret
    
    def ExpandVertex(self,vind):
        for xind in self.X_samples:
            if self.distance(xind,vind) > self.r:
                continue
            for vi in self.V:
                if vi == xind:
                    print("error!")
                if self.gT[1] < float('inf'):
                    if self.distance(0,vi)+self.distance(vi,xind)+self.distance(xind,1) >= self.gT[1]:
                        continue
                self.qeAdd(vi,xind)
        
        try:
            self.V_old.index(vind)
        except:
            V_near = []
            for wi in self.V:
                if self.distance(wi,vind) > self.r:
                    continue
                for vi in self.V:
                    if vi == wi: continue
                    try:
                        self.E[vi].index(wi)
                        self.E[wi].index(vi)
                    except:
                        if self.distance(0,vi)+self.distance(vi,wi)+self.distance(wi,1) < self.gT[1] \
                            and self.gT[vi] + self.distance(vi,wi) < self.gT[1]:
                            self.qeAdd(vi,wi)
    
    def Prune(self,c):
        assert c>0
        if c== float('inf'): return
        # 暴力删除法 会有冗余
        for xi in range(len(self.x)):
            if self.distance(0,xi)+self.distance(xi,1)>c:
                try:
                    self.V.index(xi)
                    self.V.remove(xi)
                except:
                    # should be in samples
                    # if error occur, debug...
                    self.X_samples.index(xi)
                    self.X_samples.remove(xi)
                for v2 in self.E[xi]:
                    self.E[v2].remove(xi)
                self.E[xi] = []
                self.x[xi] = []
        # 在删除时操作可能会快一点 # 但可能不想那么快剪枝呢
        for vind in self.V:
            if len(self.E[vind]) == 0:
                self.V.remove(vind)
                self.X_samples.append(vind)

    # 2*eita*((1+1/n)*(lambda(X_fh)/zeta_n)*log(q)/q)**(1/n)
    # zeta_n ~ the volume of the unit ball in the d-dimensional Euclidean space
    #          = pi^(n/2)/Gamma(n/2+1)*r^n
    #          好吧，就列表了……
    # lambda(X_fh) ~ Lebesgue measure (e.g. volume) of the X_fh = {x in X|f(x)<cbest}
    # 参考的代码里直接=2.0 ………… 行吧……
    def radius(self,q):
        # zeta = math.pi**(self.dimension/2)/GAMMA_N[self.dimension-1]
        # lamb = math.pi**(self.dimension/2)/GAMMA_N[self.dimension-1]*cMax/2*(math.sqrt(cMax**2-self.cMin**2)/2)**(self.dimension-1)
        # return self.eita*(math.log(q)/q*lamb/zeta)**(1/self.dimension)
        return 20.0 * math.sqrt((math.log(q) / q))
        # return self.eita*(math.log(q)/q*(cMax/(math.sqrt(cMax**2-self.cMin**2)))**(self.dimension-1))**(1/self.dimension)

    
    def CollisionPoint(self,x):        
        obs = self.map.obstacles
        for ob in obs:
            if _dis(x,ob[:-1])<=ob[-1]:
                return True
        return False
        
    def CollisionLine(self,x1,x2):
        dis = _dis(x1,x2)
        if dis<self.DISCRETE:
            return False
        nums = int(dis/self.DISCRETE)
        direction = (np.array(x2)-np.array(x1))/_dis(x1,x2)
        for i in range(nums+1):
            x = np.add(x1 , i*self.DISCRETE*direction)
            if self.CollisionPoint(x): return True
        if self.CollisionPoint(x2): return True
        return False
    
    def distance(self,ind1,ind2):
        return _dis(self.x[ind1],self.x[ind2])


    def Sample(self):
        ind = len(self.x)
        for i in range(self.batchSize):
            self.x.append(self._Sample())
            self.X_samples.append(ind+i)
            self.E.append([])
            self.gT.append(float('inf'))

    def _Sample(self):
        cMax = self.gT[1]
        if cMax<float('inf'):
            # ecllipse
            L = np.diag([cMax/2]+[math.sqrt(cMax**2-self.cMin**2)/2,]*(self.dimension-1))
            cl = np.dot(self.C,L)
            x = np.dot(cl,self.ballSample())+self.xCenter
            while self.CollisionPoint(x):
                x = np.dot(cl,self.ballSample())+self.xCenter
            return list(x)
        else:
            return self.SampleRandomFree()

    def ballSample(self):
        ret = []
        for i in range(self.dimension):
            ret.append(random.random()*2-1)
        ret = np.array(ret)
        return ret/np.linalg.norm(ret)*random.random()

    def SampleRandomFree(self):
        ret = self._SampleRandom()
        while self.CollisionPoint(ret):
            ret = self._SampleRandom()
        return ret        

    # random sample in whole space   
    def _SampleRandom(self):
        ret = []
        for i in range(self.dimension):
            ret.append(random.random()*(self.map.xlimmax[i]-self.map.xlimmin[i])+self.map.xlimmin[i])
        return ret

    def cost(self,ind1,ind2):
        if self.CollisionPoint(self.x[ind1]) or self.CollisionPoint(self.x[ind2]) \
            or self.CollisionLine(self.x[ind1],self.x[ind2]):
            return float('inf')
        return self.distance(ind1,ind2)


    # gT(v)+hh(v)
    def qvAdd(self,vind):
        dis = self.gT[vind] + self.distance(vind,1)
        self.qv.put((dis,vind))
    
    def qeAdd(self,vind,xind):
        dis = self.gT[vind] + self.distance(vind,xind) + self.distance(xind,1)
        try:
            self.qe.queue.index((dis,[vind,xind]))
        except:
            self.qe.put((dis,[vind,xind]))

    # # gT(v)+hh(v)
    # def BestQueueV(self):
    #     mDis = float('inf')
    #     bvind = -1
    #     for vind in self.qv:
    #         dis = self.gT[vind] + self.distance(vind,1) 
    #         if dis < mDis:
    #             bvind = vind
    #             mDis = dis
    #     return mDis, bvind
             
    # # gT(v)+ch(v,x)+hh(x)
    # def BestQueueE(self):
    #     mDis = float('inf')
    #     beind = -1
    #     for eind in range(len(self.qe)):
    #         [vind,xind] = self.qe[eind]
    #         dis = self.gT[vind] + self.distance(vind,xind) + self.distance(xind,1)
    #         if dis < mDis:
    #             mDis = dis
    #             beind = eind
    #     return mDis, beind

    def drawGraph(self):
        # print(self.V)
        # print(self.E)
        if self.dimension == 2:
            self.map.drawMap()
            for xind in self.X_samples:
                plt.plot(self.x[xind][0],self.x[xind][1],'ob')            
            
            # qes = queue.PriorityQueue()
            # while not self.qe.empty():
            #     (dis,[vind,xind]) = self.qe.get()
            #     qes.put((dis,[vind,xind]))
            #     plt.plot([self.x[vind][0], self.x[xind][0]], 
            #         [self.x[vind][1], self.x[xind][1]], '-r')
            # self.qe = qes

            for vind in self.V:
                for v2 in self.E[vind]:
                    plt.plot([self.x[vind][0], self.x[v2][0]], 
                        [self.x[vind][1], self.x[v2][1]], '-g')
    
    def drawPath(self):
        p1 = 1
        p2 = self.E[p1][0]
        plt.plot([self.x[p1][0], self.x[p2][0]], 
                        [self.x[p1][1], self.x[p2][1]], '-r')
        step = 1
        while p2!=0:
            step += 1
            mGt = float('inf')
            for p in self.E[p2]:
                if self.gT[p] <mGt:
                    mGt = self.gT[p]
                    p3 = p
            plt.plot([self.x[p2][0], self.x[p3][0]], 
                            [self.x[p2][1], self.x[p3][1]], '-r')
            p2 = p3
        print("step:",step)


class Map:
    def __init__(self,dim=2,obs_num=15,obs_size_max=2.5,xinit=[0,0],xgoal=[23,23],xlim1=[-3,-3],xlim2=[26,26]):
        self.dimension = dim
        self.xlimmin = xlim1
        self.xlimmax = xlim2
        self.xinit = xinit
        self.xgoal = xgoal
        self.obstacles = []
        self.DISCRETE = 0.05

        for i in range(obs_num):
            #TODO
            ob = []
            for j in range(dim):
                ob.append(random.random()*20+1.5)
            ob.append(random.random()*obs_size_max+0.2)
            self.obstacles.append(ob)
    
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
            plt.axis([self.xlimmin[0],self.xlimmax[0],self.xlimmin[1],self.xlimmax[1]])
            plt.grid(True)


def _dis(x1,x2):
    return np.linalg.norm(np.array(x1)-np.array(x2))

def main():
    print("Start rrt planning")

    # create map
    map2Drand = Map()

    bit = BITstar(map2Drand)
    # show map
    if show_animation:
        bit.map.drawMap()
    start_time = time.time()
    bit.search()
    print("time_use: ",time.time()-start_time)
    #debug
    bit.drawGraph()
    bit.drawPath()
    plt.show()
    print("Finished")
    
    
    rrt = RRT(_map=map2Drand,method="RRT*")
    rrt.Search()

    rrt.drawGraph()
    rrt.drawTree()
    rrt.drawPath()
    plt.show()
    


    # Plot path
    if  rrt.dimension==2 and show_animation:
        rrt.drawGraph()
        plt.show()



class RRT:
    def __init__(self,_map=None,method="RRT-Connect",maxIter=500):
        if _map == None:
            self.map = Map()
        else: self.map = _map
        self.method = method
        self.trees = []
        self.ninit = Node(self.map.xinit,cost=0,lcost=0)
        self.ngoal = Node(self.map.xgoal)
        self.dimension = self.map.dimension
        self.prob = 0.1
        self.maxIter = maxIter
        self.stepSize = 0.5
        self.DISCRETE = 0.05
        self.path = []
        self.pathCost = float('inf')

        # *
        self.nearDis = 2
    
    def Search(self):
        ret = False
        print("method: ",self.method)
        # Search
        start_time = time.time()
        if self.method == "RRT":
            ret = self.rrtSearch()
        elif self.method == "RRT-Connect":
            self.prob = 0
            ret = self.rrtConnectSearch()
        elif self.method == "RRT*":
            ret = self.rrtStarSearch()
        elif self.method == "Informed RRT*":
            ret = self.InformedRRTStarSearch()
        else:
            print("Unsupported Method, please choose one of:")
            print("RRT, RRT-Connect")
        end_time = time.time()
        if not ret:
            print("Solve Failed")
            return False
        
        print("Get!")
        # getPath
        self.getPath()
        print("path cost(distance): ", self.pathCost," steps: ",len(self.path)," time_use: ",end_time-start_time)

        if show_animation:
            self.drawGraph()
            self.drawPath()
        return ret
    
    def getPath(self):
        # knowing that no more than 2 trees
        t = self.trees[0]
        n = t.nodes[-1]
        sum = 0
        while n.parent:
            self.path.append(n.x)
            n = n.parent
            sum += n.lcost
        self.path.append(n.x)
        if len(self.trees)>1:
            nl = n
            n = self.trees[2].root
            sum += Node.distancenn(nl,n)
            while n.parent:
                self.path.insert(0,n.parent.x)
                n = n.parent
                sum += n.lcost
                
        self.pathCost = sum

    def rrtSearch(self):
        tree = Tree(self.ninit)
        self.trees.append(tree)
        for i in range(self.maxIter):
            xrand = self.SampleRandomFreeFast()
            nnearest,dis = tree.getNearest(xrand)
            nnew = self.Extend(nnearest,xrand)
            if nnew!=None:
                tree.addNode(nnew)
                if(Node.distancenn(nnew,self.ngoal)<self.stepSize):
                    tree.addNode(self.ngoal,parent=nnew)
                    print("iter ",i," find!")
                    return True
            if show_animation:
                self.drawGraph(rnd=xrand,new=nnew)
                plt.pause(0.0001)
        return False

    def rrtConnectSearch(self):
        treeI = Tree(nroot=self.ninit)
        treeG = Tree(nroot=self.ngoal)
        self.trees.append(treeI)
        self.trees.append(treeG)
        tree1 = treeI # the less points one
        tree2 = treeG
        for i in range(self.maxIter):
            xrand = self.SampleRandomFreeFast()
            nnearest,dis = tree1.getNearest(xrand)
            nnew = self.Extend(nnearest,xrand)
            if nnew!=None:
                tree1.addNode(nnew)
                nnears = tree2.getNearby(nnew,self.stepSize)
                if len(nnears):
                    ncon = nnears[0] # or chose the nearest?
                    connectTree = Tree(ncon)
                    self.trees.append(connectTree)
                    print("iter ",i," find!")
                    return True
            
                # another tree
                # directly forward to nnew
                nnearest,dis = tree2.getNearest(nnew.x)
                nnew2 = self.Extend(nnearest,nnew.x)
                while nnew2:
                    tree2.addNode(nnew2)
                    nnears = tree1.getNearby(nnew2,self.stepSize)
                    if len(nnears):
                        ncon = nnears[0]
                        connectTree = Tree(ncon)
                        self.trees = [tree2,tree1,connectTree]
                        print("iter ",i," find!")
                        return True
                    nnearest = nnew2
                    nnew2 = self.Extend(nnearest,nnew.x)
                    
                    if show_animation:
                        self.drawGraph(rnd=xrand,new=nnew,drawnodes=False)
                        self.drawTree(treeI,'g')
                        self.drawTree(treeG,'b')
                        plt.pause(0.0001)

                # check the size,
                # let the less one to random spare
                if tree1.length()>tree2.length():
                    temptree = tree1
                    tree1 = tree2
                    tree2 = temptree

            if show_animation:
                self.drawGraph(rnd=xrand,new=nnew,drawnodes=False)
                self.drawTree(treeI,'g')
                self.drawTree(treeG,'b')
                plt.pause(0.0001)
        return False

    def rrtStarSearch(self):
        tree = Tree(self.ninit)
        self.trees.append(tree)
        for i in range(self.maxIter):
            xrand = self.SampleRandomFreeFast()
            nnearest,dis = tree.getNearest(xrand)
            nnew = self.Extend(nnearest,xrand)
            if nnew!=None:
                tree.addNode(nnew)

                # adjust
                self.reParent(nnew,tree)
                self.reWire(nnew,tree)

                if(Node.distancenn(nnew,self.ngoal)<self.stepSize):
                    tree.addNode(self.ngoal,parent=nnew)
                    print("iter ",i," find!")
                    return True
            if show_animation:
                self.drawGraph(rnd=xrand,new=nnew)
                plt.pause(0.0001)
        return False

    def _CollisionPoint(self,x): 
        obs = self.map.obstacles
        for ob in obs:
            if Node.distancexx(x,ob[:-1])<=ob[-1]:
                return True
        return False
    def _CollisionLine(self,x1,x2):
        dis = Node.distancexx(x1,x2)
        if dis<self.DISCRETE:
            return False
        nums = int(dis/self.DISCRETE)
        direction = (np.array(x2)-np.array(x1))/Node.distancexx(x1,x2)
        for i in range(nums+1):
            x = np.add(x1 , i*self.DISCRETE*direction)
            if self._CollisionPoint(x): return True
        if self._CollisionPoint(x2): return True
        return False

    def Nearest(self,xto,nodes=None):
        if nodes == None:
            nodes = self.trees[0].nodes
        dis = float('inf')
        nnearest = None
        for node in nodes:
            curDis = Node.distancenx(node,xto)
            if curDis < dis:
                dis = curDis
                nnearest = node
        return nnearest

    def Extend(self,nnearest,xrand,step=None):
        if not step:
            step = self.stepSize
        dis = Node.distancenx(nnearest,xrand)
        if dis<step:
            xnew = xrand
        else:
            dis = step
            xnew = np.array(nnearest.x) + step*(np.array(xrand)-np.array(nnearest.x))/Node.distancenx(nnearest,xrand)
        if self._CollisionPoint(xnew):
            return None
        if self._CollisionLine(xnew,nnearest.x):
            return None
        nnew = Node(xnew,parent=nnearest,lcost=dis)
        return nnew
    
    def reParent(self,node,tree):
        # TODO: check node in tree
        nears = tree.getNearby(node)
        for n in nears:
            if self._CollisionLine(n.x,node.x):
                continue
            newl = Node.distancenn(n,node)
            if n.cost + newl < node.cost:
                node.parent = n
                node.lcost = newl
                node.cost = n.cost + newl

    # what if combine the both?
    def reWire(self,node,tree):
        nears = tree.getNearby(node)
        for n in nears:
            if self._CollisionLine(n.x,node.x):
                continue
            newl = Node.distancenn(n,node)
            if node.cost + newl < n.cost:
                n.parent = node
                n.lcost = newl
                n.cost = node.cost + newl


    def SampleRandomFreeFast(self):
        r = random.random()
        if r<self.prob:
            return self.map.xgoal
        else:
            ret = self._SampleRandom()
            while self._CollisionPoint(ret):
                ret = self._SampleRandom()
        return ret

    def _SampleRandom(self):
        ret = []
        for i in range(self.dimension):
            ret.append(random.random()*(self.map.xlimmax[i]-self.map.xlimmin[i])+self.map.xlimmin[i])
        return ret

    def drawGraph(self, xCenter=None, cBest=None, cMin=None, etheta=None, rnd=None, new=None,drawnodes=True):
        if self.dimension==2:
            plt.clf()
            sysstr = platform.system()
            if(sysstr =="Windows"):
                scale = 18
            elif(sysstr == "Linux"):
                scale = 24
            else: scale = 24
            for (ox, oy, size) in self.map.obstacles:
                plt.plot(ox, oy, "ok", ms=scale * size)
            if rnd is not None:
                plt.plot(rnd[0], rnd[1], "^k")
            if new is not None:
                plt.plot(new.x[0], new.x[1], "og")
            
            if drawnodes:
                self.drawTree()

            plt.plot(self.map.xinit[0], self.map.xinit[1], "xr")
            plt.plot(self.map.xgoal[0], self.map.xgoal[1], "xr")
            plt.axis([self.map.xlimmin[0],self.map.xlimmax[0],
            self.map.xlimmin[1],self.map.xlimmax[1]])
            plt.grid(True)
        elif self.dimension == 3:            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
    
    def drawTree(self,tree=None,color='g'):
        if tree==None:
            trees = self.trees
        else:
            trees = [tree]
        if self.dimension == 2:
            for t in trees:
                for node in t.nodes:
                    if node.parent is not None:
                        plt.plot([node.x[0], node.parent.x[0]], 
                        [node.x[1], node.parent.x[1]], '-'+color)
        elif self.dimension == 3:
            pass

    def drawPath(self):
        if self.dimension == 2:
            plt.plot([x for (x, y) in self.path], [y for (x, y) in self.path], '-r')  
        elif self.dimension == 3:
            pass




class Node:
    def __init__(self,x,lcost=0.0,cost=float('inf'),parent=None):
        self.x = np.array(x)
        self.lcost = lcost # from parent
        self.cost = cost # from init
        self.parent = parent
        if parent:
            self.cost = self.lcost+parent.cost
        # self.children = children

    @staticmethod
    def distancenn(n1,n2):
        return np.linalg.norm(np.array(n1.x)-np.array(n2.x))
    @staticmethod
    def distancenx(n,x):
        return np.linalg.norm(n.x-np.array(x))
    @staticmethod
    def distancexx(x1,x2):
        return np.linalg.norm(np.array(x1)-np.array(x2))

#TODO
class Tree:
    def __init__(self,nroot):
        self.root = nroot
        self.nodes = [nroot]

    def addNodeFromX(self,x,parent):
        self.nodes.append(Node(np.array(x),parent=parent))
    
    def addNode(self,n,parent=None):
        if parent:
            n.parent = parent
            n.cost = n.lcost + parent.cost
        self.nodes.append(n)
    
    def length(self):
        return len(self.nodes)
    
    def getNearest(self,x):
        dis = float('inf')
        nnearest = None
        for node in self.nodes:
            curDis = Node.distancenx(node,x)
            if curDis < dis:
                dis = curDis
                nnearest = node
        return nnearest,dis
    
    def getNearby(self,nto,dis=None):
        ret = []
        if dis==None:
            dis = 20.0 * math.sqrt((math.log(self.length()) / self.length()))
        for n in self.nodes:
            if Node.distancenn(nto,n)<dis:
                ret.append(n)
        return ret



if __name__ == '__main__':
    main()