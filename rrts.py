#!/usr/bin/env python3
#coding:utf-8
"""
RRTs path planning
with RRT*, RRT-connect, informed RRT*

Reference: 
Informed RRT*: Optimal Sampling-based Path Planning Focused via
Direct Sampling of an Admissible Ellipsoidal Heuristichttps://arxiv.org/pdf/1404.2334.pdf
Code from github TODO

"""

import random
import numpy as np
import numpy.linalg as lng
import math
import copy
import matplotlib.pyplot as plt
import time
import platform
from mpl_toolkits.mplot3d import Axes3D

show_animation = False

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
        self.stepSize = 0.4
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
        else:
            print("Unsupported Method, please choose one of:")
            print("RRT, RRT-Connect")
        end_time = time.time()
        if not ret:
            print("Solve Failed")
            return False
        
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

    def InformedRRTStarSearch(self, animation=True):

        self.nodeList = [self.start]
        # max length we expect to find in our 'informed' sample space, starts as infinite
        cBest = float('inf')
        pathLen = float('inf')
        solutionSet = set()
        path = None

        # Computing the sampling space
        cMin = math.sqrt(pow(self.start.x - self.goal.x, 2)
                         + pow(self.start.y - self.goal.y, 2))
        xCenter = np.array([[(self.start.x + self.goal.x) / 2.0],
                            [(self.start.y + self.goal.y) / 2.0], [0]])
        a1 = np.array([[(self.goal.x - self.start.x) / cMin],
                       [(self.goal.y - self.start.y) / cMin], [0]])

        etheta = math.atan2(a1[1], a1[0])
        # first column of idenity matrix transposed
        id1_t = np.array([1.0, 0.0, 0.0]).reshape(1, 3)
        M = a1 @ id1_t
        U, S, Vh = np.linalg.svd(M, 1, 1)
        C = np.dot(np.dot(U, np.diag(
            [1.0, 1.0, np.linalg.det(U) * np.linalg.det(np.transpose(Vh))])), Vh)

        for i in range(self.maxIter):
            # Sample space is defined by cBest
            # cMin is the minimum distance between the start point and the goal
            # xCenter is the midpoint between the start and the goal
            # cBest changes when a new path is found

            rnd = self.informed_sample(cBest, cMin, xCenter, C)
            nind = self.getNearestListIndex(self.nodeList, rnd)
            nearestNode = self.nodeList[nind]
            # steer
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
            newNode = self.getNewNode(theta, nind, nearestNode)
            d = self.lineCost(nearestNode, newNode)

            isCollision = self.__CollisionCheck(newNode, self.obstacleList)
            isCollisionEx = self.check_collision_extend(nearestNode, theta, d)

            if isCollision and isCollisionEx:
                nearInds = self.findNearNodes(newNode)
                newNode = self.chooseParent(newNode, nearInds)

                self.nodeList.append(newNode)
                self.rewire(newNode, nearInds)

                if self.isNearGoal(newNode):
                    solutionSet.add(newNode)
                    lastIndex = len(self.nodeList) - 1
                    tempPath = self.getFinalCourse(lastIndex)
                    tempPathLen = self.getPathLen(tempPath)
                    if tempPathLen < pathLen:
                        path = tempPath
                        cBest = tempPathLen

            if animation:
                self.drawGraph(xCenter=xCenter,
                               cBest=cBest, cMin=cMin,
                               etheta=etheta, rnd=rnd)

        return path

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
        direction = (np.array(x1)-np.array(x2))/Node.distancexx(x1,x2)
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
        """
        从nnearest向xrand扩展step长度
        找不到返回None，找到返回nnew，父子关系都处理了
        * 如果xrand很近，就到xrand为止
        """
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
        nears = tree.getNearby(node,self.nearDis)
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
        nears = tree.getNearby(node,self.nearDis)
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
            ret.append(random.random()*self.map.randLength[i]+self.map.randBias[i])
        return ret
    
    def informed_sample(self, cMax, cMin, xCenter, C):
        if cMax < float('inf'):
            r = [cMax / 2.0,
                 math.sqrt(cMax**2 - cMin**2) / 2.0,
                 math.sqrt(cMax**2 - cMin**2) / 2.0]
            L = np.diag(r)
            xBall = self.sampleUnitBall()
            rnd = np.dot(np.dot(C, L), xBall) + xCenter
            rnd = [rnd[(0, 0)], rnd[(1, 0)]]
        else:
            rnd = self.sampleFreeSpace()

        return rnd

    def sampleUnitBall(self):
        a = random.random()
        b = random.random()

        if b < a:
            a, b = b, a

        sample = (b * math.cos(2 * math.pi * a / b),
                  b * math.sin(2 * math.pi * a / b))
        return np.array([[sample[0]], [sample[1]], [0]])


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
            plt.axis([self.map.xinit[0]+self.map.randBias[0],self.map.xinit[0]+self.map.randBias[0]+self.map.randLength[0],
            self.map.xinit[1]+self.map.randBias[1],self.map.xinit[1]+self.map.randBias[1]+self.map.randLength[1]])
            plt.grid(True)
        elif self.dimension == 3:            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
    
    #TODO
    def plot_ellipse(self, xCenter, cBest, cMin, etheta):  # pragma: no cover

        a = math.sqrt(cBest**2 - cMin**2) / 2.0
        b = cBest / 2.0
        angle = math.pi / 2.0 - etheta
        cx = xCenter[0]
        cy = xCenter[1]

        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        R = np.array([[math.cos(angle), math.sin(angle)],
                      [-math.sin(angle), math.cos(angle)]])
        fx = R @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, "xc")
        plt.plot(px, py, "--c")

    def drawTree(self,tree=None,color='g'):
        """
        若不指明，将所有树都画出来
        """
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
        self.x = x
        self.lcost = lcost # from parent
        self.cost = cost # from init
        self.parent = parent
        if parent:
            self.cost = self.lcost+parent.cost
        # self.children = children

    @staticmethod
    def distancenn(n1,n2):
        return lng.norm(np.array(n1.x)-np.array(n2.x))
    @staticmethod
    def distancenx(n,x):
        return lng.norm(n.x-np.array(x))
    @staticmethod
    def distancexx(x1,x2):
        return lng.norm(np.array(x1)-np.array(x2))

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
    
    def getNearby(self,nto,dis):
        ret = []
        for n in self.nodes:
            if Node.distancenn(nto,n)<dis:
                ret.append(n)
        return ret
    


class Map:
    def __init__(self,dim=2,obs_num=10,obs_size_max=2.5,xinit=[0,0],xgoal=[23,23],randLength=[29,29],randBias=[-3,-3]):
        self.dimension = dim
        self.xinit = xinit
        self.xgoal = xgoal
        self.randLength = randLength
        self.randBias = randBias
        self.obstacles = []
        for i in range(obs_num):
            #TODO
            ob = []
            for j in range(dim):
                ob.append(random.random()*20+1.5)
            ob.append(random.random()*obs_size_max+0.2)
            self.obstacles.append(ob)

def main():
    print("Start rrt planning")

    # create map
    map2Drand = Map()
    map3Drand = Map(dim=3,obs_num=64,obs_size_max=5, xinit=[0,0,0],xgoal=[23,23,23],randBias=[-3,-3,-3],randLength=[29,29,29])
    
    rrt = RRT(_map=map2Drand,method="RRT*")
    if show_animation:
        rrt.drawGraph()
        plt.pause(0.01)
        # input("any key to start")
    rrt.Search()

    # #debug for 3D
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # for ob in rrt.map.obstacles:
    #     ax.scatter(ob[0], ob[1], ob[2], c='B',s=ob[3]*100)    
    # nx = []
    # ny = []
    # nz = []
    # for node in rrt.nodes:
    #     nx.append(node.x[0])
    #     ny.append(node.x[1])
    #     nz.append(node.x[2])
    # ax.scatter(nx, ny, nz, c='r',s=1)
    # nx = []
    # ny = []
    # nz = []
    # for node in rrt.path:
    #     nx.append(node[0])
    #     ny.append(node[1])
    #     nz.append(node[2])
    # ax.plot(nx,ny,nz, label='parametric curve')
    # plt.show()

    rrt2 = RRT(_map=map2Drand,method="RRT")
    rrt2.Search()

    rrt.drawGraph()
    rrt.drawTree()
    rrt.drawPath()
    plt.show()

    rrt2.drawGraph()
    rrt2.drawTree()
    rrt2.drawPath()
    plt.show()


    print("Finished")
    
    # Plot path
    if  rrt.dimension==2 and show_animation:
        rrt.drawGraph()
        rrt.drawTree()
        rrt.drawPath()
        plt.show()


if __name__ == '__main__':
    main()