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

show_animation = True

class RRT:
    def __init__(self,_map=None,method="RRT-Connect",maxIter=500):
        if _map == None:
            self.map = Map()
        else: self.map = _map
        self.method = method
        self.nodes = []
        self.edges = None
        self.dimension = self.map.dimension
        self.prob = 0.1
        self.maxIter = maxIter
        self.stepSize = 0.4
        self.DISCRETE = 0.05
        self.path = []
        self.pathCost = float('inf')
    
    def Search(self):
        ret = None
        print("method: ",self.method)
        start_time = time.time()
        if self.method == "RRT":
            ret = self.rrtSearch()
        elif self.method == "RRT-Connect":
            self.prob = 0
            ret = self.rrtConnectSearch()
        else:
            print("Unsupported Method, please choose one of:")
            print("RRT, RRT-Connect")
        end_time = time.time()
        if not ret:
            print("Solve Failed")
            return False
        
        #TODO 这里不严谨
        nlast = self.nodes[-1]
        sum = 0
        while nlast.parent is not None:
            self.path.append(list(nlast.x))
            sum += Node.distancenn(nlast,nlast.parent)
            nlast = nlast.parent
        self.pathCost = sum
        print("path cost(distance): ", sum," steps: ",len(self.path)," time_use: ",end_time-start_time)

        if show_animation:
            self.drawGraph()
            self.drawPath()
        return ret

    def rrtSearch(self):
        self.nodes.append(Node(np.array(self.map.xinit)))
        ngoal = Node(np.array(self.map.xgoal))
        for i in range(self.maxIter):
            xrand = self.SampleRandomFreeFast()
            nnearest = self.Nearest(xrand)
            nnew = self.Extend(nnearest,xrand)
            if nnew!=None:
                self.nodes.append(nnew)
                if(Node.distancenn(nnew,ngoal)<self.stepSize):
                    self.nodes.append(ngoal)
                    ngoal.parent=nnew
                    nnew.children.append(ngoal)
                    print("iter ",i," find!")
                    return True
            if show_animation:
                self.drawGraph(rnd=xrand,new=nnew)
                plt.pause(0.0001)
        return False

    def rrtConnectSearch(self):
        ngoal = Node(np.array(self.map.xgoal))
        ninit = Node(np.array(self.map.xinit))
        treeI = [ninit]
        treeG = [ngoal]
        tree1 = treeI # the less points one
        tree2 = treeG
        for i in range(self.maxIter):
            xrand = self.SampleRandomFreeFast()
            nnearest = self.Nearest(xrand,nodes=tree1)
            nnew = self.Extend(nnearest,xrand)
            if nnew!=None:
                tree1.append(nnew)
                ncon = self.theOneNearby(nnew,tree2)
                if ncon:
                    self.nodes = tree1+tree2
                    nLast = nnew2
                    while ncon:
                        nNext = ncon.parent
                        self.nodes.append(ncon)
                        ncon.parent = nLast
                        nLast = ncon
                        ncon = nNext
                    print("iter ",i," find!")
                    return True
            
                # another tree
                # directly forward to nnew
                nnearest = self.Nearest(nnew.x,nodes=tree2)
                nnew2 = self.Extend(nnearest,nnew.x)
                while nnew2:
                    tree2.append(nnew2)
                    ncon = self.theOneNearby(nnew2,tree1)
                    if ncon:
                        self.nodes = tree1+tree2
                        nLast = nnew2
                        while ncon:
                            nNext = ncon.parent
                            self.nodes.append(ncon)
                            ncon.parent = nLast
                            nLast = ncon
                            ncon = nNext
                        print("iter ",i," find!")
                        return True
                    nnearest = nnew2
                    nnew2 = self.Extend(nnearest,nnew.x)
                    
                    if show_animation:
                        self.drawGraph(rnd=xrand,new=nnew,drawnodes=False)
                        self.drawNodes(treeI,'g')
                        self.drawNodes(treeG,'b')
                        plt.pause(0.0001)

                # check the size,
                # let the less one to random spare
                if(len(tree1)>len(tree2)):
                    temptree = tree1
                    tree1 = tree2
                    tree2 = temptree

            if show_animation:
                self.drawGraph(rnd=xrand,new=nnew,drawnodes=False)
                self.drawNodes(treeI,'g')
                self.drawNodes(treeG,'b')
                plt.pause(0.0001)
            


        return False

    def theOneNearby(self,node,tree):
        for n in tree:
            if Node.distancenn(node,n)<self.stepSize:
                return n
        return None
            

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
            nodes = self.nodes
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
        nnew = Node(xnew,parent=nnearest,cost=dis)
        nnearest.children.append(nnew)
        return nnew

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


    def drawGraph(self, xCenter=None, cBest=None, cMin=None, etheta=None, rnd=None, new=None,drawnodes=True):
        if self.dimension!=2:
            return
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
            self.drawNodes()

        plt.plot(self.map.xinit[0], self.map.xinit[1], "xr")
        plt.plot(self.map.xgoal[0], self.map.xgoal[1], "xr")
        plt.axis([self.map.xinit[0]+self.map.randBias[0],self.map.xinit[0]+self.map.randBias[0]+self.map.randLength[0],
        self.map.xinit[1]+self.map.randBias[1],self.map.xinit[1]+self.map.randBias[1]+self.map.randLength[1]])
        plt.grid(True)

    def drawNodes(self,nodes=None,color='g'):
        if nodes==None:
            nodes=self.nodes
        for node in nodes:
            if node.parent is not None:
                plt.plot([node.x[0], node.parent.x[0]], 
                [node.x[1], node.parent.x[1]], '-'+color)



    def drawPath(self):
        plt.plot([x for (x, y) in self.path], [y for (x, y) in self.path], '-r')        
        return



class Node:
    def __init__(self,x,cost=0.0,parent=None,children=[]):
        self.x = x
        self.cost = cost
        self.parent = parent
        self.children = children

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
        self.vertice = [nroot]
        self.edges [[]]

    def addNode(self,x,parent):
        self.vertice.append(Node(np.array(x)),parent=parent)


class Map:
    def __init__(self,dim=2,obs_num=50,obs_size_max=2.5,xinit=[0,0],xgoal=[23,23],randLength=[29,29],randBias=[-3,-3]):
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
    amap = Map()
    
    rrt = RRT(_map=amap)
    rrt.drawGraph()
    plt.pause(0.01)
    input("any key to start")
    rrt.Search()
    print("Finished")
    
    # Plot path
    if  rrt.dimension==2 and show_animation:
        rrt.drawGraph()
        rrt.drawPath()
        plt.show()


if __name__ == '__main__':
    main()