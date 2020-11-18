# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:25:48 2020

@author: CMH
"""
import numpy as np
import heapq


class MyPlanner:
    __slots__ = ['boundary', 'blocks', 'counter','open_list','close_list','myGraph','expand_num','epsilon','visited','g']

    def __init__(self, boundary, blocks):
        self.boundary = boundary
        self.blocks = blocks
        self.counter = 0 
        self.open_list =[]
        self.close_list = []
        self.myGraph = {}
        self.expand_num = 20
        self.epsilon = 3
        self.visited = []
        self.g = 0 
        
    def computeHeuristic(self,start,goal):
        z = np.abs(start.point[2] - start.point[2])
        dis = sum((start-goal)**2)-30*z
        return dis
    
    def computeCost(self,dR):
        dis = 0.05
        #dis = sum(dR**2)
        return dis
    
    def check_inside(self, point, idx, b_idx):
        return point[idx] >= self.blocks[b_idx,idx] and point[idx] <= self.blocks[b_idx,3+idx] 
    
    def segmentCollision(self, start, end):
        valid = False
        for b in range(self.blocks.shape[0]):
            if (end[0] - self.blocks[b,0])*(start[0] - self.blocks[b,0]) <= 0 and \
                self.check_inside(end, 1, b) and self.check_inside(start, 1, b) and \
                self.check_inside(end, 2, b) and self.check_inside(start, 2, b):
                valid = True
            if (end[1] - self.blocks[b,1])*(start[1] - self.blocks[b,1]) <= 0 and \
                self.check_inside(end, 0, b) and self.check_inside(start, 0, b) and \
                self.check_inside(end, 2, b) and self.check_inside(start, 2, b):
                valid = True
            if (end[2] - self.blocks[b,2])*(start[2] - self.blocks[b,2]) <= 0 and \
                self.check_inside(end, 0, b) and self.check_inside(start, 0, b) and \
                self.check_inside(end, 1, b) and self.check_inside(start, 1, b):
                valid = True
        return valid
    
    def isInBoundary(self,newrp):
        if( newrp[0] < self.boundary[0,0] or newrp[0] > self.boundary[0,3] or \
           newrp[1] < self.boundary[0,1] or newrp[1] > self.boundary[0,4] or \
           newrp[2] < self.boundary[0,2] or newrp[2] > self.boundary[0,5] ):
            return False
        return True
    
    def isCollisionFree(self,newrp):
        valid = True
        for i in range(self.blocks.shape[0]):
            if( newrp[0] > self.blocks[i,0] and newrp[0] < self.blocks[i,3] and\
               newrp[1] > self.blocks[i,1] and newrp[1] < self.blocks[i,4] and\
               newrp[2] > self.blocks[i,2] and newrp[2] < self.blocks[i,5] ):
                valid = False
                break
        return valid
    
    def isVisited(self, point):
        for i in self.visited:
            if sum((i-point)**2) <= 0.1:
                return True
        return False
    
    def Astar(self,point,goal):
        newrobotpos = np.copy(point)
        numofdirs = 26
        [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
        
        #27 different next step directions in the size of (3*27) 
        dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
        
        #Delete the (0,0,0) directions because it means the robot doesn't move
        dR = np.delete(dR,13,axis=1)
        
        #make every movement as the distance of 0.5 
        dR = dR / np.sqrt(np.sum(dR**2,axis=0)) / 2.0

        #myGraph is a dictionary with key: tuple(point) and value: (g,h)
        #open_list: (g+h, counter, point)
        #close_list: tuple(newrobotpos)

        if tuple(newrobotpos) not in self.myGraph:
            self.myGraph[tuple(newrobotpos)] = (self.g,self.computeHeuristic(newrobotpos,goal))
            
        for k in range(numofdirs):
            self.counter += 1

            #Go through the direction
            newrp = newrobotpos + dR[:,k]
            
            #Check if the newrp is valid
            if( newrp[0] < self.boundary[0,0] or newrp[0] > self.boundary[0,3] or \
               newrp[1] < self.boundary[0,1] or newrp[1] > self.boundary[0,4] or \
               newrp[2] < self.boundary[0,2] or newrp[2] > self.boundary[0,5] ):
                continue
            valid = True
            for i in range(self.blocks.shape[0]):
                if( newrp[0] > self.blocks[i,0] and newrp[0] < self.blocks[i,3] and\
                   newrp[1] > self.blocks[i,1] and newrp[1] < self.blocks[i,4] and\
                   newrp[2] > self.blocks[i,2] and newrp[2] < self.blocks[i,5] ):
                    valid = False
                    break
            if not valid:
                continue
            
            #Check if new point is in the Close List 
            if tuple(newrp) not in self.close_list:
                
                #Compute the cost of the moving direction
                cij = self.computeCost(dR[:,k])
            
                #Compute the heuristic of the new position
                h = self.computeHeuristic(newrp,goal)
            
                #Compute gj
                if tuple(newrp) not in self.myGraph: 
                    gj = self.g + cij
                    self.myGraph[tuple(newrp)] = (gj, h)
                    self.open_list.append((gj + h,gj,self.counter, newrp))
                
                elif tuple(newrp) in self.myGraph: 
                    gj = self.myGraph[tuple(newrp)][0]
                    if gj > (self.g + cij):
                        gj = self.g + cij
                        self.myGraph[tuple(newrp)] = (gj,h) 
                        self.open_list.append((gj + h,gj, self.counter, newrp))
        switch = 0 
        tmp = []
        while switch == 0:
            node = heapq.heappop(self.open_list)
            if sum((node[3] - newrobotpos)**2) > 1:
                tmp.append(node)
                continue
            else:
                switch = 1 

        for i in tmp:
            heapq.heappush(self.open_list,i)
        
        newrobotpos = node[3]
    
        self.close_list.append(tuple(newrobotpos))
        self.g = node[1]
        
        
        return newrobotpos

