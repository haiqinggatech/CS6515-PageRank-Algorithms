# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 14:40:02 2018

@author: Haiqing zhu
"""

import time
#argparse allows the parsing of command line arguments
import argparse
#utility functions for the PageRank project
import GA_ProjectUtils as util
#useful structure to build dictionaries of lists - provides 
#a default entry for keys that don't exist in map
from collections import defaultdict
#to build a dictionary that will have a list as its value
#use : mydict = defaultdict(list)
#may be used if desired
import numpy as np    

#PageRank object to hold graph representation and code to solve for algorithm
class PageRank(object):     
    #this object will build a page rank vector on a map of nodes
    def __init__(self, inFileName, alpha=.85, sinkHandling=False):
        """        
        Args:
            inFileName : the name of the file describing the graph to be 
                        ranked using the page rank algorithm
        """
        self.inFileName = inFileName
        #set alpha value
        self.alpha = alpha
        #whether using self loops or using alpha==0 for sinks default 1
        self.sinkHandling = sinkHandling 
        #make output file name
        self.outFileName = util.makeResOutFileName(self.inFileName,self.alpha, self.sinkHandling)  
        self.selfLoops = False 


        #this is only placed here to prevent errors when running empty template code
        self.rankVec = []        
        #dictionary of every node and its outlist and list of all node ids
        self.adjList, self.nodeIDs = util.loadGraphADJList(self.inFileName)
        #number of nodes total in map
        self.N = len(self.nodeIDs)

        #Task 1 : initialize data structures you will use
        self.initAllStructs()   
    
    """
    Task 1 : using self.adjList, do the following, in this order :
    1. conditionally add self loops to self.adjList : for type 1 sink handling only
    2. build in-list structure to relate a node to all the nodes pointing to it
    3. build out-degree structure to hold reference to the out degree of all nodes
    4. conditionally build list of all sink nodes, for type 3 sink handling only)
    5. initialize self.rankVec : pagerank vector -> initialize properly(uniformly)
    """
    def initAllStructs(self):

        #your code goes here < 1 >

        flag = self.sinkHandling

        length = self.N

        out_degree = list(0 for i in xrange(length)) 
        temp_list = list([] for i in xrange(length)) 

        if flag==0:
            for i in self.nodeIDs:
                self.adjList[i].append(self.nodeIDs[i])

        for item in self.adjList.items():
            key,value = item
            out_degree[key] = len(value)
            for k in value:
                temp_list[k].append(key)

        if flag == 1:
            temp = []
            count = 0
            for n in range(self.N):
                if out_degree[n] == 0:
                    temp.append(n)
                    count = 1 + count
            self.sink_nodes = temp[:]
            self.k = count

        self.rankVec = [float(1)/self.N]*self.N
        self.degree =out_degree 
        self.list= temp_list


        
    """
    Task 2 : using in-list structure, out-degree structure, (and sink-related 
    structure if appropriate for current sink handling strategy) :
    
    Perform single iteration of PageRank algorithm and return resultant vector
    """
    def solveRankIter(self, oldRankVec):
        #need to make copy of old rank vector 

        newRankVec = oldRankVec[:]

        alpha = self.alpha
        length = self.N

        if self.sinkHandling == 0:
            xxxx  = (1 - alpha ) / length 
        else:
            temp = 0
            for item in self.sink_nodes:
                temp = oldRankVec[item] + temp
            xxxx = (( alpha * temp) + (1 - alpha)) / length
 
        temp_list=[]
        for item in self.list:
            temp = 0
            for node in item: 
                temp = (oldRankVec[node] / self.degree[node]) + temp
            temp_list.append(xxxx +temp *alpha)
    
        return temp_list

    """
    Task 3 : Find page rank vector by iterating through solveRankIter calls  
    until rank vector updates are within eps.
    """
    def solveRankToEps(self, eps):
        #copy current page rank vector
        newRankVec = self.rankVec[:]
        while 1:
            updatedRankVec = self.solveRankIter(newRankVec)
            diff=[abs(x - y) for x, y in zip(updatedRankVec, newRankVec)]
            if all (x <= eps for x in diff) ==True: break
            newRankVec = updatedRankVec

        return newRankVec  
    
    
    """
        find page rank vector, save results.  Optionally sort results, although
        unnecessary for proper verification.
    """
    def solvePageRank(self, eps, saveVals=True):
        #DO NOT MODIFY THIS FUNCTION -add any extra functions you want to use
        #in solveRankToEps
        
        self.rankVec = self.solveRankToEps(eps)
                
        if(len(self.rankVec)==0):
            print('Zero-size PageRank vector Error.')
            
        if(saveVals):
            print('Saving results')
            util.saveRankData(self.outFileName, self.rankVec)
            print('Results saved')           
            
        return self.rankVec, self.nodeIDs    

    
#once the appropriate runs have been performed, plot results
#alpha specified in prObj is ignored - .75,.85 and .95 alphas are tested
def plotRes(prObj):
    #for graphs you may wish to code
    #import matplotlib as plt
    import matplotlib.pyplot as plt
    
    stSiteLoc = 0
    endSiteLoc = 20
    # use results for .75, .85 and .95 alphas to build plots - vNodeIDxx and 
    # vRankVecXX results come back sorted in descending rank order  
    vNodeID75, vRankVec75, dictRV75 = util.getResForPlots(prObj, .75)
    vNodeID85, vRankVec85, dictRV85 = util.getResForPlots(prObj, .85)    
    vNodeID95, vRankVec95, dictRV95 = util.getResForPlots(prObj, .95)
    
    #find union of all top x sites' site ids
    allNodesSet = set()
    allNodesSet.update(vNodeID75[stSiteLoc:endSiteLoc])
    allNodesSet.update(vNodeID85[stSiteLoc:endSiteLoc])
    allNodesSet.update(vNodeID95[stSiteLoc:endSiteLoc])
    #all nodes in top 20 for any of the 3 alpha settings - unsorted
    allTopNodes = list(allNodesSet)
    
    #list of values in allTopNodes order
    pltPRVals85tmp = [dictRV85[x] for x in allTopNodes] 
    #find appropriate order  - use idxs to find actual node IDS in allTopNodes      
    srtdIDXsInATNlst, pltPRVals85 = util.getSortResIDXs(pltPRVals85tmp)
    allTopNodes2 = [allTopNodes[x] for x in srtdIDXsInATNlst]    
    #comparing alpha of .75 and .95 to alpha of .85
   
    pltPRVals75 = [dictRV75[x] for x in allTopNodes2]
    pltPRVals95 = [dictRV95[x] for x in allTopNodes2]
    
    # data to plot
    n_groups = len(allTopNodes2)
     
    # create plot
    fig, ax = plt.subplots()
    #index = np.asarray(allTopNodes)#np.arange(n_groups)
    index = np.arange(n_groups)
    bar_width = 0.30
    opacity = 0.8
     
    r1 = plt.bar(index, pltPRVals75, bar_width,
                     alpha=opacity,
                     color='b',
                     label='alpha=.75')
     
    r2 = plt.bar(index + bar_width, pltPRVals85, bar_width,
                     alpha=opacity,
                     color='r',
                     label='alpha=.85')
    r3 = plt.bar(index + 2*bar_width, pltPRVals95, bar_width,
                     alpha=opacity,
                     color='g',
                     label='alpha=.95')
     
    plt.xlabel('Site Rank')
    plt.ylabel('Score')
    ttlStr = 'Scores for dataset {} for sites ranked {} through {}\n'.format(prObj.inFileName,(stSiteLoc+1),endSiteLoc) \
                  + 'for 3 alpha values using {} sink handling'.format('self-loop' if prObj.sinkHandling==0 else 'Type 3')
    plt.title(ttlStr)
    plt.legend()
     
    plt.tight_layout()
    plt.show()


#calculate page rank vector
def calcRes(prObj, args, prMadeTime):
    #calculate page rank vector for passed arguments   
    print('\nCalculating PageRank vector')
    rankVec, nodeIDsInRankOrder = prObj.solvePageRank(float(args.eps))
    rvDoneTime = time.time()
    print('\nPageRank vector calculated.  Elapsed time : {} seconds'.format(rvDoneTime - prMadeTime))
    print('\nVerifying PageRank vector')
    compareRes = util.verifyResults(prObj)  
    resStr = 'matches' if compareRes else 'does not match'
    print('\nYour calculated PageRank vector {} the validation file'.format(resStr))


"""     
main
"""     
def main():	
    #DO NOT REMOVE ANY ARGUMENTS FROM THE ARGPARSER BELOW
    parser = argparse.ArgumentParser(description='Page Rank Project')
    parser.add_argument('-i', '--infile',  help='Input file adjacency information of graph', default='testCaseSmall.txt', dest='inFileName')
    parser.add_argument('-a', '--alpha', help='Alpha Value', type=float, default=.85, dest='alpha')
    parser.add_argument('-e', '--epsilon', help='Epsilon Value for convergence test.', type=float, default=1e-10, dest='eps')
    parser.add_argument('-p', '--plot', help='Execute algorithm (0=default) or plot pregenerated results instead (1)', type=int, choices=[0,1], default=0, dest='plot')
    parser.add_argument('-s', '--selfloop', help='Use Self Loops to handle sinks (0) or Type 3 method (1=default).', type=int, choices=[0,1], default=1, dest='sinkHandling')
    #args for autograder, DO NOT MODIFY
    parser.add_argument('-n', '--sName',  help='Student name, used by autograder', default='GT', dest='studentName')	
    parser.add_argument('-z', '--autograde',  help='Autograder-called (2) or not (1=default)', type=int, choices=[1, 2], default=1, dest='autograde')	
    args = parser.parse_args(  ) 
    
    #DO NOT MODIFY ANY OF THIS CODE :     
    
    #input file name is used to build names for output files to save rank vector and ranked ordering of nodes  
    startTime = time.time()
    print('\nMaking pagerank object using input file {} and alpha {} and {} sink handling.'.format(args.inFileName, args.alpha, 'self-loop' if args.sinkHandling==0 else 'Type 3'))


    prObj = PageRank(args.inFileName, alpha=float(args.alpha), sinkHandling=args.sinkHandling)
    prMadeTime = time.time()
    print('\nPageRank object made.  Elapsed time : {} seconds'.format(prMadeTime - startTime))

    if (args.autograde == 2):
        util.autogradePR(prObj, args, prMadeTime)
        return   
    elif (args.plot) :
        #make sure you have .75, .85 and .95 alpha results generated for a 
        #particular input file before calling plotRes.  Args-specified alpha value 
        #is ignored in plotRes, but file name and self-loop method is used
        plotRes(prObj)
        return
    else:   
        calcRes(prObj, args, prMadeTime)    
    
if __name__ == '__main__':
    main()
    