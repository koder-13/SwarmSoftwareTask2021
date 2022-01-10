import sys

from numpy.lib.function_base import diff

from api import *
from time import sleep
import numpy as np
import random as r


#######    YOUR CODE FROM HERE #######################
import math
import time

grid =[]
neighs=[[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0]]
goals = []
goalList = get_original_greenZone_list()
itn = 0

class Node:
	def __init__(self,point,parent= None, value=1):
		self.value = value  #0 for blocked, 1 for unblocked, 2 for red nodes
		self.point = point
		self.parent = parent
		self.move = None
		self.H = 0
		self.G = 0
		self.f = self.G + self.H

def initialise():
	'''
	Work:
	Initialises global variable grid with a list of nodes with subscripts as respective x and y coordinates
	The value attribute of the node is given as:
	2 for red zones,
	0 for obstacle zones and
	1 for green or walkable zones.
	'''

	global grid
	obstacles = get_obstacles_list()
	redzones = get_redZone_list()

	for i in range(200):
		grid.append([])
		for j in range(200):
			grid[i].append(Node([i,j]))

	for obstacle in obstacles:
		for i in range(obstacle[0][0], obstacle[3][0]+1):
			for j in range(obstacle[0][1], obstacle[1][1]+1):
				grid[i][j].value = 0

	for redzone in redzones:
		for i in range(redzone[0][0], redzone[3][0]+1):
			for j in range(redzone[0][1], redzone[1][1]+1):
				grid[i][j].value = 2


def goal_centroids():
	'''
    Returns:
        goal_mean    list     list of coordinates of centroids of goals

	Work:
	    Finds and returns the coordinates of centroid of each goal
	'''
	global goalList
	goal_mean = []
    #Store mean locations of each goal in goalList
	for i, goal in enumerate(goalList):
		x = (goal[0][0] + goal[3][0])*0.5
		y = (goal[0][1] + goal[1][1])*0.5
		node = Node([x,y])
		node.H = 800
		goal_mean.append(node)
	return goal_mean

def isValid(pt):
	'''
	Inputs:
        pt        list        Set of x, y coordinates

    Returns:
        bool    True, if it lies on the map, False if outside it

	Work:
	    Initialises global variable grid with a list of nodes with subscripts as respective x and y coordinates
		The value attribute of the node is given as:
	    2 for red zones,
	    0 for obstacle zones and
	    1 for green or walkable zones.
	'''
	return pt[0]>=0 and pt[1]>=0 and pt[0]<200 and pt[1]<200


def neighbours(point):
	'''
	Inputs:
        point        list        Set of x, y coordinates

    Returns:
        links        list        List of index and corresponding coordinates of neighbours

	Work:
	    Returns the list of valid neighbouring nodes to the node whose coordinates are passed
	'''
	global neighs
	x,y = point.point
	links=[]
	for i in range(len(neighs)):
		newX = x+neighs[i][0]
		newY = y+neighs[i][1]
		if not isValid((newX,newY)):
			continue
		links.append([i+1,[newX, newY]])

	return links


def euclidean(pt1 , pt2):
	'''
	Inputs:
        pt1, pt2        lists        Set of x, y coordinates of the two points

    Returns:
        float        List of index and corresponding coordinates of neighbours

	Work:
	    Returns the decimal value of the euclidean distance between two points
	'''
	return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def best_path (botId, goal_mean, weights):
	'''
	Inputs:
        botId        int     The ID of the bot
		goal_mean    list    List of x,y coordinates of centroid of goal
		weights      list    Cost matrix of edges

    Returns:
        best_path     list    List of indices of goals

	Work:
	    Returns the path as the order of goals, while trying to minimise sum of the distances - distance of bot from goal,
		total cost for going to nearest next goal 
	'''
	min_cost = 8000
	path_min = []

	if len(goal_mean) == 0:
		return path_min

	for i in range(len(goal_mean)):
		arr = weights.copy()
		cost = 0
		check_path= [i]
		index = i
		arr[:, index] =  8000
		for j in range(len(goal_mean)-1):
			next = np.argmin(arr[index].copy())
			cost += weights[index][next]
			arr[:, index] =  8000
			index = next
			check_path.append(index)
		if cost < min_cost:
			min_cost = cost
			path_min = check_path

	min_cost = 8000
	bot = (get_botPose_list())[botId]
	best_bet = euclidean(bot, goal_mean[0].point)
	best_path = path_min.copy()
	l = len(path_min)
	for i in range(l):
		index = path_min.index(i)
		if index == 0 or index == 1:
			continue
		criteria = euclidean(bot, goal_mean[i].point) - euclidean(goal_mean[i].point, goal_mean[path_min[index - 1]].point)
        #bot's distance to that node minus the edge untraversed
		if criteria < best_bet:
			best_bet = criteria
			best_path = path_min[index:l]
			if euclidean(goal_mean[i].point, bot) > euclidean(goal_mean[path_min[l-1]].point, bot):
				best_path = path_min[::-1]
			add = path_min[0:index]
			if euclidean(goal_mean[best_path[-1]].point, goal_mean[path_min[0]].point) < euclidean(goal_mean[path_min[index-1]].point, goal_mean[best_path[-1]].point):
				best_path += add
			else:
				best_path += add[::-1]

	path_min = path_min[::-1]
	for i in range(l):
		index = path_min.index(i)
		if index == 0 or index == 1:
			continue
		criteria = euclidean(bot, goal_mean[i].point) - euclidean(goal_mean[i].point, goal_mean[path_min[index - 1]].point)
        #bot's distance to that node minus the edge untraversed
		if criteria < best_bet:
			best_bet = criteria
			best_path = path_min[index:l]
			if euclidean(goal_mean[i].point, bot) > euclidean(goal_mean[path_min[l-1]].point, bot):
				best_path = path_min[::-1]
			add = path_min[0:index]
			if euclidean(goal_mean[best_path[-1]].point, goal_mean[path_min[0]].point) < euclidean(goal_mean[path_min[index-1]].point, goal_mean[best_path[-1]].point):
				best_path += add
			else:
				best_path += add[::-1]

	return best_path


def find_weights(list):
	'''
	Inputs:
        list       list        List of centroid coordinates of goals

    Returns:
        weight     2Dmatrix    Cost matrix of edges

	Work:
	    Returns the cost matrix of edges between every set of points
	'''
	weights = np.zeros((len(list), len(list)))
	for i in range(len(list)):
		for j in range(len(list)):
			if i == j:
				weights[i][j] = 8000
			else:
				weights[i][j] = euclidean(list[i].point,list[j].point)
	return weights

def check_move(current, next):
	'''
	Inputs:
        current      Node object      Current node in the path
		next         Node object      Next node in the path

    Returns:
        moveType     int              movement type, as described in the README

	Work:
	    Returns movetype based on difference in x and y coordinates
	'''
	x_diff = next.point[0] - current.point[0]
	y_diff = next.point[1] - current.point[1]
	if x_diff == 0:
		if y_diff == 1:
			return 4
		else:
			return 8
	elif x_diff == 1:
		if y_diff == 1:
			return 5
		elif y_diff == 0:
			return 6
		else:
			return 7
	else:
		if y_diff == 1:
			return 3
		elif y_diff == 0:
			return 2
		else:
			return 1


def astar(startCoord, endCoord, botId = 0):
	'''
	Inputs:
        startCoord       list      Coordinates of start
		endCoord         list      Coordinates of goal
		botId            int       The ID of the bot

    Returns:
                         bool      returns True if mission_complete is True or if goal has been reached

	Work:
	    Implements A star algorithm to find the shortest path between start and goal nodes
	'''
	openList = []
	closedList = []

	end = Node(endCoord)
	start = Node(startCoord)

	openList.append(start)
	current = start
					
	#Loop until no nodes in openList(no path)
	while len(openList) > 0:
        
        #Search lowest fcost node and make it the current node
		i = 0
		current = openList[0]
		for j, node in enumerate(openList):
			if node.f < current.f or (int(current.f) == int(node.f) and node.G < current.G):
				current = node
				i = j

        #Remove current from openlist and add it to closedlist
		openList.pop(i)
		closedList.append(current)

        #If we reached the goal node, backtrack the path by going to the parent of the previous node
		if current.point[0] <= end.point[0] + 1 and current.point[0] >= end.point[0]-1 and current.point[1] <= end.point[1]+1 and current.point[1] >= end.point[1]-1:
			path = []
			x = current
			while x is not None:
				path.append(x)
				x = x.parent
			path = path[::-1]
            
            #Path is now traced. Send move_type to send_command function using check_move
			for i, node in enumerate(path):
				if i == (len(path) - 1):
					continue
				successful_move, mission_complete = send_command(botId, check_move(node, path[i+1]))
				print(check_move(node, path[i+1]))
				if successful_move:
					print("YES")
				else:
					print("NO")
				if mission_complete:
					print("MISSION COMPLETE")
					return True
			return True

		#Check adjacent squares from the steps - diagonal and straight movement
		for value, neigh in neighbours(current):

            #Go to next node if this one is in closedlist
			broken = 0
			for node in closedList:
				if node.point == neigh:
					broken = 1
					break
			if broken == 1:
				continue

            #Go to next node if this one is an obstacle
			if grid[neigh[0]][neigh[1]].value == 0:
				continue
			
			#Record gcost
			if value % 2 == 0:
				gcost = current.G + 1
			else:
				gcost = current.G + 1.4

            #Check if it is already in openlist - if it is then is this a better path?
			for node in openList:
				if node.point == neigh:
					broken = 1
					if node.G >= gcost:
						node.G = gcost
						node.f = gcost + node.H
						node.parent = current
					break
			if broken == 1:
				continue

			child = Node(neigh, current)
                
            #Record f, H costs
			child.G = gcost
			child.H = (euclidean(child.point, end.point)) * (grid[neigh[0]][neigh[1]].value)
			child.f = child.G + child.H

            #Add to openList
			openList.append(child)


def navigate(botId, goals):
	"""
	Inputs:
		botId            int       The ID of the bot
		goals            list      List of coordinates of goals     

	Work:
	    Calls best_path function to find best path for bot through assigned goals 
	"""
	path = best_path(botId, goals, find_weights(goals))
	for i in path:
		bots = get_botPose_list()
		astar(bots[botId], goals[i].point, botId)


########## Level 1 ##########
def level1(botId):
	global goalList
	endList = goalList[0][2]
	startCoord=  get_botPose_list()
	initialise()                      #initialises grid
	astar(startCoord[0], endList)     #calls astar function to find path


def level2(botId):
	goal_mean = goal_centroids()
	initialise()
	weights = find_weights(goal_mean)
	path_min = best_path(botId, goal_mean, weights)
	for index in path_min:
		bot = (get_botPose_list())[0]
		astar(bot, goal_mean[index].point) 


def level3(botId):
	goal_mean = goal_centroids()
	bots = get_botPose_list()
	goals = [[],[]]
	initialise()

	for goal in goal_mean:
		min = 8000
		for i in range(len(bots)):
			if min > euclidean(bots[i], goal.point):
				min = euclidean(bots[i], goal.point)
				index = i
		goals[index].append(goal)
	navigate(botId, goals[botId])


def level4(botId):
	goal_mean = goal_centroids()
	bots = get_botPose_list()
	goals = [[],[],[],[],[],[],[],[]]
	initialise()
	for goal in goal_mean:
		min = 8000
		for i in range(len(bots)):
			if min > euclidean(bots[i], goal.point):
				min = euclidean(bots[i], goal.point)
				index = i
		goals[index].append(goal)
	print("Goals length for bot ",botId,": ",len(goals[botId]))

	navigate(botId, goals[botId])


def level5(botId):
	goal_mean = goal_centroids()
	bots = get_botPose_list()
	goals = [[],[]]
	initialise()
	for goal in goal_mean:
		min = 8000
		for i in range(len(bots)):
			if min > euclidean(bots[i], goal.point):
				min = euclidean(bots[i], goal.point)
				index = i
		goals[index].append(goal)
	
	print("Goals length for bot ",botId,": ",len(goals[botId]))

	navigate(botId, goals[botId])


def level6(botId):    
	goal_mean = goal_centroids()
	bots = get_botPose_list()
	goals = [[],[],[],[],[],[],[],[]]
	initialise()
	for goal in goal_mean:
		min = 8000
		for i in range(len(bots)):
			if min > euclidean(bots[i], goal.point):
				min = euclidean(bots[i], goal.point)
				index = i
		goals[index].append(goal)
	print("Goals length for bot ",botId,": ",len(goals[botId]))

	navigate(botId, goals[botId])

""" 
NOTE: Alternative approaches that I tried for best_path function -- to find best path through goal nodes [Traveling Salesman Problem]

---- Greedy Approach ----
(Worked but present best_path function works better)
	min_cost = 8000
	path_min = []

	if len(goal_mean) == 0:
		return path_min

	bot = (get_botPose_list())[botId]
	for i in range(len(goal_mean)):
		arr = weights.copy()
		cost = euclidean(bot, goal_mean[i].point)
		check_path= [i]
		index = i
		arr[:, index] =  8000
		for j in range(len(goal_mean)-1):
			next = np.argmin(arr[index].copy())
			cost += weights[index][next]
			arr[:, index] =  8000
			index = next
			check_path.append(index)
		if cost < min_cost:
			min_cost = cost
			path_min = check_path
	return path_min
 

---- 2-opt swap approach ----
(Didn't make much difference to path shown by greedy approach)
best = route
    improvement_factor = 1
    best_dist = 8000
    cnt = 0
    while improvement_factor > 0.01:
        previous_best = calc_cost(best, goalList)
        for swap_first in range(1, len(route) - 2):
            for swap_last in range(swap_first + 1, len(route)-1):
                before_start = best[swap_first - 1]
                start = best[swap_first]
                end = best[swap_last]
                after_end = best[swap_last+1]
                before = manhattan(goalList[before_start].point, goalList[start].point) + manhattan(goalList[end].point, goalList[after_end].point)
                after = manhattan(goalList[before_start].point, goalList[end].point) + manhattan(goalList[start].point, goalList[after_end].point)
                if after < before:
                    new_route = np.concatenate((best[0:swap_first],best[swap_last:-len(best) + swap_first - 1:-1],best[swap_last + 1:len(best)]))
                    new_route = new_route.tolist()
                    best = new_route
                    best_dist = calc_cost(best, goalList)
        route = best
        cnt +=1
        improvement_factor = 1 - best_dist/previous_best



---- Hamiltonian cycle approach ----

def Hamiltonian(k):
	while True:
		NextVertex(k)
		if x[k] == 0:
			return
		if k == n:
			print(x[1:n])
		else:
			Hamiltonian(k+1)

def NextVertex(k):
	while True:
		x[k] = (x[k+1])mod(n+1)
		if x[k] == 0:
			return
		for j in range(k):
			while True:
				if x[j] == x[k]:
					break
				if j == k:
					if k<n or k==n:
						return


"""


#######    DON'T EDIT ANYTHING BELOW  #######################

if  __name__=="__main__":
	botId = int(sys.argv[1])
	level = get_level()
	if level == 1:
		level1(botId)
	elif level == 2:
		level2(botId)
	elif level == 3:
		level3(botId)
	elif level == 4:
		level4(botId)
	elif level == 5:
		level5(botId)
	elif level == 6:
		level6(botId)
	else:
		print("Wrong level! Please restart and select correct level")