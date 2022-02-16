'''
ECE 172A, Homework 2 Maze Pathfinding
Author: regreer@ucsd.edu
Maze generator adapted from code by Łukasz Nojek
For use by UCSD ECE 172A students only.
'''

import matplotlib.pyplot as plt 
import numpy as np
import pickle
import sys


class Stack:
	def __init__ (self):
		self.items = []
	def push(self, item):
		self.items.append(item) #add item to the end of list
	def pop(self):
		self.items.pop() #remove the last item in the list

class Queue:
	def __init__ (self):
		self.items = []
	def enqueue(self, item):
		self.items.insert(0,item) #add item to the start of list
	def dequeue(self):
		self.items.pop() #remove the last item in the list

def DFS(maze, startNode, goalNode):

	stack = Stack()

	exploredNodes = []

	stack.push(startNode)
	currentNode = None
	iterations = 0

	while len(stack.items) != 0 and currentNode != goalNode: 


		currentNode = stack.items[-1]

		if not currentNode in exploredNodes:
			exploredNodes.append(currentNode)

		if maze[currentNode][0] == True and not (currentNode[0],currentNode[1]+1) in exploredNodes: #north
			newNode = (currentNode[0],currentNode[1]+1)
			stack.push(newNode)
		elif maze[currentNode][1] == True and not (currentNode[0]+1,currentNode[1]) in exploredNodes: #east
			newNode = (currentNode[0]+1,currentNode[1])
			stack.push(newNode)
		elif maze[currentNode][2] == True and not (currentNode[0],currentNode[1]-1) in exploredNodes: #south
			newNode = (currentNode[0],currentNode[1]-1)
			stack.push(newNode)
		elif maze[currentNode][3] == True and not (currentNode[0]-1,currentNode[1]) in exploredNodes: #west
			newNode = (currentNode[0]-1,currentNode[1])
			stack.push(newNode)
		else:
			stack.pop()

		iterations = iterations + 1

	if len(stack.items) == 0:
		print("no path found")
		return 1
	else:
		print("path found")
		print("\n")
		print(iterations)
		print("iterations")
		return stack.items, exploredNodes


def BFS(maze, startNode, goalNode):

	queue = Queue()

	exploredNodes = []

	queue.enqueue(startNode)
	currentNode = None
	iterations = 0

	adjacentNodes = {}

	while len(queue.items) != 0 and currentNode != goalNode: 

		currentNode = queue.items[-1]

		if not currentNode in exploredNodes:
			exploredNodes.append(currentNode)

		if maze[currentNode][0] == True and not (currentNode[0],currentNode[1]+1) in exploredNodes: #north
			newNode = (currentNode[0],currentNode[1]+1)
			queue.enqueue(newNode)
			adjacentNodes[newNode] = currentNode
		if maze[currentNode][1] == True and not (currentNode[0]+1,currentNode[1]) in exploredNodes: #east
			newNode = (currentNode[0]+1,currentNode[1])
			queue.enqueue(newNode)
			adjacentNodes[newNode] = currentNode
		if maze[currentNode][2] == True and not (currentNode[0],currentNode[1]-1) in exploredNodes: #south
			newNode = (currentNode[0],currentNode[1]-1)
			queue.enqueue(newNode)
			adjacentNodes[newNode] = currentNode
		if maze[currentNode][3] == True and not (currentNode[0]-1,currentNode[1]) in exploredNodes: #west
			newNode = (currentNode[0]-1,currentNode[1])
			queue.enqueue(newNode)
			adjacentNodes[newNode] = currentNode
		
		queue.dequeue()

		iterations = iterations + 1

	if len(queue.items) == 0:
		print("no path found")
		return 1

	else:
		foundPath = []
		currentNode = goalNode

		while currentNode != startNode:
			currentNode = adjacentNodes[currentNode]
			print(currentNode)
			foundPath.append(currentNode)

		print("path found")
		print("\n")
		print(iterations)
		print("iterations")
		return foundPath, exploredNodes





def draw_path(final_path_points, other_path_points):
	'''
	final_path_points: the list of points (as tuples or lists) comprising your final maze path. 
	other_path_points: the list of points (as tuples or lists) comprising all other explored maze points. 
	(0,0) is the start, and (49,49) is the goal.
	Note: the maze template must be in the same folder as this script.
	'''
	im = plt.imread('172maze2021.png')
	x_interval = (686-133)/49
	y_interval = (671-122)/49
	plt.imshow(im)
	fig = plt.gcf()
	ax = fig.gca()
	circle_start = plt.Circle((133,800-122), radius=4, color='lime')
	circle_end = plt.Circle((686, 800-671), radius=4, color='red')
	ax.add_patch(circle_start)
	ax.add_patch(circle_end)
	for point in other_path_points:
		if not (point[0]==0 and point[1]==0) and not (point[0]==49 and point[1]==49):
			circle_temp = plt.Circle((133+point[0]*x_interval, 800-(122+point[1]*y_interval)), radius=4, color='blue')
			ax.add_patch(circle_temp)
	for point in final_path_points:
		if not (point[0]==0 and point[1]==0) and not (point[0]==49 and point[1]==49):
			circle_temp = plt.Circle((133+point[0]*x_interval, 800-(122+point[1]*y_interval)), radius=4, color='yellow')
			ax.add_patch(circle_temp)
	plt.show()

### Your Work Below: 


maze = pickle.load(open("172maze2021.p","rb"))

start = (0,0)
goal = (49,49)

pathNodes, exploredNodes = DFS(maze,start,goal)
draw_path(pathNodes, exploredNodes)

pathNodes, exploredNodes = BFS(maze,start,goal)
draw_path(pathNodes, exploredNodes)