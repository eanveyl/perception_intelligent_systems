'''
MIT License
Copyright (c) 2019 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.

Modified by Eduardo Nunez for Perception and Decision Making for higher efficiency and execution speed.
'''

import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from collections import deque
from tqdm import tqdm


class Line():
	def __init__(self, p0, p1):
		self.p = np.array(p0)
		self.dirn = np.array(p1) - np.array(p0)
		self.dist = np.linalg.norm(self.dirn)
		self.dirn /= self.dist # normalize

	def path(self, t):
		return self.p + t * self.dirn

def Intersection(line, center, radius):
	a = np.dot(line.dirn, line.dirn)
	b = 2 * np.dot(line.dirn, line.p - center)
	c = np.dot(line.p - center, line.p - center) - radius * radius

	discriminant = b * b - 4 * a * c
	if discriminant < 0:
		return False

	t1 = (-b + np.sqrt(discriminant)) / (2 * a);
	t2 = (-b - np.sqrt(discriminant)) / (2 * a);

	if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
		return False

	return True

def distance(x, y):
	return np.linalg.norm(np.array(x) - np.array(y))

def isThruObstacle(line, obstacles, radius):
	for obs in obstacles:
		if Intersection(line, obs, radius):
			return True
	return False

def isInObstacle(vex, obstacles, radius):
	distances_to_curr_point = np.linalg.norm(np.array(vex)-np.array(obstacles), axis=1)
	if distances_to_curr_point[distances_to_curr_point < radius].size > 0:
		return True  # we have an obstacle
	else:
		return False


def nearest(G, vex, obstacles, radius, stepSize):
	distances_to_curr_point = np.linalg.norm(np.array(vex)-np.array(G.vertices), axis=1)
	vertices_sorted_by_ascending_distance = np.argsort(distances_to_curr_point)

	for idx in vertices_sorted_by_ascending_distance:
		v = G.vertices[idx]
		
		invalid_node = False  # initialize default value
		for i in range(1, int(np.ceil(stepSize/(2*radius)))):
			dirn = (np.array(vex)-np.array(v))
			test_point = v + (dirn/np.linalg.norm(dirn)) * (2*radius*i)  # this is a point in the direction of vex, with a distance of 
			if isInObstacle(test_point, obstacles, radius):
				invalid_node = True
				break
		if invalid_node:
			continue

		return v, idx


def newVertex(randvex, nearvex, stepSize):
	dirn = np.array(randvex) - np.array(nearvex)
	length = np.linalg.norm(dirn)
	dirn = (dirn / length) * stepSize#min (stepSize, length)

	newvex = (nearvex[0]+dirn[0], nearvex[1]+dirn[1])
	return newvex


def window(startpos, endpos):  # Define seach window - 2 times of start to end rectangle
	width = endpos[0] - startpos[0]
	height = endpos[1] - startpos[1]
	winx = startpos[0] - (width / 2.)
	winy = startpos[1] - (height / 2.)
	return winx, winy, width, height


def isInWindow(pos, winx, winy, width, height):  # Restrict new vertex insides search window
	if winx < pos[0] < winx+width and \
		winy < pos[1] < winy+height:
		return True
	else:
		return False


class Graph:  # Define graph
	def __init__(self, startpos, endpos):
		self.startpos = startpos
		self.endpos = endpos

		self.vertices = [startpos]
		self.edges = []
		self.success = False

		self.vex2idx = {startpos:0}
		self.neighbors = {0:[]}
		self.distances = {0:0.}

		self.sx = endpos[0] - startpos[0]
		self.sy = endpos[1] - startpos[1]

	def add_vex(self, pos):
		try:
			idx = self.vex2idx[pos]
		except:
			idx = len(self.vertices)
			self.vertices.append(pos)
			self.vex2idx[pos] = idx
			self.neighbors[idx] = []
		return idx

	def add_edge(self, idx1, idx2, cost):
		self.edges.append((idx1, idx2))
		self.neighbors[idx1].append((idx2, cost))
		self.neighbors[idx2].append((idx1, cost))


	def randomPosition(self):
		# rx = random()
		# ry = random()

		# posx = self.startpos[0] - (self.sx / 2.) + rx * self.sx * 2
		# posy = self.startpos[1] - (self.sy / 2.) + ry * self.sy * 2
		my_generator = np.random.default_rng()
		posx = my_generator.uniform(low=-3, high=6)
		posy = my_generator.uniform(low=-10, high=5)
		return posx, posy


def RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize):
	G = Graph(startpos, endpos)

	for _ in tqdm(range(n_iter)):
		randvex = G.randomPosition()
		if isInObstacle(randvex, obstacles, radius):
			continue

		nearvex, nearidx = nearest(G, randvex, obstacles, radius, stepSize)
		if nearvex is None:
			continue

		newvex = newVertex(randvex, nearvex, stepSize)

		newidx = G.add_vex(newvex)
		dist = distance(newvex, nearvex)
		G.add_edge(newidx, nearidx, dist)
		G.distances[newidx] = G.distances[nearidx] + dist

		# update nearby vertices distance (if shorter)
		for vex in G.vertices:
			if vex == newvex:
				continue

			dist = distance(vex, newvex)
			if dist > radius:
				continue

			line = Line(vex, newvex)
			if isThruObstacle(line, obstacles, radius):
				continue

			idx = G.vex2idx[vex]
			if G.distances[newidx] + dist < G.distances[idx]:
				G.add_edge(idx, newidx, dist)
				G.distances[idx] = G.distances[newidx] + dist

		dist = distance(newvex, G.endpos)
		if dist < 3 * radius:  # a bit of god's hand
			endidx = G.add_vex(G.endpos)
			G.add_edge(newidx, endidx, dist)
			try:
				G.distances[endidx] = min(G.distances[endidx], G.distances[newidx]+dist)
			except:
				G.distances[endidx] = G.distances[newidx]+dist

			G.success = True
			print('success')
			break
	
	
	for _ in tqdm(range(int(0.20*n_iter))):
		randvex = G.randomPosition()
		if isInObstacle(randvex, obstacles, radius):
			continue

		nearvex, nearidx = nearest(G, randvex, obstacles, radius, stepSize)
		if nearvex is None:
			continue

		newvex = newVertex(randvex, nearvex, stepSize)

		newidx = G.add_vex(newvex)
		dist = distance(newvex, nearvex)
		G.add_edge(newidx, nearidx, dist)
		G.distances[newidx] = G.distances[nearidx] + dist

		# update nearby vertices distance (if shorter)
		for vex in G.vertices:
			if vex == newvex:
				continue

			dist = distance(vex, newvex)
			if dist > radius:
				continue

			line = Line(vex, newvex)
			if isThruObstacle(line, obstacles, radius):
				continue

			idx = G.vex2idx[vex]
			if G.distances[newidx] + dist < G.distances[idx]:
				G.add_edge(idx, newidx, dist)
				G.distances[idx] = G.distances[newidx] + dist

	return G


def RRT(startpos, endpos, obstacles, n_iter, radius, stepSize):  # RRT algorithm
	G = Graph(startpos, endpos)

	for _ in tqdm(range(n_iter)):
		randvex = G.randomPosition()
		if isInObstacle(randvex, obstacles, radius):
			continue

		nearvex, nearidx = nearest(G, randvex, obstacles, radius, stepSize)  # this checks at the same time if the 
		# potential nearest vertex and the random vertex can be connected with a line without hitting any 
		# obstacles...
		if nearvex is None:
			continue

		newvex = newVertex(randvex, nearvex, stepSize)  # ... that's why we don't need to check here if 
		# the new vertex collides with something else.

		newidx = G.add_vex(newvex)
		#print("Added vertex at {},{}".format(newvex[0], newvex[1]))
		dist = distance(newvex, nearvex)
		G.add_edge(newidx, nearidx, dist)

		dist = distance(newvex, G.endpos)
		if dist < 3 * radius:  # changed from 2 to add a little bit of god's hand
			endidx = G.add_vex(G.endpos)
			G.add_edge(newidx, endidx, dist)
			G.success = True
			print('success')
			break
	return G


def dijkstra(G):  # Dijkstra algorithm for finding shortest path from start position to end.
	srcIdx = G.vex2idx[G.startpos]
	dstIdx = G.vex2idx[G.endpos]

	# build dijkstra
	nodes = list(G.neighbors.keys())
	dist = {node: float('inf') for node in nodes}
	prev = {node: None for node in nodes}
	dist[srcIdx] = 0

	while nodes:
		curNode = min(nodes, key=lambda node: dist[node])
		nodes.remove(curNode)
		if dist[curNode] == float('inf'):
			break

		for neighbor, cost in G.neighbors[curNode]:
			newCost = dist[curNode] + cost
			if newCost < dist[neighbor]:
				dist[neighbor] = newCost
				prev[neighbor] = curNode

	# retrieve path
	path = deque()
	curNode = dstIdx
	while prev[curNode] is not None:
		path.appendleft(G.vertices[curNode])
		curNode = prev[curNode]
	path.appendleft(G.vertices[curNode])
	return list(path)


def plot(G, obstacles, radius, path=None):  # Plot RRT, obstacles and shortest path
	px = [x for x, y in G.vertices]
	py = [y for x, y in G.vertices]
	fig, ax = plt.subplots()

	for obs in obstacles:
		circle = plt.Circle(obs, radius, color='red')
		ax.add_artist(circle)

	ax.scatter(px, py, c='cyan')
	ax.scatter(G.startpos[0], G.startpos[1], c='black')
	ax.scatter(G.endpos[0], G.endpos[1], c='black')

	lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
	lc = mc.LineCollection(lines, colors='green', linewidths=2)
	ax.add_collection(lc)

	if path is not None:
		paths = [(path[i], path[i+1]) for i in range(len(path)-1)]
		lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
		ax.add_collection(lc2)

	#ax.autoscale()
	ax.margins(0.1)
	plt.show()


if __name__ == '__main__':  # this is the demo
	startpos = (0,0)
	endpos = (0.35, -1.55)
	obstacles = [(1,1)]
	n_iter = 100
	radius = 0.1
	stepSize = 0.5
	 
	G = RRT(startpos, endpos, obstacles, n_iter, radius, stepSize)

	if G.success:
		path = dijkstra(G)
		print(path)
		plot(G, obstacles, radius, path)
	else:
		plot(G, obstacles, radius)