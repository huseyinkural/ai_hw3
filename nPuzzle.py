
from datetime import datetime
from collections import deque
import random
import numpy as np



#goalState = [[1, 2, 3, 4],
#[5, 6, 7, 8],
#[9, 10, 11, 12],
#[13, 14, 0, 15]]

# # Easy Solvable
#initial = [[0, 2, 3], [1, 4, 5], [8, 7, 6]]
#goalState = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]


#initial = [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 0],[12, 13, 15, 14]]
#goalState = [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12],[13, 14, 0, 15]]

#initial = [[1, 2, 3],
#[4, 5, 0],
#[6, 7, 8]]

#initial = [[10, 17, 22, 11, 2], [6, 7, 20, 24, 21], [14, 12, 5, 23, 1], [16, 18, 13, 15, 9], [4, 8, 3, 19, 0]]
#goalState = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 0]]

#initial = [[2, 1, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
#goalState = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]



#initial = [[1, 7, 2],[0, 5, 3],[4, 8, 6]]

#goalState = [[1, 2, 3],[4, 5, 6],[7, 8, 0]]

#initial = [[10, 17, 22, 11, 2], [6, 7, 20, 24, 21], [14, 12, 5, 23, 1], [16, 18, 13, 15, 9], [4, 8, 3, 19, 0]]
#goalState = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 0]]


#initial = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
#goalState = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 10, 12], [13, 14, 11, 15]]

#excel ex1
#initial = [[1, 7, 2],[0, 5, 3],[4, 8, 6]]
#goalState = [[1, 2, 3],[4, 5, 6],[7, 8, 0]]

#excel ex2
#initial = [[0, 2, 3], [1, 4, 5], [8, 7, 6]]
#goalState = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]


#excel ex3
#initial = [[2, 1, 5], [0, 3, 6], [4, 7, 8]]
#goalState = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

#excel ex4
initial = [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 0],[12, 13, 15, 14]]
goalState = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]


# heap functions:


# Push item onto heap, maintaining the heap invariant.
def heappush(heap, item):
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)


# Pop the smallest item off the heap, maintaining the heap invariant.
def heappop(heap):
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt


# 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
# is the index of a leaf with a possibly out-of-order value.  Restore the
# heap invariant.
def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem[1] < parent[1]:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos][1] < heap[rightpos][1]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now. Put new item there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)


""""-----------------------------------------------------------------------------------------------------"""

"""class Node is used as nodes for the Goal tree class, in addition to having the expansion function and any other
cost functions."""


class Node:
    """constructor of the Node class."""
    def __init__(self, state, g, parent=None, action=None, children=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = children
        self.g = g

    """"return if self.state = state, false otherwise."""
    def equal(self, state):
        for i in range(len(state)):
            for j in range(len(state)):
                if self.state[i][j] != state[i][j]:
                    return False
        return True

    """calculate the distance between every square and its goal position measured along axes at right angles"""
    def manhattan_distance(self):
        distance = 0
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                if self.state[i][j] != 0:
                    x, y = divmod(self.state[i][j] - 1, len(self.state))
                    distance += abs(x - i) + abs(y - j)
        return distance

    def CalculateManhattanDistance(self):
        arr = [0]*(len(self.state)*len(self.state))
        brr = [0]*(len(self.state)*len(self.state))

        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                arr[self.state[i][j]] = [i, j]

        for i in range(len(goalState)):
            for j in range(len(goalState[i])):
                brr[goalState[i][j]] = [i, j]
        dist = 0
        for i in range(1, len(arr)):
            dist += abs(arr[i][0]-brr[i][0]) + abs(arr[i][1]-brr[i][1])
        print(dist)
        return dist

    """return the estimated cost to reach the goal state."""
    def get_h(self):
        return self.CalculateManhattanDistance(self)

    def get_f(self):
        return self.get_h(self) + self.g

    """return the list of possible states after exactly one move."""
    def expand(self):
        x, y = self.find(self.state, 0)
        """ Direc contains position values for moving the blank space in either of
            the 4 directions [up,down,left,right] . """
        direc = {"Left": [x, y - 1], "Right": [x, y + 1], "Up": [x - 1, y], "Down": [x + 1, y]}
        childs = []
        for i in direc:
            child = self.move(self.state, x, y, direc[i][0], direc[i][1])
            if child is not None:
                child_node = Node(child, self.g + 1, self, i)
                childs.append(child_node)
        self.children = childs
        return childs

    """move the blank space in the given direction and if the position value are out
                of limits then return None """
    def move(self, state, x1, y1, x2, y2):
        if (0 <= x2 < len(self.state)) and (0 <= y2 < len(self.state)):
            # You can also use { copy.deepcopy(state) } by importing copy library, but it's a little bit slower than
            # the explicit one down there.
            temp_state = self.copy(state)

            temp = temp_state[x2][y2]
            temp_state[x2][y2] = temp_state[x1][y1]
            temp_state[x1][y1] = temp
            return temp_state
        else:
            return None

    """specifically used to find the position of the blank space """
    def find(self, state, x):
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                if state[i][j] == x:
                    return i, j

    """ Copy function to create a similar matrix of the given node"""
    def copy(self, state):
        temp = []
        for row in state:
            temp_row = []
            for element in row:
                temp_row.append(element)
            temp.append(temp_row)
        return temp


""""-----------------------------------------------------------------------------------------------------"""

""" Goal tree is the main class of the software, 
having the search methods and the closed list, in addition to other things """


class GoalTree:
    # constructor for the class goal tree.
    def __init__(self, initial_state):
        self.root = Node(initial_state, 0)

    # Goal test function.
    @staticmethod
    def is_goal(state):
        if state == goalState:
            return True
        return False

    # the interactive method of the class
    def solve(self, strategy):
        #bfs
        if strategy.lower() == '1':
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.breadth_first()

        #dfs
        elif strategy.lower() == '2':
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.depth_first()

        #ucs
        elif strategy.lower() == '3':
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.uniform_cost()

        #dls
        elif strategy.lower() == '4':
            limit = int(input('Enter a limit -> '))
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.depth_limited(limit)

        #iterative deep
        elif strategy.lower() == '5':
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.iterative_deepening()

        #a*
        elif strategy.lower() == '6':
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.a_star()

        #greedy
        elif strategy.lower() == '7':
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.greedy()

        return sol_state, sol, g, processed_nodes, max_stored_nodes, flag, start

    """"-------------------------------------------------------------------------------------------"""
    """ Search methods:
            uninformed search methods: (breadth-first, depth-first, uniformed cost, depth limited, iterative deepening) """

    # find the shallowest solution.
    def breadth_first(self):
        node = Node(self.root.state, 0)
        self.root = node
        processed_nodes = 1
        max_stored_nodes = 1
        dim = len(node.state) * len(node.state)

        # case 1: if the initial state is the goal state
        if self.is_goal(node.state):
            sol = self.solution(node)
            return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

        # case 2: searching for the goal state
        frontier = deque([node])
        explored = set()
        while True:
            # Failed outcome, (i.e. didn't find the goal state)
            if len(frontier) == 0:
                sol = self.solution(self.root)
                # will return the root state instead of node state to indicate that we don't find the solution
                return self.root.state, sol, node.g, processed_nodes, max_stored_nodes, False

            stored = len(frontier)
            if stored > max_stored_nodes:
                max_stored_nodes = stored

            # checking for the goal state
            node = frontier.popleft()
            if self.is_goal(node.state):
                sol = self.solution(node)
                return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

            # recording visited states.
            temp = tuple(np.reshape(node.state, dim))
            explored.add(temp)
            children = node.expand()
            for child in children:
                child1 = tuple(np.reshape(child.state, dim))
                if child1 not in explored:
                    processed_nodes += 1
                    frontier.append(child)

    # find the deepest solution
    def depth_first(self):
        node = Node(self.root.state, 0)
        self.root = node
        processed_nodes = 1
        max_stored_nodes = 1
        dim = len(node.state) * len(node.state)

        # case 1: if the initial state is the goal state
        if self.is_goal(node.state):
            sol = self.solution(node)
            return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

        # case 2: searching for the goal state
        frontier = [node]
        explored = set()
        while True:
            # Failed outcome, (i.e. didn't find the goal state)
            if len(frontier) == 0:
                sol = self.solution(self.root)
                # will return the root state instead of node state to indicate that we don't find the solution
                return self.root.state, sol, node.g, processed_nodes, max_stored_nodes, False

            stored = len(frontier)
            if stored > max_stored_nodes:
                max_stored_nodes = stored

            # checking for the goal state
            node = frontier.pop()
            if self.is_goal(node.state):
                sol = self.solution(node)
                return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

            # recording visited states.
            temp = tuple(np.reshape(node.state, dim))
            explored.add(temp)
            children = node.expand()
            for child in children:
                child1 = tuple(np.reshape(child.state, dim))
                if child1 not in explored:
                    processed_nodes += 1
                    frontier.append(child)

    def uniform_cost(self):
        node = Node(self.root.state, 0)
        self.root = node
        processed_nodes = 1
        max_stored_nodes = 1
        dim = len(node.state) * len(node.state)

        # case 1: if the initial state is the goal state
        if self.is_goal(node.state):
            sol = self.solution(node)
            return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

        # case 2: searching for the goal state
        frontier = []
        heappush(frontier, (node, node.g))
        explored = set()
        while True:
            # Failed outcome, (i.e. didn't find the goal state)
            if len(frontier) == 0:
                sol = self.solution(self.root)
                # will return the root state instead of node state to indicate that we don't find the solution
                return self.root.state, sol, node.g, processed_nodes, max_stored_nodes, False

            stored = len(frontier)
            if stored > max_stored_nodes:
                max_stored_nodes = stored

            # checking for the goal state
            node = heappop(frontier)[0]
            if self.is_goal(node.state):
                sol = self.solution(node)
                return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

            # recording visited states.
            temp = tuple(np.reshape(node.state, dim))
            explored.add(temp)
            children = node.expand()
            # add unexplored states to frontier
            for child in children:
                child1 = tuple(np.reshape(child.state, dim))
                if child1 not in explored:
                    processed_nodes += 1
                    heappush(frontier, (child, child.g))

    # without explored set
    def depth_limited(self, limit):
        node = Node(self.root.state, 0)
        self.root = node
        processed_nodes = 1
        max_stored_nodes = 1

        # case 1: if the initial state is the goal state
        if self.is_goal(node.state):
            sol = self.solution(node)
            return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

        # case 2: searching for the goal state
        frontier = [node]
        # explored = set()
        while True:
            # Failed outcome, (i.e. didn't find the goal state)
            if len(frontier) == 0:
                sol = self.solution(self.root)
                # will return the root state instead of node state to indicate that we don't find the solution
                return self.root.state, sol, node.g, processed_nodes, max_stored_nodes, False

            stored = len(frontier)
            if stored > max_stored_nodes:
                max_stored_nodes = stored

            # checking for the goal state
            node = frontier.pop()
            if self.is_goal(node.state):
                sol = self.solution(node)
                return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

            # recording visited states.
            if limit >= node.g + 1:
                children = node.expand()
                for child in children:
                    processed_nodes += 1
                    frontier.append(child)

    def iterative_deepening(self):
        level = 0
        flag = False
        total_processed_nodes = 0
        final_max_stored_nodes = 0
        while not flag:
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.depth_limited(level)
            total_processed_nodes += processed_nodes
            if final_max_stored_nodes < max_stored_nodes:
                final_max_stored_nodes = max_stored_nodes
            level += 1
        return sol_state, sol, g, total_processed_nodes, final_max_stored_nodes, flag

    """-------------------------------------------------------------------------------------------"""
    """Informed search methods: (greedy search(best-first search), A*) """
    def greedy(self):
        node = Node(self.root.state, 0)
        self.root = node
        max_stored_nodes = 1
        processed_nodes = 1
        dim = len(node.state) * len(node.state)

        # case 1: if the initial state is the goal state
        if self.is_goal(node.state):
            sol = self.solution(node)
            return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

        # case 2: searching for the goal state
        frontier = []
        heappush(frontier, (node, node.CalculateManhattanDistance()))
        explored = set()
        while True:
            # Failed outcome, (i.e. didn't find the goal state)
            if len(frontier) == 0:
                sol = self.solution(node)
                # will return the root state instead of node state to indicate that we don't find the solution
                return self.root.state, sol, node.g, processed_nodes, max_stored_nodes, False

            stored = len(frontier)
            if stored > max_stored_nodes:
                max_stored_nodes = stored
            # checking for the goal state
            node = heappop(frontier)[0]
            if self.is_goal(node.state):
                sol = self.solution(node)
                return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

            # recording visited states.
            temp = tuple(np.reshape(node.state, dim))
            explored.add(temp)
            children = node.expand()
            # add unexplored states to frontier
            for child in children:
                child1 = tuple(np.reshape(child.state, dim))
                if child1 not in explored:
                    processed_nodes += 1
                    heappush(frontier, (child, child.CalculateManhattanDistance()))

    def a_star(self):
        node = Node(self.root.state, 0)
        self.root = node
        max_stored_nodes = 1
        processed_nodes = 1
        dim = len(node.state)*len(node.state)

        # case 1: if the initial state is the goal state
        if self.is_goal(node.state):
            sol = self.solution(node)
            return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

        # case 2: searching for the goal state
        frontier = []
        #heappush(frontier, (node, node.g + node.manhattan_distance()))
        heappush(frontier, (node, node.g + node.CalculateManhattanDistance()))
        explored = set()
        while True:

            # Failed outcome, (i.e. didn't find the goal state)
            if len(frontier) == 0:
                sol = self.solution(self.root)
                # will return the root state instead of node state to indicate that we don't find the solution
                return self.root.state, sol, node.g, processed_nodes, max_stored_nodes, False

            stored = len(frontier)
            if stored > max_stored_nodes:
                max_stored_nodes = stored

            # checking for the goal state
            node = heappop(frontier)[0]
            if self.is_goal(node.state):
                sol = self.solution(node)
                return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

            # recording visited states.
            temp = tuple(np.reshape(node.state, dim))
            explored.add(temp)
            children = node.expand()
            # add unexplored states to frontier
            for child in children:
                child1 = tuple(np.reshape(child.state, dim))
                if child1 not in explored:
                    processed_nodes += 1
                    heappush(frontier, (child, child.g + child.CalculateManhattanDistance()))

    @staticmethod
    def solution(node):
        sol = []
        current = node
        while current.parent is not None:
            sol.append(current.action)
            current = current.parent

        sol.reverse()
        return sol


""""-----------------------------------------------------------------------------------------------------"""
# Functions.


# check if a given state is solvable or not
def solvable(state):
    noi = number_of_inversion(state)
    n = len(state)
    rob = row_of_blank_from_bottom(state)
    if n % 2 == 1 and noi % 2 == 0:
        return True
    elif n % 2 == 0 and ((rob % 2 == 0 and noi % 2 == 1) or (rob % 2 == 1 and noi % 2 == 0)):
        return True
    else:
        return False


# generate random solvable state
def random_state(n):
    state1 = np.array(random.sample(range(n*n), n*n))
    state = np.reshape(state1, (n, n))
    while not solvable(state):
        state1 = np.array(random.sample(range(n * n), n * n))
        state = np.reshape(state1, (n, n))

    return state


"""-----------------------------------------------------------------------------------------------------"""


# Helper functions:
def number_of_inversion(state):
    row = []
    # turning the matrix into a raw
    n = len(state)
    for x in range(n):
        for y in range(n):
            row.append(state[x][y])
    noi = 0  # number of inversion (n.o.i)
    for i in range(len(row)):
        for j in row[i + 1:]:  # go through all elements after i
            if row[i] > j != 0:  # j is bigger than i and j isn't the empty element
                noi = noi + 1
    return noi


def row_of_blank_from_bottom(state):
    rob = 0
    for i in state:
        if 0 in i:
            return len(state) - rob
        rob = rob + 1


""""-----------------------------------------------------------------------------------------------------"""


""" ------- Just for testing --------"""

""" Random state """
# dim = int(input("Enter dimension -> "))
# initial = random_state(dim)
# print("The initial random state is:")
# for row in initial:
#     print(row)



""" Choosing an algorithm """
algorithm = input("Choose an algorithm [1-Breadth first, 2-Depth first, 3-Uniform cost, 4-Depth limited, 5-Iterative deepening, \
6-A*, 7-Greedy] -> ")

gt = GoalTree(initial)
info = gt.solve(algorithm)
stop_time = datetime.now()
sol_state, sol, g, processed_nodes, max_stored_nodes, find_sol, start_time = info

print("\n---------- Output Information ----------")
print("Time taken:", stop_time-start_time)
print(f'G-value (level solution found in goal tree): {g}')
print(f'Processed nodes: {processed_nodes}')
print(f'Max stored nodes: {max_stored_nodes}')
print(f'Do we find solution: {find_sol}')
print("Solution:\n" + str(sol))
print("Solution state:")
for row in sol_state:
    print(row)
