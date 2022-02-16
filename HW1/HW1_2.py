'''
ECE 172A, Homework 1 Robot Traversal
Author: regreer@ucsd.edu
For use by UCSD ECE 172A students only.
'''

import numpy as np
import matplotlib.pyplot as plt

# Set room size
vSize = 9 
hSize = 9

# Initialize position and object locations, then display.
loc = np.array([[0, 6]])


def haveIBeenHereBefore(loc, nextStep):
    ''' 
    This function can be used to determine if the robot has previously been 
    to the location specified in nextStep.
    loc is the set of previous locations traversed by the robot
    nextStep is the new location for which the test is to be performed
    out is a boolean value, True if nextStep has been previosuly
    visited
    '''
    return nextStep.tolist() in loc.tolist()

def displayRoom(loc,obj,vSize,hSize):
    # Create empty room
    room = np.zeros((vSize,hSize))
    
    # Represent objects with gray
    for ob in obj:
        room[ob[0],ob[1]] = 127
    
    # Represent past locations with light gray
    for lo in loc[:-1]:
        room[lo[0], lo[1]] = 191
    
    print(loc[-1])
    # Represent current location with white
    room[loc[-1][0], loc[-1][1]] = 255
    

    plt.imshow(room, cmap='gray')
    plt.title('Press \'q\' to continue. Ctrl+C (or Cmd+C) to stop simulation.')
    plt.show()

def detectObject(loc, obj, dir):
    # Check for object in specified direction
    if dir == 'N':
        dirLoc = loc[-1] + np.array([-1, 0])
    elif dir == 'E':
        dirLoc = loc[-1] + np.array([0, 1])
    elif dir == 'S':
        dirLoc = loc[-1] + np.array([1, 0])
    else:
        dirLoc = loc[-1] + np.array([0, -1])
    objectDetected = dirLoc.tolist() in obj.tolist()
    return objectDetected


# Add additional obstacles to this array per instructions:
obj = np.array([[0, 3], [2, 7], [3, 8], [5, 5], [2, 0], [1, 2], [7, 1], [3, 6], [4, 4], [4, 6]]) 

displayRoom(loc, obj, vSize, hSize)

while(True):
    # Make the robot move a certain direction
    nextStep = loc[-1] + np.array([1, 0])
    
    # If there is an object to the South, move a different direction
    # START


    if detectObject(loc, obj, 'S'):
        if not detectObject(loc, obj, 'W'):
            nextStep = loc[-1] + np.array([0, -1])
        elif not detectObject(loc, obj, 'E'):
            nextStep = loc[-1] + np.array([0, 1])
        elif not detectObject(loc, obj, 'N'):
            nextStep = loc[-1] + np.array([-1, 0])

    if haveIBeenHereBefore(loc, nextStep):
        if not detectObject(loc, obj, 'W'):
            nextStep = loc[-1] + np.array([0, -1])
        elif not detectObject(loc, obj, 'N'):
            nextStep = loc[-1] + np.array([-1, 0])
        
        
    # STOP
    
    # Update location if no object is in the way and within bounds
    if (nextStep.tolist() not in obj.tolist()) and nextStep[0] >= 0 and nextStep[0] <= (vSize - 1) and nextStep[1] >= 0:
        loc = np.vstack([loc, nextStep])
    
    # Show new position
    displayRoom(loc, obj, vSize, hSize)
    
    # Check if the South side of the image has been reached
    if loc[-1][0] == vSize-1:
        print("Success!")
        break


# loc keeps track of the current and all the past locations of the robot. This is a list of 2D coordinates, with the last added coordinate being the current position.
# Adding [1, 0] will make the robot move 1 step south only; this is because on the image display, the first coordinate represent the row index.
# Adding an object to [3, 6] causes the robot to be stuck in an infinite loop. This is due to the break condition of the while loop: since there is an object in the way of the robot as it moves south, it can never add the location [3, 6] to loc and thus advance past the obstacle. The break condition requires the robot to reach the south side i.e. the last coordinate in loc to have a row index of vSize - 1, in this case 8. However, this can never happen, so the while loop carries on infinitely.
# The robot is not intelligent because it does not act upon the obstacle to adapt and find a new solution to reach its south side goal. It would need to somehow acquire the new information regarding the obstacle, and change its state so as the simulation carries on, it is not the same result (staying in one spot) infinitely. Additionally, if the goal cannot be achieved, it must make a decision to break the infinite loop.

# By adding the detectObject() function, the robot has the ability to sense the presence of an obstacle infront of it; in this case, it detects an obstacle in the southern direction, and as a result adds -1 to its column coordinate which is equivalent to moving west 1 step, effectively getting out of the obstacle's way. The next while iteration will continue as usual with the robot moving south, unblocked (or blocked in which case it can be handled similarly). This robot is more intelligent because it uses information from its environment - the presence of obstacles - to change states internally and not repeat moving south infinitely, thus increasing chances of solving the problem.