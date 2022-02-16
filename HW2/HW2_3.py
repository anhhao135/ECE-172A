'''
ECE 172A, Homework 2 Robot Kinematics
Author: regreer@ucsd.edu
For use by UCSD ECE 172A students only.
'''

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

def forwardKinematics(theta0, theta1, theta2, l0, l1, l2):

    T_2E = np.array([l2, 0]) #translation to get EE frame to J2 frame
    T_12 = np.array([l1, 0]) #translation to get J2 frame to J1 frame
    T_01 = np.array([l0, 0]) #translation to get J1 frame to J0 frame

    R_12 = createRotationMatrix(-theta2) #clockwise rotation about J2; J2 frame to J1 frame
    R_01 = createRotationMatrix(theta1) #anticlockwise rotation about J1; J1 frame to J0 frame
    R_G0 = createRotationMatrix(theta0) #anticlockwise rotation about J0; J0 frame to global frame

    #get position of E in global frame

    P_E_1 = np.matmul(R_12,T_2E) + T_12 #position in J1 frame
    P_E_0 = np.matmul(R_01,P_E_1) + T_01 #position in J0 frame
    P_E_G = np.matmul(R_G0,P_E_0) #position in global frame

    #get position of J2 in global frame

    P_J2_0 = np.matmul(R_01,T_12) + T_01 #position in J0 frame 
    P_J2_G = np.matmul(R_G0,P_J2_0) #position in global frame

    #get position of J1 in global frame

    P_J1_G = np.matmul(R_G0,T_01) #position in global frame

    return [P_J1_G[0], P_J1_G[1], P_J2_G[0], P_J2_G[1], P_E_G[0], P_E_G[1]] #return global positions of joints and EE



def createRotationMatrix(theta): #rotation clockwise about origin, in radians
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])



def inverseKinematics(l0,l1,l2,x_e_target,y_e_target,thetas):

    jointPositions = forwardKinematics(thetas[0],thetas[1],thetas[2],l0,l1,l2)

    closeEnoughDistance = 0.01

    targetEEPosition = np.array([x_e_target,y_e_target])
    currentEEPosition = np.array([jointPositions[4],jointPositions[5]])

    stepSize = 0.1
    
    e_x = []
    e_y = [] #keep track of end effector position

    while np.linalg.norm(targetEEPosition - currentEEPosition) > closeEnoughDistance: #if distance is smaller than "close enough" metric, stop

        e_x.append(jointPositions[4])
        e_y.append(jointPositions[5]) #record down starting EE pos as well as the subsequent iterations

        EEPositionIncrement = (targetEEPosition - currentEEPosition) * stepSize  #get delta e in the direction of target

        jacobian = calulateJacobian(thetas[0],thetas[1],thetas[2],l0,l1,l2)
        pinvJacobian = np.linalg.pinv(jacobian)

        deltaJointAngles = np.matmul(pinvJacobian,EEPositionIncrement) #get delta thetas to move EE in that direction

        thetas = thetas + deltaJointAngles #change thetas

        jointPositions = forwardKinematics(thetas[0],thetas[1],thetas[2],l0,l1,l2)
        currentEEPosition = np.array([jointPositions[4],jointPositions[5]]) #calculate new position of EE


    jointPositions = forwardKinematics(thetas[0],thetas[1],thetas[2],l0,l1,l2) #get the final state of the robot with the calculated thetas

    drawRobot(jointPositions[0], jointPositions[1], jointPositions[2], jointPositions[3], jointPositions[4], jointPositions[5])
    plt.scatter(e_x, e_y)
    plt.show()

    return thetas

def calulateJacobian(theta0, theta1, theta2, l0, l1, l2): #analytically calculated by algebraically getting EE position in terms of thetas and l's. Then doing partial derivates.

    dx_theta0 = - l0 * np.sin(theta0) - l1 * np.sin(theta0 + theta1) - l2 * np.sin(theta0 + theta1 + theta2)
    dx_theta1 = - l1 * np.sin(theta0 + theta1) - l2 * np.sin(theta0 + theta1 + theta2)
    dx_theta2 = - l2 * np.sin(theta0 + theta1 + theta2)

    dy_theta0 = l0 * np.cos(theta0) + l1 * np.cos(theta0 + theta1) + l2 * np.cos(theta0 + theta1 + theta2)
    dy_theta1 = l1 * np.cos(theta0 + theta1) + l2 * np.cos(theta0 + theta1 + theta2)
    dy_theta2 = l2 * np.cos(theta0 + theta1 + theta2)

    jacobian = np.array([[dx_theta0,dx_theta1,dx_theta2],[dy_theta0,dy_theta1,dy_theta2]])

    return jacobian

def drawRobot(x_1,y_1,x_2,y_2,x_e,y_e):

    x_0, y_0 = 0, 0
    ax = plt.axes()
    ax.axis('equal') #make axes 1:1 scale for true visualization
    plt.plot([x_0, x_1, x_2, x_e], [y_0, y_1, y_2, y_e], lw=4.5)
    plt.scatter([x_0, x_1, x_2, x_e], [y_0, y_1, y_2, y_e], color='r')
    plt.show()





jointPositions = forwardKinematics(pi/3, pi/12, -pi/6, 3, 5, 7)
print(jointPositions)
drawRobot(jointPositions[0], jointPositions[1], jointPositions[2], jointPositions[3], jointPositions[4], jointPositions[5])

jointPositions = forwardKinematics(pi/4, pi/4, pi/4, 3, 5, 2)
print(jointPositions)
drawRobot(jointPositions[0], jointPositions[1], jointPositions[2], jointPositions[3], jointPositions[4], jointPositions[5])


print(inverseKinematics(10,10,10,6,12,np.array([pi/6,0,0])))
print(inverseKinematics(10,10,10,6,12,np.array([pi/3,0,0])))