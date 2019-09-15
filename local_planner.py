#/usr/bin/env python3
import numpy as np
import copy
from path_optimizer import PathOptimizer
from velocity_planner import VelocityPlanner
from collision_checker import CollisionChecker

class LocalPlanner:
    def __init__(self, num_paths, path_offset, circle_offsets, circle_radii,
                 path_select_weight, time_gap, a_max, coasting_speed,
                 stop_line_buffer):
        self.num_paths = num_paths
        self.path_offset = path_offset
        self.path_optimizer = PathOptimizer()
        self.collision_checker = CollisionChecker(circle_offsets, circle_radii,
                                                  path_select_weight)
        self.velocity_planner = VelocityPlanner(time_gap, a_max, coasting_speed,
                                                stop_line_buffer)

    def get_goal_state_set(self, goal_index, goal_state, waypoints, ego_state):
        if goal_index == len(waypoints) - 1:
            dx = waypoints[goal_index][0] - waypoints[goal_index-1][0]
            dy = waypoints[goal_index][1] - waypoints[goal_index-1][1]
        else:
            dx = waypoints[goal_index+1][0] - waypoints[goal_index][0]
            dy = waypoints[goal_index+1][1] - waypoints[goal_index][1]
        heading = np.arctan2(dy, dx)

        goal_state_copy = copy.copy(goal_state)

        goal_state_copy[0] -= ego_state[0]
        goal_state_copy[1] -= ego_state[1]

        x = goal_state_copy[0]
        y = goal_state_copy[1]
        theta = -ego_state[2]

        goal_x = np.cos(theta)*x - np.sin(theta)*y
        goal_y = np.sin(theta)*x + np.cos(theta)*y
        goal_yaw = heading + theta
        goal_v = goal_state[2]

        if goal_yaw > np.pi:
            goal_yaw -= 2*np.pi
        elif goal_yaw < -np.pi:
            goal_yaw += 2*np.pi

        goal_state_set = []
        for i in range(self.num_paths):
            offset = (i - self.num_paths // 2) * self.path_offset

            x_offset = np.cos(np.pi/2 - goal_yaw) * offset
            y_offset = np.sin(np.pi/2 - goal_yaw) * offset

            goal_state_set.append([goal_x + x_offset,
                                   goal_y + y_offset,
                                   goal_yaw,
                                   goal_v])
        return goal_state_set

    def plan_paths(self, goal_state_set):
        paths = []
        is_desired_path = []
        for goal_state in goal_state_set:
            path = self.path_optimizer.optimize_spiral(goal_state[0],
                                                       goal_state[1],
                                                       goal_state[2])
            if np.sqrt((path[0][-1] - goal_state[0])**2 +
                       (path[1][-1] - goal_state[1])**2 +
                       (path[2][-1] - goal_state[2])**2) > 0.1:
                is_desired_path.append(False)
            else:
                paths.append(path)
                is_desired_path.append(True)

        return paths, is_desired_path

    @staticmethod
    def transform_paths(paths, ego_state):
        transformed_paths = []
        for path in paths:
            x_transformed = []
            y_transformed = []
            yaw_transformed = []

            for i in range(len(path[0])):
                x_transformed.append(ego_state[0] +
                                     path[0][i]*np.cos(ego_state[2]) -
                                     path[1][i]*np.sin(ego_state[2]))
                y_transformed.append(ego_state[1] +
                                     path[0][i]*np.sin(ego_state[2]) +
                                     path[1][i]*np.cos(ego_state[2]))
                yaw_transformed.append(path[2][i] + ego_state[2])

            transformed_paths.append([x_transformed, y_transformed, yaw_transformed])
        return transformed_paths
