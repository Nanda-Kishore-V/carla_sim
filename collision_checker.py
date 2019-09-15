#!/usr/bin/env python3
import numpy as np
import scipy.spatial

class CollisionChecker:
    def __init__(self, circle_offsets, circle_radii,
                 obstacle_proximity_weight=10.0):
        self.circle_offsets = np.array(circle_offsets)
        self.circle_radii = circle_radii
        self.obstacle_proximity_weight = obstacle_proximity_weight

    def collision_check(self, paths, obstacles):
        # Circle based collision detection
        collision_free_mask = [False] * len(paths)
        for i in range(len(paths)):
            path = paths[i]
            collision_free = True
            for j in range(len(path[0])):
                circle_locations = np.zeros((len(self.circle_offsets), 2))
                circle_locations[:, 0] = path[0][j] + \
                    self.circle_offsets*np.cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + \
                    self.circle_offsets*np.sin(path[2][j])

                for k in range(len(obstacles)):
                    distances = scipy.spatial.distance.cdist(obstacles[k],
                                                             circle_locations)
                    distances = distances - self.circle_radii
                    collision_free = collision_free and not np.any(distances < 0)

                    if not collision_free:
                       break

            if not collision_free:
               break

            collision_free_mask[i] = collision_free

        return collision_free_mask

    def select_best_path_index(self, paths, collision_free_mask, goal_state):
        best_index = None
        best_score = np.inf

        for i in range(len(paths)):
            path = paths[i]
            if collision_free_mask[i]:
                # Far from goal?
                score = np.sqrt((path[0][-1] - goal_state[0])**2 +
                                (path[1][-1] - goal_state[1])**2)

                for j in range(len(paths)):
                    if i == j:
                        continue
                    else:
                        if not collision_free_mask[j]:
                            # Close to obstacle?
                            score += self.obstacle_proximity_weight * \
                                np.sqrt((path[0][-1] - paths[j][0][-1])**2 +
                                        (path[1][-1] - paths[j][1][-1])**2)

            else:
                score = np.inf

            if score < best_score:
                best_score = score
                best_index = i

        return best_index
