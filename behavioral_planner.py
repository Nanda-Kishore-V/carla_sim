import numpy as np
import time

# FSM States
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2

# Threshold under which the car is assumed to have stopped
SPEED_THRESHOLD = 0.02  # km/hr

# Stop timesteps
STOP_TIME = 3.0  # s


class BehavioralPlanner:
    def __init__(self, lookahead_distance, stopsign_lines, lead_vehicle_lookahead):
        self.lookahead_distance = lookahead_distance
        self.stopsign_lines = stopsign_lines
        self.lead_vehicle_lookahead = lead_vehicle_lookahead
        self.fsm_state = FOLLOW_LANE
        self.follow_lead_vehicle = False
        self.goal_index = 0
        self.goal_waypoint = [0.0, 0.0, 0.0]
        self.stop_time_start = 0

    def set_lookahead_distance(self, lookahead_distance):
        self.lookahead_distance = lookahead_distance

    def detect_stopsign(self, waypoints, closest_index, goal_index):
        """
        Line segment intersection method from:
        http://paulbourke.net/geometry/pointlineplane/
        """
        for i in range(closest_index, goal_index):
            for stop_line in self.stopsign_lines:
                P1 = np.array(waypoints[i][0:2])
                P2 = np.array(waypoints[i+1][0:2])
                P3 = np.array(stop_line[0:2])
                P4 = np.array(stop_line[2:4])

                denominator = (P4[1] - P3[1])*(P2[0] - P1[0]) - \
                    (P4[0] - P3[0])*(P2[1] - P1[0])
                numerator_a = (P4[0] - P3[0])*(P1[1] - P3[1]) - \
                    (P4[1] - P3[1])*(P1[0] - P3[0])
                numerator_b = (P2[0] - P1[0])*(P1[1] - P3[1]) - \
                    (P2[1] - P1[1])*(P1[0] - P3[0])
                if denominator == 0:
                    if numerator_a == 0 and numerator_b == 0:
                        # Line segments are coincidental
                        goal_index = i
                        return goal_index, True
                    # Line segments are parallel
                    continue

                u_a = numerator_a / denominator
                u_b = numerator_b / denominator
                if 0 <= u_a <= 1 and 0 <= u_b <= 1:
                    # Intersection
                    goal_index = i
                    return goal_index, True
        return goal_index, False

    def get_lookahead_index(self, waypoints, ego_state, closest_dist, closest_index):
        distance_so_far = closest_dist
        lookahead_index = closest_index

        if distance_so_far >= self.lookahead_distance:
            return lookahead_index

        if closest_index == len(waypoints)-1:
            return closest_index

        for i in range(closest_index+1, len(waypoints)):
            distance_so_far += np.sqrt((waypoints[i][0] - waypoints[i-1][0])**2 +
                                       (waypoints[i][1] - waypoints[i-1][1])**2)
            lookahead_index = i

            if distance_so_far >= self.lookahead_distance:
                break

        return lookahead_index

    def state_transition(self, waypoints, ego_state, closed_loop_speed, lead_vehicle_position):
        self.detect_lead_vehicle(ego_state, lead_vehicle_position)

        if self.fsm_state == FOLLOW_LANE:
            closest_dist, closest_index = find_closest_waypoint_index(waypoints,
                                                                      ego_state)

            provisional_goal_index = self.get_lookahead_index(waypoints,
                                                              ego_state,
                                                              closest_dist,
                                                              closest_index)

            new_goal_index, stopsign_found = self.detect_stopsign(waypoints,
                                                                     closest_index,
                                                                     provisional_goal_index)
            self.goal_index = new_goal_index if stopsign_found else provisional_goal_index
            self.goal_waypoint = waypoints[self.goal_index]

            if stopsign_found:
                # State transition
                self.goal_waypoint[2] = 0
                self.fsm_state = DECELERATE_TO_STOP

        elif self.fsm_state == DECELERATE_TO_STOP:
            if closed_loop_speed < SPEED_THRESHOLD:
                # State transition
                self.fsm_state = STAY_STOPPED
                self.stop_time_start = time.time()

        elif self.fsm_state == STAY_STOPPED:
            if time.time() >= self.stop_time_start + STOP_TIME:
                closest_dist, closest_index = find_closest_waypoint_index(waypoints,
                                                                          ego_state)

                goal_index = self.get_lookahead_index(waypoints, ego_state, closest_dist,
                                                      closest_index)

                _, stopsign_found = self.detect_stopsign(
                    waypoints, closest_index, goal_index)
                self.goal_index = goal_index
                self.goal_waypoint = waypoints[self.goal_index]

                if not stopsign_found:
                    # Reset stop time after we cross the stop sign
                    # State transition
                    self.stop_time_start = 0
                    self.fsm_state = FOLLOW_LANE
            else:
                self.stop_time = time.time()

        else:
            raise ValueError("This FSM state does not exist")

        return

    def detect_lead_vehicle(self, ego_state, lead_vehicle_position):
        ego_to_lead_vector = np.array([lead_vehicle_position[0] - ego_state[0],
                                       lead_vehicle_position[1] - ego_state[1]])
        lead_vehicle_distance = np.linalg.norm(ego_to_lead_vector)

        current_heading = np.array([np.cos(ego_state[2]),
                                    np.sin(ego_state[2])])
        ego_to_lead_unit_vector = ego_to_lead_vector / lead_vehicle_distance

        relative_heading = np.dot(current_heading, ego_to_lead_unit_vector)
        if not self.follow_lead_vehicle:
            if lead_vehicle_distance > self.lead_vehicle_lookahead:
                return

            if relative_heading < np.sqrt(1/2):
                return

            self.follow_lead_vehicle = True
        else:
            # Add buffer to prevent oscillations
            if lead_vehicle_distance < self.lead_vehicle_lookahead + 15:
                return

            if relative_heading > np.sqrt(1/2):
                return

            self.follow_lead_vehicle = False


def find_closest_waypoint_index(waypoints, location):
    wps = np.array(waypoints)[:, :2]
    distances = np.sum((wps - np.array(location[:2]))**2, axis=1)
    closest_index = np.argmin(distances)
    return distances[closest_index], closest_index
