#/usr/env/bin python3
import numpy as np

class VelocityPlanner:
    def __init__(self, time_gap, a_max, coasting_speed, stop_line_buffer):
        self.time_gap = time_gap
        self.a_max = a_max
        self.coasting_speed = coasting_speed
        self.stop_line_buffer = stop_line_buffer
        self.prev_trajectory = [[0, 0, 0]]

    def get_open_loop_speed(self, timestep):
        if len(self.prev_trajectory) == 1:
            return self.prev_trajectory[0][2]

        if timestep < 1e-4:
            return self.prev_trajectory[0][2]

        for i in range(len(self.prev_trajectory)-1):
            distance = np.linalg.norm(np.subtract(self.prev_trajectory[i+1][0:2],
                                                  self.prev_trajectory[i][0:2]))

            velocity = self.prev_trajectory[i][2]
            dt = distance / velocity

            if dt > timestep:
                v1 = self.prev_trajectory[i][2]
                v2 = self.prev_trajectory[i+1][2]
                dv = v2 - v1
                ratio = timestep / dt
                return v1 + ratio * dv
            else:
                timestep -= dt

        return self.prev_trajectory[-1][2]

    def compute_velocity_profile(self, path, desired_speed, ego_state,
                                 closed_loop_speed, decelerate_to_stop,
                                 lead_vehicle_state, follow_lead_vehicle):
        profile = []
        start_speed = ego_state[3]

        if decelerate_to_stop:
            profile = self.deceleration_profile(path, start_speed)

        elif follow_lead_vehicle:
            profile = self.follow_profile(path, start_speed, desired_speed, lead_vehicle_state)

        else:
            profile = self.nominal_profile(path, start_speed, desired_speed)

        if len(profile) > 1:
            profile_start = [(profile[1][0] - profile[0][0]) * 0.1 + profile[0][0],
                             (profile[1][1] - profile[0][1]) * 0.1 + profile[0][1],
                             (profile[1][2] - profile[0][2]) * 0.1 + profile[0][2]]
            del profile[0]
            profile.insert(0, profile_start)

        self.prev_trajectory = profile

        return profile

    def deceleration_profile(self, path, start_speed):
        profile = []

        distance_to_coasting = calc_distance(start_speed, self.coasting_speed, -self.a_max)
        coasting_to_stop_distance = calc_distance(self.coasting_speed, 0, -self.a_max)

        total_path_length = 0
        for i in range(len(path[0])-1):
            total_path_length += np.sqrt((path[0][i+1] - path[0][i])**2 +
                                         (path[1][i+1] - path[1][i])**2)

        stop_index = len(path[0]) - 1
        distance = 0
        while stop_index >= 1 and distance < self.stop_line_buffer:
            distance += np.sqrt((path[0][stop_index] - path[0][stop_index-1])**2 +
                                (path[1][stop_index] - path[1][stop_index-1])**2)
            stop_index -= 1


        if distance_to_coasting + coasting_to_stop_distance + \
                self.stop_line_buffer > total_path_length:
            speed_profile = []
            vf = 0

            for i in reversed(range(stop_index, len(path[0]))):
                speed_profile.insert(0, 0)

            for i in reversed(range(stop_index)):
                distance = np.sqrt((path[0][i+1] - path[0][i])**2 +
                               (path[1][i+1] - path[1][i])**2)

                vi = calc_final_speed(vf, -self.a_max, distance)
                if vi > start_speed:
                    vi = start_speed

                speed_profile.insert(0, vi)
                vf = vi

            for i in range(len(speed_profile)):
                profile.append([path[0][i], path[1][i], speed_profile[i]])

        else:
            brake_index = stop_index
            distance = 0

            while brake_index >= 1 and distance < coasting_to_stop_distance:
                distance += np.sqrt((path[0][brake_index] - path[0][brake_index-1])**2 +
                                    (path[1][brake_index] - path[1][brake_index-1])**2)
                brake_index -= 1

            coasting_index = 0
            distance = 0
            while coasting_index < brake_index and distance < distance_to_coasting:
                distance += np.sqrt((path[0][coasting_index] - path[0][coasting_index-1])**2 +
                                    (path[1][coasting_index] - path[1][coasting_index-1])**2)
                coasting_index += 1

            vi = start_speed
            for i in range(coasting_index):
                distance = np.sqrt((path[0][i+1] - path[0][i])**2 +
                                    (path[1][i+1] - path[1][i])**2)
                vf = calc_final_speed(vi, -self.a_max, distance)

                if vf < self.coasting_speed:
                    vf = self.coasting_speed

                profile.append([path[0][i], path[1][i], vi])
                vi = vf

            for i in range(coasting_index, brake_index):
                profile.append([path[0][i], path[1][i], vi])

            for i in range(brake_index, stop_index):
                distance = np.sqrt((path[0][i+1] - path[0][i])**2 +
                                    (path[1][i+1] - path[1][i])**2)
                vf = calc_final_speed(vi, -self.a_max, distance)
                profile.append([path[0][i], path[1][i], vi])
                vi = vf

            for i in range(stop_index, len(path[0])):
                profile.append([path[0][i], path[1][i], 0])

        return profile

    def follow_profile(self, path, start_speed, desired_speed, lead_vehicle_state):
        profile = []

        closest_index_to_lead = len(path[0])-1
        closest_distance = np.inf

        for i in range(len(path[0])-1):
            distance = np.sqrt((path[0][i] - lead_vehicle_state[0])**2 +
                                (path[1][i] - lead_vehicle_state[1])**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_index_to_lead = i

        desired_speed = min(lead_vehicle_state[2], desired_speed)

        ramp_end_index = closest_index_to_lead
        distance = closest_distance
        distance_gap = desired_speed * self.time_gap

        while ramp_end_index >= 1 and distance > distance_gap:
            distance += np.sqrt((path[0][ramp_end_index] - path[0][ramp_end_index-1])**2 +
                                (path[1][ramp_end_index] - path[1][ramp_end_index-1])**2)
            ramp_end_index -= 1

        if desired_speed < start_speed:
            accel_distance = calc_distance(start_speed, desired_speed, -self.a_max)
        else:
            accel_distance = calc_distance(start_speed, desired_speed, self.a_max)

        vi = start_speed
        for i in range(ramp_end_index + 1):
            distance = np.sqrt((path[0][i+1] - path[0][i])**2 +
                               (path[1][i+1] - path[1][i])**2)
            if desired_speed < start_speed:
                vf = calc_final_speed(vi, -self.a_max, distance)
            else:
                vf = calc_final_speed(vi, self.a_max, distance)

            profile.append([path[0][i], path[1][i], vi])
            vi = vf

        for i in range(ramp_end_index+1, len(path[0])):
            profile.append([path[0][i], path[1][i], desired_speed])

        return profile


    def nominal_profile(self, path, start_speed, desired_speed):
        profile = []

        if desired_speed < start_speed:
            accel_distance = calc_distance(start_speed, desired_speed, -self.a_max)
        else:
            accel_distance = calc_distance(start_speed, desired_speed, self.a_max)

        ramp_end_index = 0
        distance = 0
        while distance < accel_distance and ramp_end_index < len(path[0])-1:
            distance += np.sqrt((path[0][ramp_end_index+1] - path[0][ramp_end_index])**2 +
                                (path[1][ramp_end_index+1] - path[1][ramp_end_index])**2)
            ramp_end_index += 1

        vi = start_speed
        for i in range(ramp_end_index):
            distance = np.sqrt((path[0][i+1] - path[0][i])**2 +
                               (path[1][i+1] - path[1][i])**2)
            if desired_speed < start_speed:
                vf = calc_final_speed(vi, -self.a_max, distance)
                if vf < desired_speed:
                    vf = desired_speed
            else:
                vf = calc_final_speed(vi, self.a_max, distance)
                if vf > desired_speed:
                    vf = desired_speed

            profile.append([path[0][i], path[1][i], vi])
            vi = vf

        for i in range(ramp_end_index+1, len(path[0])):
            profile.append([path[0][i], path[1][i], desired_speed])

        return profile


def calc_distance(vi, vf, a):
    d = (vf**2 - vi**2 ) / (2*a)
    return d

def calc_final_speed(vi, a, d):
    vf_squared = vi**2 + 2*a*d
    vf = np.sqrt(vf_squared) if vf_squared >0 else 0
    return vf
