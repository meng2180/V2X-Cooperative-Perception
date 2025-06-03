#!/usr/bin/env python

"""
Custom evaluation criteria that inherits from SRUNNER's CollisionTest,
and overrides _count_collisions to enable real-time collision printout.
"""

import py_trees
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorSpeedAboveThresholdTest)

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType
from opencood.utils import eval_utils

import json
import math
import time
from collections import Counter
import os

class CustomCollisionTest(CollisionTest):
    """
    Custom collision test with real-time print of collision events.
    """
    def __init__(self, actor, other_actor=None, other_actor_type=None,
                 optional=False, name="CollisionTest", terminate_on_failure=False, mis_error = None):
        super(CustomCollisionTest, self).__init__(actor=actor, other_actor=other_actor, other_actor_type=other_actor_type,
                 optional=optional, name=name, terminate_on_failure=terminate_on_failure)
        self.mis_error = mis_error
        self.last_speed = []
        self._time_last_valid_state = None
    
    def update(self):
        """
        Check collision count
        """
        new_status = py_trees.common.Status.RUNNING

        linear_speed = CarlaDataProvider.get_velocity(self.actor)
        if linear_speed is not None:
            if linear_speed < 2 and self._time_last_valid_state:
                if len(self.last_speed) > 100:
                    self.last_speed.pop(0)
                self.last_speed.append(linear_speed)
            else:
                self.last_speed = []
                self._time_last_valid_state = GameTime.get_time()

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        actor_location = CarlaDataProvider.get_location(self.actor)
        new_registered_collisions = []

        # Loops through all the previous registered collisions
        for collision_location in self.registered_collisions:

            # Get the distance to the collision point
            distance_vector = actor_location - collision_location
            distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

            # If far away from a previous collision, forget it
            if distance <= self.MAX_AREA_OF_COLLISION:
                new_registered_collisions.append(collision_location)

        self.registered_collisions = new_registered_collisions

        if self.last_id and GameTime.get_time() - self.collision_time > self.MAX_ID_TIME:
            self.last_id = None

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    @staticmethod
    def _count_collisions(weak_self, event):  # override original method
        """
        Callback to update collision count
        """
        self = weak_self()
        if not self:
            return

        actor_location = CarlaDataProvider.get_location(self.actor)

        # Ignore the current one if it is the same id as before
        if self.last_id == event.other_actor.id:
            return

        # Filter to only a specific actor
        if self.other_actor and self.other_actor.id != event.other_actor.id:
            return

        # Filter to only a specific type
        if self.other_actor_type:
            if self.other_actor_type == "miscellaneous":
                if "traffic" not in event.other_actor.type_id and "static" not in event.other_actor.type_id:
                    return
            else:
                if self.other_actor_type not in event.other_actor.type_id:
                    return

        # Ignore it if its too close to a previous collision (avoid micro collisions)
        for collision_location in self.registered_collisions:

            distance_vector = actor_location - collision_location
            distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))
            
            if distance <= self.MIN_AREA_OF_COLLISION:
                return

        # Classify type of collision
        if ('static' in event.other_actor.type_id or 'traffic' in event.other_actor.type_id) \
                and 'sidewalk' not in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_STATIC
        elif 'vehicle' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_VEHICLE
        elif 'walker' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_PEDESTRIAN
        else:
            return

        # Create and store event
        collision_event = TrafficEvent(event_type=actor_type)
        collision_event.set_dict({
            'type': event.other_actor.type_id,
            'id': event.other_actor.id,
            'x': actor_location.x,
            'y': actor_location.y,
            'z': actor_location.z})
        collision_event.set_message(
            "Agent collided with type={} id={} at (x={}, y={}, z={}), time={}".format(
                event.other_actor.type_id,
                event.other_actor.id,
                round(actor_location.x, 3),
                round(actor_location.y, 3),
                round(actor_location.z, 3),
                GameTime.get_time()))

        
        average_speed = 2
        if self.last_speed:
            average_speed = sum(self.last_speed) / len(self.last_speed)
        if average_speed < 0.8:
            collision_step = len(self.last_speed) / 4
            self.mis_error[4] = int(collision_step)
            print("Collision invalid frame: {}".format(int(collision_step)))
        else:
            self.last_speed = []
        self.mis_error[0] = True
       
        print(collision_event.get_message())
        self.test_status = "FAILURE"
        self.actual_value += 1
        self.collision_time = GameTime.get_time()

        self.registered_collisions.append(actor_location)
        self.list_traffic_events.append(collision_event)

        if event.other_actor.id != 0:
            self.last_id = event.other_actor.id

class CustomInRouteTest(InRouteTest):

    def __init__(self, actor, route, offroad_min=-1, offroad_max=30, name="InRouteTest", terminate_on_failure=False, mis_error = None):
        super(CustomInRouteTest, self).__init__(actor, route, offroad_min=offroad_min, offroad_max=offroad_max, 
                                                name=name, terminate_on_failure=terminate_on_failure)
        self.mis_error = mis_error

    def update(self):
        """
        Check if the actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)
        if location is None:
            return new_status

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        elif self.test_status == "RUNNING" or self.test_status == "INIT":

            off_route = True

            shortest_distance = float('inf')
            closest_index = -1

            # Get the closest distance
            for index in range(self._current_index,
                               min(self._current_index + self.WINDOWS_SIZE + 1, self._route_length)):
                ref_waypoint = self._waypoints[index]
                distance = math.sqrt(((location.x - ref_waypoint.x) ** 2) + ((location.y - ref_waypoint.y) ** 2))
                if distance <= shortest_distance:
                    closest_index = index
                    shortest_distance = distance

            if closest_index == -1 or shortest_distance == float('inf'):
                return new_status

            # Check if the actor is out of route
            if shortest_distance < self._offroad_max:
                off_route = False
                self._in_safe_route = bool(shortest_distance < self._offroad_min)

            # If actor advanced a step, record the distance
            if self._current_index != closest_index:

                new_dist = self._accum_meters[closest_index] - self._accum_meters[self._current_index]

                # If too far from the route, add it and check if its value
                if not self._in_safe_route:
                    self._out_route_distance += new_dist
                    out_route_percentage = 100 * self._out_route_distance / self._accum_meters[-1]
                    if out_route_percentage > self.MAX_ROUTE_PERCENTAGE:
                        off_route = True

                self._current_index = closest_index

            if off_route:
                # Blackboard variable
                blackv = py_trees.blackboard.Blackboard()
                _ = blackv.set("InRoute", False)

                route_deviation_event = TrafficEvent(event_type=TrafficEventType.ROUTE_DEVIATION)
                route_deviation_event.set_message(
                    "Agent deviated from the route at (x={}, y={}, z={})".format(
                        round(location.x, 3),
                        round(location.y, 3),
                        round(location.z, 3)))
                route_deviation_event.set_dict({
                    'x': location.x,
                    'y': location.y,
                    'z': location.z})

                self.mis_error[0] = True

                self.list_traffic_events.append(route_deviation_event)

                self.test_status = "FAILURE"
                self.actual_value += 1
                new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status
    
class CustomRouteCompletionTest(RouteCompletionTest):

    def __init__(self, actor, route, name="RouteCompletionTest", terminate_on_failure=False, mis_error=None):
        super(CustomRouteCompletionTest, self).__init__(actor=actor, route=route, name=name, terminate_on_failure=terminate_on_failure)
        self.mis_error = mis_error

    def terminate(self, new_status):
        """
        Set test status to failure if not successful and terminate
        """
        self.actual_value = round(self._percentage_route_completed, 2)

        if self.test_status == "INIT":
            self.test_status = "FAILURE"

            self.mis_error[0] = True
            
        super(CustomRouteCompletionTest, self).terminate(new_status)


class CustomOutsideRouteLanesTest(OutsideRouteLanesTest):
    ALLOWED_OUT_DISTANCE = 0.7          # At least 0.5, due to the mini-shoulder between lanes and sidewalks

    def __init__(self, actor, route, optional=False, name="OutsideRouteLanesTest", mis_error=None):
        super(CustomOutsideRouteLanesTest, self).__init__(actor=actor, route=route, optional=optional, name=name)
        self.mis_error = mis_error
        self._terminate_on_failure = True
    def update(self):
        """
        Transforms the actor location and its four corners to waypoints. Depending on its types,
        the actor will be considered to be at driving lanes, sidewalk or offroad.

        returns:
            py_trees.common.Status.FAILURE: when the actor has left driving and terminate_on_failure is active
            py_trees.common.Status.RUNNING: the rest of the time
        """
        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        # Some of the vehicle parameters
        location = CarlaDataProvider.get_location(self._actor)
        if location is None:
            return new_status

        # Check if outside route lanes
        self._is_outside_driving_lanes(location)
        self._is_at_wrong_lane(location)

        if self._outside_lane_active or self._wrong_lane_active:
            self.test_status = "FAILURE"
            outside_lane = TrafficEvent(event_type=TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION)
            outside_lane.set_message("Agent went outside its route lanes")
            print("Agent went outside its route lanes")
            self.list_traffic_events.append(outside_lane)
            self.mis_error[0] = True

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        If there is currently an event running, it is registered
        """
        super(OutsideRouteLanesTest, self).terminate(new_status)


class CustomActorSpeedAboveThresholdTest(ActorSpeedAboveThresholdTest):

    def __init__(self, actor, speed_threshold, below_threshold_max_time,
                 name="ActorSpeedAboveThresholdTest", terminate_on_failure=False, mis_error=None):
        super(CustomActorSpeedAboveThresholdTest, self).__init__(actor=actor, speed_threshold=speed_threshold, below_threshold_max_time=below_threshold_max_time,
                 name=name, terminate_on_failure=terminate_on_failure)
        self.mis_error = mis_error

    def update(self):
        """
        Check if the actor speed is above the speed_threshold
        """
        new_status = py_trees.common.Status.RUNNING

        linear_speed = CarlaDataProvider.get_velocity(self._actor)
        if linear_speed is not None:
            if linear_speed < self._speed_threshold and self._time_last_valid_state:
                if (GameTime.get_time() - self._time_last_valid_state) > self._below_threshold_max_time:
                    # Game over. The actor has been "blocked" for too long
                    self.test_status = "FAILURE"
                    
                    self.mis_error[0] = True

                    # record event
                    vehicle_location = CarlaDataProvider.get_location(self._actor)
                    blocked_event = TrafficEvent(event_type=TrafficEventType.VEHICLE_BLOCKED)
                    ActorSpeedAboveThresholdTest._set_event_message(blocked_event, vehicle_location)
                    ActorSpeedAboveThresholdTest._set_event_dict(blocked_event, vehicle_location)
                    self.list_traffic_events.append(blocked_event)

            else:
                self._time_last_valid_state = GameTime.get_time()

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


