import math
from math import cos, sin
from shapely import Point, Polygon, LineString
import typing
class Mecanum():
    def __init__(self, max_motor_velocity, motor_acceleration, wheel_radius, track_width, wheelbase_length, start_point):
        #assumptions of same motor characteristics on all 4 wheels with orientations of pi/4, 3pi/4, -3pi/4, -pi/4

        #motor characteristics
        self.motor_velocities = [0,0,0,0]
        # 0 --> front left
        # 1 --> front right
        # 2 --> back left
        # 3 --> back right
        self.max_motor_velocity = max_motor_velocity # rotations/s
        self.motor_acceleration = motor_acceleration # rotation/s^2
        
        #wheel characteristics
        self.wheel_radius = wheel_radius # in

        #drivetrain characteristics
        self.track_width = track_width#left & right distance, in
        self.wheelbase_length = wheelbase_length#front & back distance, in


        #robot characteristics
        self.x = start_point[0] #based on 2-D center of robot
        self.y = start_point[1]
        self.theta = start_point[2] # rad

        #geometry
        self.robot_diagonal = math.sqrt(track_width**2 + wheelbase_length**2)
        robot_p1 = Point(self.x + cos(self.theta)*self.wheelbase_length + cos(self.theta)*self.track_width, 
                        self.y + sin(self.theta)*self.wheelbase_length + sin(self.theta)*self.track_width)
        robot_p2 = Point(self.x + cos(self.theta)*self.wheelbase_length - cos(self.theta)*self.track_width, 
                        self.y + sin(self.theta)*self.wheelbase_length - sin(self.theta)*self.track_width)
        robot_p3 = Point(self.x - cos(self.theta)*self.wheelbase_length + cos(self.theta)*self.track_width, 
                        self.y - sin(self.theta)*self.wheelbase_length + sin(self.theta)*self.track_width)
        robot_p4 = Point(self.x - cos(self.theta)*self.wheelbase_length - cos(self.theta)*self.track_width, 
                        self.y - sin(self.theta)*self.wheelbase_length - sin(self.theta)*self.track_width)
        
        self.robot_shape = Polygon([robot_p1, robot_p2, robot_p3, robot_p4])

        self.last_coords = (None, None, None)


    def move(self, power, strafe, turn, timestep):
        self.last_coords = (self.x, self.y, self.theta)
        #power, strafe, turn given in terms of 0-->1, like a controller
        denom = max(abs(power)+abs(strafe)+abs(turn),1)
        demanded_m0_v = (power + strafe + turn)/denom * self.max_motor_velocity
        demanded_m1_v = (power - strafe - turn)/denom * self.max_motor_velocity
        demanded_m2_v = (power - strafe + turn)/denom * self.max_motor_velocity
        demanded_m3_v = (power + strafe - turn)/denom * self.max_motor_velocity

        demanded_motor_velocities = [demanded_m0_v, demanded_m1_v, demanded_m2_v, demanded_m3_v]

        for i in range(4):
            if(abs(self.motor_velocities[i] - demanded_motor_velocities[i]) < self.motor_acceleration*timestep):
                self.motor_velocities[i] = demanded_motor_velocities[i]
            else:
                if(demanded_motor_velocities [i] < self.motor_velocities[i]):
                    self.motor_velocities[i] -= self.motor_acceleration*timestep
                else:
                    self.motor_velocities[i] += self.motor_acceleration*timestep

        v_forward = (self.motor_velocities[0] + self.motor_velocities[1] + self.motor_velocities[2] + self.motor_velocities[3])*self.wheel_radius/4
        v_strafe = (-self.motor_velocities[0] + self.motor_velocities[1] + self.motor_velocities[2] - self.motor_velocities[3])*self.wheel_radius/4
        w = (-self.motor_velocities[0] + self.motor_velocities[1] - self.motor_velocities[2] + self.motor_velocities[3])*self.wheel_radius/(2 * (self.track_width + self.wheelbase_length))
        
        
        
        self.theta += w*timestep
        self.x = self.x + v_forward * cos(self.theta) * timestep + v_strafe * cos(self.theta + math.pi/2) * timestep
        self.y = self.y + v_forward * sin(self.theta) * timestep + v_strafe * sin(self.theta + math.pi/2) * timestep
        robot_p1 = Point(self.x + cos(self.theta)*self.wheelbase_length + cos(self.theta)*self.track_width, 
                        self.y + sin(self.theta)*self.wheelbase_length + sin(self.theta)*self.track_width)
        robot_p2 = Point(self.x + cos(self.theta)*self.wheelbase_length - cos(self.theta)*self.track_width, 
                        self.y + sin(self.theta)*self.wheelbase_length - sin(self.theta)*self.track_width)
        robot_p3 = Point(self.x - cos(self.theta)*self.wheelbase_length + cos(self.theta)*self.track_width, 
                        self.y - sin(self.theta)*self.wheelbase_length + sin(self.theta)*self.track_width)
        robot_p4 = Point(self.x - cos(self.theta)*self.wheelbase_length - cos(self.theta)*self.track_width, 
                        self.y - sin(self.theta)*self.wheelbase_length - sin(self.theta)*self.track_width)
        
        self.robot_shape = Polygon([robot_p1, robot_p2, robot_p3, robot_p4])
        

    def return_characteristics(self):
        return "("+str(self.x) + ", " + str(self.y) + "), theta = " + str(self.theta) + " (rad), wheel velos = " + str(self.motor_velocities)
    
    def revert(self):
        self.x = self.last_coords[0]
        self.y = self.last_coords[1]
        self.theta = self.last_coords[2]
        robot_p1 = Point(self.x + cos(self.theta)*self.wheelbase_length + cos(self.theta)*self.track_width, 
                        self.y + sin(self.theta)*self.wheelbase_length + sin(self.theta)*self.track_width)
        robot_p2 = Point(self.x + cos(self.theta)*self.wheelbase_length - cos(self.theta)*self.track_width, 
                        self.y + sin(self.theta)*self.wheelbase_length - sin(self.theta)*self.track_width)
        robot_p3 = Point(self.x - cos(self.theta)*self.wheelbase_length + cos(self.theta)*self.track_width, 
                        self.y - sin(self.theta)*self.wheelbase_length + sin(self.theta)*self.track_width)
        robot_p4 = Point(self.x - cos(self.theta)*self.wheelbase_length - cos(self.theta)*self.track_width, 
                        self.y - sin(self.theta)*self.wheelbase_length - sin(self.theta)*self.track_width)
        
        self.robot_shape = Polygon([robot_p1, robot_p2, robot_p3, robot_p4])

    
    
    def point_contained(self, point: Point):
        return self.robot_shape.contains(point)
        
    def clips_polygons(self, list_of_polygons: typing.List[Polygon]):
        fill = []
        for poly in list_of_polygons:
            fill.append(self.robot_shape.overlaps(poly))
        return fill
    
    def out_of_bounds(self, bounds: typing.List[Point]):
        return not(Polygon(bounds).contains_properly(self.robot_shape))
        

    
    
    
    





        


