import numpy as np
import modern_robotics as mr
import yaml
from math import cos, sin, fabs, copysign

import tf
import rospy
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory
from nav_msgs.msg import Odometry
from sensor.msg import JointState

JOINT_NAMES = ['ur5_arm_shoulder_pan_joint', 'ur5_arm_shoulder_lift_joint', 'ur5_arm_elbow_joint','ur5_arm_wrist_1_joint', 'ur5_arm_wrist_2_joint', 'ur5_arm_wrist_3_joint']

class FeedbackController:

    def __init__(self, Kp, Ki, timestep, bot_params):
        
        self.Kp = np.eye(6) * Kp
        self.Ki = np.eye(6) * Ki
        self.timestep = timestep
        self.integral = np.zeros(6)

        # Vb = F*Δθ
        self.l = bot_params["chasis"]["l"]
        self.w = bot_params["chasis"]["w"]
        self.rad = bot_params["chasis"]["wheel_rad"]
        #self.height = bot_params["chasis"]["height"]
        self.height = 0
        self.limit_joint_vel = bot_params["velocity_limits"]["arm"]

        self.Tbo = np.array(bot_params["arm"]["Tbo"])  # chasis frame wrt arm base
        self.Moe = np.array(bot_params["arm"]["Moe"]) # Arm at home config (all joint angles zero).. end effector {e} wrt arm base {o}
        self.Slist = np.array(bot_params["arm"]["Slist"]) # Blist of arm

        self.thetalist=[0.0,0.0,0.0,0.0,0.0,0.0]
        self.phi = 0
        self.x = 0
        self.y = 0
        self.angles = [0.0,0.0,0.0]

        row_val = 1.0/(self.w)        
        F = (self.rad/2) * np.array([[-row_val, row_val],[1,1]])
        self.F6 = np.array([[0,0],[0,0],F[0],F[1],[0,0],[0,0]]) # this maps change in wheel angle to body twist

        rospy.init_node('controller',anonymous=True)
        self.sample_rate = 50.0

        self.pub1 = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pub2 = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)





    def FeedbackControl(self, config, Tse_des, Tse_des_next):
        '''
        Compute wheel and joint velocity required to achieve the desired configuration
        Input:
            config : {chasis: [phi, x , y], arm : [theta1, 2,3,4,5]}
            Tse_des : Desired configuration of end effector wrt space frame
            Tse_des_next : Desired configuration of end effector wrt space frame after timestep (used to calculate feedforward twist)
        
        Return:
            end_eff_twist : End effector twist wrt space frame required to move end effector from current transformation to the desired
            vels[:4] : Corresponds to required wheel velocities
            vels[4:] : Corresponds to required joint velocities
        '''
        #thetalist = np.array(config["arm"])
        #phi = config["chasis"][0]
        #x = config["chasis"][1]
        #y = config["chasis"][2]

        Tsb = np.array([[cos(self.phi),-sin(self.phi),0,self.x],[sin(self.phi),cos(self.phi),0,self.y],[0,0,1,self.height],[0,0,0,1]]) # chasis base wrt space frame

        Toe = mr.FKinSpace(self.Moe, self.Slist, self.thetalist) # base of the arm wrt to end effector
        Tse_curr = np.dot(np.dot(Tsb,self.Tbo),Toe) # Tse = Tsb * Tbo * Toe

        Tcurr_des = np.dot(mr.TransInv(Tse_curr),Tse_des) # Transform of desired wrt to current
        Tdes_des_next = np.dot(mr.TransInv(Tse_des), Tse_des_next) # Transform of next_desired wrt desired

        # Feedforward twist i.e twist required to transform Tdes to Tdes_next. This is for timestep hence 1/delta_time
        feedforward_twist_se3 = mr.MatrixLog6(Tdes_des_next) / self.timestep
        feedforward_twist_V = mr.se3ToVec(feedforward_twist_se3)

        # Error twist i.e twist when done for unit time moves from Tcurr to Tdes
        error_twist_se3 = mr.MatrixLog6(Tcurr_des)        
        error_twist_V = mr.se3ToVec(error_twist_se3)
        
        # end_eff_twist (wrt space frame) =  Adjoint(Tcurr_des)*Feedforward twist (Feedforward twist wrt to the current tranform) + Kp * error_twist + Ki * sum of error_twist
        feedforward_term = np.dot(mr.Adjoint(Tcurr_des), feedforward_twist_V)
        P_term = np.dot(self.Kp, error_twist_V)
        self.integral = self.integral + error_twist_V
        I_term = np.dot(self.Ki,self.integral*self.timestep)

        end_eff_twist = feedforward_term + P_term + I_term # End effector twist wrt to space frame

        Teb = np.dot(mr.TransInv(Toe), mr.TransInv(self.Tbo)) # chasis base wrt the end effector
        
        # J_base is the component of base velocity contributing to end effector velocity
        J_base = np.dot(mr.Adjoint(Teb),self.F6)
        # J_b is the body jacobian
        J_s = mr.JacobianSpace(self.Slist, self.thetalist)
        J_b = np.dot(mr.TransInv(mr.Adjoint(Toe)),J_s)


        # J_e is the jacobian to convert end effector twist to joint speed and wheel velocity
        J_e = np.hstack((J_base, J_b))

        # Checking if joints exceeds the defined joint limits
        while(True):
            vels = np.dot(np.linalg.pinv(J_e,1e-3),end_eff_twist) # [u,theta_dot].. tolerance of 1e-3 is added to pinv to avoid close to singularity situation
            constraint_joints = self.TestJointLimits(vels[4:], config["arm"])
            if(len(constraint_joints)):
                for const_joint in constraint_joints:
                    # Making the corresponding column of the constraint joints to zero in Je so that it indicates that these joints cannot be used to contribute to the end effector velocity
                    J_e[:,J_base.shape[1] + const_joint] = 0.0
            else:
                break

       # end_eff_twist, vels[:4], vels[4:] # V, wheel speed, joint speed
        
        vel = Twist()
        vel.linear.x = (vels[1]+vels[0])/2
		vel.linear.y = 0
		vel.linear.z = 0
		vel.angular.x = 0
		vel.angular.y = 0
		vel.angular.z = (vels[1]-vels[0])/2*self.w

        self.pub1.publish(vel)

        jointvel = JointTrajectory()
        jointvel.joint_names = JOINT_NAMES
        jointvel.points.velocities = [vels[5],vels[4],vels[3],vels[6],vels[7],vels[8]]

        self.pub2.publish(jointvel)
       



    def base_orientation(self,data)

        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y

        (self.angles[0], self.angles[1], self.angles[2]) = tf.transformations.euler_from_quaternion([data.pose.pose.orientation[0], data.pose.pose.orientation[1], data.pose.pose.orientation[2],data.pose.pose.orientation[3]])
        rospy.loginfo(self.angles[0])     
        self.phi = self.angles[0]   


    def arm_orientation(self,data)

        self.thetalist = [data.position[7],data.position[6],data.position[5],data.position[8],data.position[9],data.position[10]]      #check Order


    def TestJointLimits(self,joint_vels, curr_joint_pos):
        '''
        Tests if the joints if moved by the joint_vels for timestep will cross limit or not
        Input:
            joint_vels : List having joint velocities
            curr_joint_pos : Current position of the joints in the arm
        
        Return:
            constraint_joints : List of joints which crosses the limit
        '''
        joint_pos = [None] * len(joint_vels)
        constraint_joints = list()

        for joint_no in range(len(joint_vels)):
            # Checking if arm joint speed is off limits
            if fabs(joint_vels[joint_no]) > fabs(self.limit_joint_vel[joint_no]):
                print ("Joint velocity exceeded for Joint ",joint_no + 1, joint_vels[joint_no])
                joint_vels[joint_no] = copysign(self.limit_joint_vel[joint_no],joint_vels[joint_no])
                

            joint_pos[joint_no] = curr_joint_pos[joint_no] + (joint_vels[joint_no] * self.timestep)

            if(joint_pos[joint_no] > self.limit_joint_pos[joint_no][1] or joint_pos[joint_no] < self.limit_joint_pos[joint_no][0]):
                constraint_joints.append(joint_no)
                print ("Joint limit pos exceeded for Joint : ", joint_no + 1, joint_pos[joint_no])
        
        return constraint_joints 


    if __name__ == "__main__":
        
        params_filename = 'config/control_params.yaml'
        bot_params_filename = 'config/bot_params.yaml'

        # Reading params from the params file
        with open(params_filename) as file:  
            params = yaml.load(file)

        # Reading params from the params file
        with open(bot_params_filename) as file:  
            bot_params = yaml.load(file)

        

        feedback_controller = FeedbackController(params["Kp"], params["Ki"], params["timestep"], bot_params)




        r = rospy.Rate(feedback_controller.sample_rate)
        try:  
            while not rospy.is_shutdown():    
                rospy.Subscriber('/odometry/filtered',Odometry,self.base_orientation)
                rospy.Subscriber('/joint_states',JointState,self.arm_orientation)
                feedback_controller.FeedbackControl(params["config"], params["Tse_des"], params["Tse_des_next"])
                r.sleep()
        except ValueError as e:
                rospy.loginfo(e)