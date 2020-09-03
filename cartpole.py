


#import gym
import pybullet as p
#import pybullet_envs
import os
import pybullet_data #pybullet_data.getDataPath() #/home/roman/anaconda3/envs/roman_playful/lib/python3.6/site-packages/pybullet_data

#physicsClientId = p.connect(p.GUI) #self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
#p.loadURDF(fileName=os.path.join( pybullet_data.getDataPath() , "romans_urdf_files/octopus_files/python_scripts_edit_urdf/octopus_generated_1_links.urdf" ) )
#p.setGravity(gravX=0, gravY=0, gravZ=-9.8*1, physicsClientId = physicsClientId)

#while(1):
#for i in range(1000000):
#  p.stepSimulation()

#import pdb
#pdb.set_trace()

import pybullet as p
physicsClientId = p.connect(p.GUI)
#bot_id = p.loadURDF("/home/sgillen/work/scratch/roman_debug/bot.urdf" )
bot_id = p.loadURDF(fileName=os.path.join( pybullet_data.getDataPath() , "romans_urdf_files/octopus_files/python_scripts_edit_urdf/octopus_generated_1_links.urdf" ) )

p.setGravity(gravX=0, gravY=0, gravZ=-9.8*1, physicsClientId = physicsClientId)

# Joints will have dummy motors by default, turn those motors down to zero force for a free joint.
p.setJointMotorControlArray(bodyUniqueId=bot_id, jointIndices=[0], controlMode = p.POSITION_CONTROL, positionGains=[0.1], velocityGains=[0.1], forces=[0])

# Generally preferable stepping for watching / interacting with the physics
# Try both to see what I mean, don't mix the two approaches either.

p.setRealTimeSimulation(1)
#while(1):
#  p.stepSimulation()
input("Press any button to end")




