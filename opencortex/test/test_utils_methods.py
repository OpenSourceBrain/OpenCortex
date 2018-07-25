#####################
### Subject to change without notice!!
#####################
##############################################################
### Author : Rokas Stanislovas
###
### GSoC 2016 project: Cortical Networks
###
##############################################################

import opencortex.core as oc
import opencortex.build as oc_build
import opencortex.utils as oc_utils
import neuroml
import numpy as np
import os
import math

try:
    import unittest2 as unittest
except ImportError:
    import unittest
    

class TestUtilsMethods(unittest.TestCase):

      def test_read_connectivity(self):
      
          proj_info=oc_utils.read_connectivity(pre_pop='CG3D_L23PyrRS',post_pop='CG3D_L23PyrFRB',path_to_txt_file='ConnListTest')
          
          self.assertTrue(isinstance(proj_info,list))
          
          for proj_ind in range(0,len(proj_info)):
          
              self.assertEqual(len(proj_info[proj_ind].keys() ), len(proj_info[proj_ind].values() ) )
          
              self.assertTrue( 'SynapseList' in proj_info[proj_ind].keys() )
              self.assertTrue( 'PreCellGroup' in proj_info[proj_ind].keys() )
              self.assertTrue( 'PostCellGroup' in proj_info[proj_ind].keys() )
              self.assertTrue( 'Type' in proj_info[proj_ind].keys() )
              self.assertTrue( 'LocOnPostCell' in proj_info[proj_ind].keys() )
              self.assertTrue( 'NumPerPostCell' in proj_info[proj_ind].keys() or 'NumPerPreCell' in proj_info[proj_ind].keys() )
              

      def test_add_populations_in_rectangular_layers(self): 
      
          network = neuroml.Network(id='Net0')
          popDict={}
          popDict['CG3D_L23PyrRS'] = (1000, 'L23','Test','multi',None)
          popDict['CG3D_L23PyrFRB']= (50,'L23','Test2','multi',None)
          t1=-0
          t2=-250
          t3=-250
          boundaries={}
          boundaries['L1']=[0,t1]
          boundaries['L23']=[t1,t1+t2+t3]
          xs = [0,500]
          zs = [0,500] 
          
          pop_params=oc_utils.add_populations_in_rectangular_layers(net=network,
                                                                    boundaryDict=boundaries,
                                                                    popDict=popDict,
                                                                    x_vector=xs,
                                                                    z_vector=zs,
                                                                    storeSoma=False)
          
          for cell_pop in popDict.keys():
          
             self.assertTrue( cell_pop in pop_params.keys() )
             
             self.assertTrue('PopObj' in pop_params[cell_pop] )
             
             self.assertTrue('Positions' in pop_params[cell_pop] )
             
             self.assertTrue('Compartments' in pop_params[cell_pop] )
             
             self.assertTrue(pop_params[cell_pop]['Positions']==None)
             
          for pop_index in range(0,len(network.populations)):
          
              pop=network.populations[pop_index]
              
              self.assertTrue(pop.id in popDict.keys() )
              
              self.assertTrue(popDict[pop.id][2]==pop.component )
              
              self.assertTrue(popDict[pop.id][0]==pop.size )
              
              for cell_loc in range(0,len(pop.instances) ):
              
                  instance_case=pop.instances[cell_loc]
                  
                  location=instance_case.location
                  
                  self.assertTrue(xs[0] <= location.x and location.x <=xs[1] )
                  
                  self.assertTrue(zs[0] <= location.z and location.z <= zs[1] )
                  
                  self.assertTrue(boundaries[popDict[pop.id][1]][0] >= location.y and location.y >= boundaries[popDict[pop.id][1]][1] )  
                  
          ###### check storing of soma positions
          
          network = neuroml.Network(id='Net1')
          
          pop_params=oc_utils.add_populations_in_rectangular_layers(net=network,
                                                                    boundaryDict=boundaries,
                                                                    popDict=popDict,
                                                                    x_vector=xs,
                                                                    z_vector=zs)
             
          for pop_index in range(0,len(network.populations)):
          
              pop=network.populations[pop_index]
              
              stored_cell_positions=pop_params[pop.id]['Positions']
              
              for cell_loc in range(0,len(pop.instances) ):
                  
                  instance_case=pop.instances[cell_loc]
                  
                  location=instance_case.location
                  
                  check_x= abs(location.x-stored_cell_positions[cell_loc][0]) < 0.00005 
                  check_y= abs(location.y-stored_cell_positions[cell_loc][1]) < 0.00005         
                  check_z= abs(location.z-stored_cell_positions[cell_loc][2]) < 0.00005  
                  
                  self.assertTrue(check_x)       
                  
                  self.assertTrue(check_y)
                  
                  self.assertTrue(check_z) 
                  
          ###### check distances between somata
          
          network = neuroml.Network(id='Net1')
          
          cell_diameters={}
          
          for pop_id in popDict.keys():
          
              cell_diameter=oc_build.get_soma_diameter(popDict[pop_id][2])
              
              cell_diameters[popDict[pop_id][2]]=cell_diameter
          
          pop_params=oc_utils.add_populations_in_rectangular_layers(net=network,
                                                                    boundaryDict=boundaries,
                                                                    popDict=popDict,
                                                                    x_vector=xs,
                                                                    z_vector=zs,
                                                                    cellBodiesOverlap=False,
                                                                    cellDiameterArray=cell_diameters)
                    
          for pop_index in range(0,len(network.populations)):
          
              pop=network.populations[pop_index]
              
              stored_cell_positions=pop_params[pop.id]['Positions']
              
              for cell_loc in range(0,len(pop.instances) ):
              
                  instance_case=pop.instances[cell_loc]
                  
                  location=instance_case.location
                  
                  check_x= abs(location.x-stored_cell_positions[cell_loc][0]) < 0.00001  
                  
                  check_y= abs(location.y-stored_cell_positions[cell_loc][1]) < 0.00001           
                  
                  check_z= abs(location.z-stored_cell_positions[cell_loc][2]) < 0.00001   
                  
                  self.assertTrue(check_x)       
                  
                  self.assertTrue(check_y)
                  
                  self.assertTrue(check_z) 
                  
                  for cell_loc_inner in range(0,len(pop.instances) ):
                  
                      if cell_loc != cell_loc_inner:
                      
                         inner_instance_case=pop.instances[cell_loc]
                  
                         inner_location=inner_instance_case.location
                         
                         d=oc_build.distance([location.x, location.y,location.z],[inner_location.x,inner_location.y,inner_location.z]) 
                         
                         self.assertTrue( d < (cell_diameters[pop.component]+cell_diameters[pop.component] )/2 )
                         
                  for pop_index_inner in range(0,len(network.populations)):
                  
                      pop_inner=network.populations[pop_index_inner]
                      
                      if pop.id != pop_inner.id:
                      
                         for cell_loc_inner in range(0,len(pop_inner.instances) ):
                  
                             if cell_loc != cell_loc_inner:
                      
                                inner_instance_case=pop.instances[cell_loc]
                  
                                inner_location=inner_instance_case.location
                         
                                d=oc_build.distance([location.x, location.y,location.z],[inner_location.x,inner_location.y,inner_location.z]) 
                         
                                self.assertTrue( d < (cell_diameters[pop_inner.component]+cell_diameters[pop_inner.component] )/2 )
                                
                         
      def test_add_populations_in_cylindrical_layers(self): 
      
          network = neuroml.Network(id='Net0')
          popDict={}
          popDict['CG3D_L23PyrRS'] = (1000, 'L23','Test','multi',None)
          popDict['CG3D_L23PyrFRB']= (50,'L23','Test2','multi',None)
          t1=-0
          t2=-250
          t3=-250
          boundaries={}
          boundaries['L1']=[0,t1]
          boundaries['L23']=[t1,t1+t2+t3]
          xs = [0,500]
          zs = [0,500] 
          
          pop_params=oc_utils.add_populations_in_cylindrical_layers(net=network,
                                                                    boundaryDict=boundaries,
                                                                    popDict=popDict,
                                                                    radiusOfCylinder=250,
                                                                    storeSoma=False)
          
          for cell_pop in popDict.keys():
          
             self.assertTrue( cell_pop in pop_params.keys() )
             
             self.assertTrue('PopObj' in pop_params[cell_pop] )
             
             self.assertTrue('Positions' in pop_params[cell_pop] )
             
             self.assertTrue('Compartments' in pop_params[cell_pop] )
             
             self.assertTrue(pop_params[cell_pop]['Positions']==None)
             
          for pop_index in range(0,len(network.populations)):
          
              pop=network.populations[pop_index]
              
              self.assertTrue(pop.id in popDict.keys() )
              
              self.assertTrue(popDict[pop.id][2]==pop.component )
              
              self.assertTrue(popDict[pop.id][0]==pop.size )
              
              for cell_loc in range(0,len(pop.instances) ):
              
                  instance_case=pop.instances[cell_loc]
                  
                  location=instance_case.location
                  
                  self.assertTrue(oc_build.distance([location.x,location.z],[0,0])  <= 250 )
                  
                  self.assertTrue(boundaries[popDict[pop.id][1]][0] >= location.y and location.y >= boundaries[popDict[pop.id][1]][1] )  
                  
          ###### check storing of soma positions
          
          network = neuroml.Network(id='Net1')
          
          pop_params=oc_utils.add_populations_in_cylindrical_layers(net=network,
                                                                    boundaryDict=boundaries,
                                                                    popDict=popDict,
                                                                    radiusOfCylinder=250)
             
          for pop_index in range(0,len(network.populations)):
          
              pop=network.populations[pop_index]
              
              stored_cell_positions=pop_params[pop.id]['Positions']
              
              for cell_loc in range(0,len(pop.instances) ):
              
                  instance_case=pop.instances[cell_loc]
                  
                  location=instance_case.location
                  
                  check_x= abs(location.x-stored_cell_positions[cell_loc][0]) < 0.00000001  
                  
                  check_y= abs(location.y-stored_cell_positions[cell_loc][1]) < 0.00000001           
                  
                  check_z= abs(location.z-stored_cell_positions[cell_loc][2]) < 0.00000001   
                  
                  self.assertTrue(check_x)       
                  
                  self.assertTrue(check_y)
                  
                  self.assertTrue(check_z) 
                  
          ###### check distances between somata
          
          network = neuroml.Network(id='Net1')
          
          cell_diameters={}
          
          for pop_id in popDict.keys():
          
              cell_diameter=oc_build.get_soma_diameter(popDict[pop_id][2])
              
              cell_diameters[popDict[pop_id][2]]=cell_diameter
          
          pop_params=oc_utils.add_populations_in_cylindrical_layers(net=network,
                                                                    boundaryDict=boundaries,
                                                                    popDict=popDict,
                                                                    radiusOfCylinder=250,
                                                                    cellBodiesOverlap=False,
                                                                    cellDiameterArray=cell_diameters)
             
          for pop_index in range(0,len(network.populations)):
          
              pop=network.populations[pop_index]
              
              stored_cell_positions=pop_params[pop.id]['Positions']
              
              for cell_loc in range(0,len(pop.instances) ):
              
                  instance_case=pop.instances[cell_loc]
                  
                  location=instance_case.location
                  
                  check_x= abs(location.x-stored_cell_positions[cell_loc][0]) < 0.00000001  
                  
                  check_y= abs(location.y-stored_cell_positions[cell_loc][1]) < 0.00000001           
                  
                  check_z= abs(location.z-stored_cell_positions[cell_loc][2]) < 0.00000001   
                  
                  self.assertTrue(check_x)       
                  
                  self.assertTrue(check_y)
                  
                  self.assertTrue(check_z) 
                  
                  for cell_loc_inner in range(0,len(pop.instances) ):
                  
                      if cell_loc != cell_loc_inner:
                      
                         inner_instance_case=pop.instances[cell_loc]
                  
                         inner_location=inner_instance_case.location
                         
                         d=oc_build.distance([location.x, location.y,location.z],[inner_location.x,inner_location.y,inner_location.z]) 
                         
                         self.assertTrue( d < (cell_diameters[pop.component]+cell_diameters[pop.component] )/2 )
                         
                  for pop_index_inner in range(0,len(network.populations)):
                  
                      pop_inner=network.populations[pop_index_inner]
                      
                      if pop.id != pop_inner.id:
                      
                         for cell_loc_inner in range(0,len(pop_inner.instances) ):
                  
                             if cell_loc != cell_loc_inner:
                      
                                inner_instance_case=pop.instances[cell_loc]
                  
                                inner_location=inner_instance_case.location
                         
                                d=oc_build.distance([location.x, location.y,location.z],[inner_location.x,inner_location.y,inner_location.z]) 
                         
                                self.assertTrue( d < (cell_diameters[pop_inner.component]+cell_diameters[pop_inner.component] )/2 )   
                                
          #### check cell distribution in the regular polygon, number of sides =6 
          
          network = neuroml.Network(id='Net1')
          
          pop_params=oc_utils.add_populations_in_cylindrical_layers(net=network,
                                                                    boundaryDict=boundaries,
                                                                    popDict=popDict,
                                                                    radiusOfCylinder=250,
                                                                    numOfSides=6)
          
          for cell_pop in popDict.keys():
          
             self.assertTrue( cell_pop in pop_params.keys() )
             
             self.assertTrue('PopObj' in pop_params[cell_pop] )
             
             self.assertTrue('Positions' in pop_params[cell_pop] )
             
             self.assertTrue('Compartments' in pop_params[cell_pop] )
             
          vertex_array=[]
         
          xy_sides=[]
         
          angle_array=np.linspace(0, 2*math.pi*(1-(1.0 /6) ),6)
         
          for angle in angle_array:
         
             vertex=[]
             
             x=250*math.cos(angle)
             
             y=250*math.sin(angle)
             
             vertex.append(x)
             
             vertex.append(y)
             
             vertex_array.append(vertex)
             
          for v_ind in range(0,len(vertex_array)):
         
             v1=vertex_array[v_ind]
             
             v2=vertex_array[v_ind-1]
             
             if abs(v1[0] - v2[0]) > 0.00000001 and abs(v1[1] -v2[1]) > 0.00000001:
             
                A=np.array([[v1[0],1],[v2[0],1]])
             
                b=np.array([v1[1],v2[1]])
             
                xcyc=np.linalg.solve(A,b)
             
                xy_sides.append(list(xcyc))
                
             else:
             
                if abs(v1[0] - v2[0]) <= 0.00000001:
                   
                   xy_sides.append([v1[0], None] )
                   
                if abs(v1[1] -v2[1] ) <= 0.00000001:
                
                   xy_sides.append([None,v1[1]] )
             
          for pop_index in range(0,len(network.populations)):
          
              pop=network.populations[pop_index]
              
              self.assertTrue(pop.id in popDict.keys() )
              
              self.assertTrue(popDict[pop.id][2]==pop.component )
              
              self.assertTrue(popDict[pop.id][0]==pop.size )
              
              for cell_loc in range(0,len(pop.instances) ):
              
                  instance_case=pop.instances[cell_loc]
                  
                  location=instance_case.location
                  
                  self.assertTrue(oc_build.distance([location.x,location.z],[0,0])  <= 250 )
                  
                  self.assertTrue(boundaries[popDict[pop.id][1]][0] >= location.y and location.y >= boundaries[popDict[pop.id][1]][1] ) 
                  
                  count_intersections=0
                  
                  for side_index in range(0,len(xy_sides) ):
                 
                      if abs(vertex_array[side_index][1] - vertex_array[side_index-1][1]) > 0.0000001:
            
                         if location.z < vertex_array[side_index][1] and location.z > vertex_array[side_index-1][1] :
                         
                             if xy_sides[side_index][0] !=None and xy_sides[side_index][1]==None :
                    
                                if location.x <= xy_sides[side_index][0]:
                       
                                   count_intersections+=1
                    
                             if xy_sides[side_index][0] != None and xy_sides[side_index][1] != None:
                    
                                if location.x <= (location.z - xy_sides[side_index][1]) / xy_sides[side_index][0]:
                       
                                   count_intersections +=1
                    
                         if location.z < vertex_array[side_index-1][1] and location.z > vertex_array[side_index][1]:
                    
                            if xy_sides[side_index][0] !=None and xy_sides[side_index][1]==None :
                    
                               if location.x <= xy_sides[side_index][0]:
                       
                                  count_intersections+=1
                    
                            if xy_sides[side_index][0] != None and xy_sides[side_index][1] != None:
                    
                               if location.x <= (location.z - xy_sides[side_index][1]) / xy_sides[side_index][0]:    
                               
                                  count_intersections +=1
                                  
                  self.assertTrue( count_intersections ==1)        
      ############################################################################# 
                  
      def test_check_cached_dicts(self):
      
          cached_target_dict={}
          
          PostSegLengthDict, cached_target_dict =oc_utils.check_cached_dicts(cell_component="Test",
                                                                             cached_dicts=cached_target_dict,
                                                                             list_of_target_seg_groups=['basal_obl_dends'],
                                                                             path_to_nml2=None) 
                                                                             
          self.assertTrue( 'basal_obl_dends' in PostSegLengthDict.keys() )
          
          self.assertTrue('SegList' in PostSegLengthDict['basal_obl_dends'].keys() )
          
          self.assertTrue(PostSegLengthDict['basal_obl_dends']['SegList'] != None)
          
          self.assertTrue('LengthDist' in PostSegLengthDict['basal_obl_dends'].keys() )
          
          self.assertTrue(PostSegLengthDict['basal_obl_dends']['LengthDist'] != None )
          
          self.assertTrue( 'Test' in cached_target_dict.keys() )
          
          self.assertTrue( 'TargetDict' in cached_target_dict['Test'].keys() )
          
          self.assertTrue( cached_target_dict['Test']['TargetDict']['basal_obl_dends']['SegList'] != None)
          
          self.assertTrue( cached_target_dict['Test']['TargetDict']['basal_obl_dends']['LengthDist'] != None)
          
          self.assertTrue( 'CellObject' in cached_target_dict['Test'].keys() )
          
          self.assertTrue(cached_target_dict['Test']['CellObject'].id =="L23PyrRS" )
          
          PostSegLengthDict, cached_target_dict =oc_utils.check_cached_dicts(cell_component="Test",
                                                                             cached_dicts=cached_target_dict,
                                                                             list_of_target_seg_groups=['distal_axon'],
                                                                             path_to_nml2=None) 
                                                                             
          self.assertTrue( 'basal_obl_dends' not in PostSegLengthDict.keys() )
          
          self.assertTrue( 'distal_axon' in PostSegLengthDict.keys() )
          
          self.assertTrue('SegList' in PostSegLengthDict['distal_axon'].keys() )
          
          self.assertTrue(PostSegLengthDict['distal_axon']['SegList'] != None)
          
          self.assertTrue('LengthDist' in PostSegLengthDict['distal_axon'].keys() )
          
          self.assertTrue(PostSegLengthDict['distal_axon']['LengthDist'] != None )
          
          self.assertTrue( 'Test' in cached_target_dict.keys() )
          
          self.assertTrue( 'TargetDict' in cached_target_dict['Test'].keys() )
          
          self.assertTrue( cached_target_dict['Test']['TargetDict']['basal_obl_dends']['SegList'] != None)
          
          self.assertTrue( cached_target_dict['Test']['TargetDict']['basal_obl_dends']['LengthDist'] != None)
          
          self.assertTrue( cached_target_dict['Test']['TargetDict']['distal_axon']['SegList'] != None)
          
          self.assertTrue( cached_target_dict['Test']['TargetDict']['distal_axon']['LengthDist'] != None)
          
          self.assertTrue( 'CellObject' in cached_target_dict['Test'].keys() )
          
          self.assertTrue(cached_target_dict['Test']['CellObject'].id =="L23PyrRS" )
          
      def test_build_connectivity(self):
      
          network = neuroml.Network(id='Net0')
          popDict={}
          popDict['CG3D_L23PyrRS'] = (1000, 'L23','Test','multi',None)
          popDict['CG3D_L23PyrFRB']= (50,'L23','Test2','multi',None)
          t1=-0
          t2=-250
          t3=-250
          boundaries={}
          boundaries['L1']=[0,t1]
          boundaries['L23']=[t1,t1+t2+t3]
          xs = [0,500]
          zs = [0,500] 
          
          pop_params=oc_utils.add_populations_in_rectangular_layers(net=network,
                                                                    boundaryDict=boundaries,
                                                                    popDict=popDict,
                                                                    x_vector=xs,
                                                                    z_vector=zs)
          
          all_synapse_components,proj_array,cached_segment_dicts=oc_utils.build_connectivity(net=network,
                                                                                             pop_objects=pop_params,
                                                                                             path_to_cells=None,
                                                                                             full_path_to_conn_summary='ConnListTest',
                                                                                             pre_segment_group_info=[{'PreSegGroup':"distal_axon",'ProjType':'Chem'}],
                                                                                             synaptic_scaling_params=[{'weight':2.0,
                                                                                                                       'synComp':'AMPA',
                                                                                                                       'synEndsWith':[],
                                                                                                                       'targetCellGroup':[]}],
                                                                                             synaptic_delay_params=[{'delay':0.05,'synComp':'all'}])               
                  
       
          self.assertTrue(all_synapse_components != [])
          
          self.assertTrue(proj_array != [])
          
          self.assertTrue(cached_segment_dicts != [])
          
          self.assertTrue( network.projections != [])
          
          self.assertTrue( network.electrical_projections != [])
          
          self.assertTrue(len(network.projections)==8)
          
          num_of_checked_chemical_projections=0
          
          num_of_checked_electrical_projections=0
          
          test_group_segments={}
          
          for pre_pop_id in popDict.keys():
          
              for post_pop_id in popDict.keys():

                  proj_summary=oc_utils.read_connectivity(pre_pop=pre_pop_id,post_pop=post_pop_id,path_to_txt_file='ConnListTest')
                  
                  for proj_ind in range(0,len(proj_summary)):
                          
                      projInfo=proj_summary[proj_ind]
                      
                      synapse_list=projInfo['SynapseList'] 
                      
                      num_per_post_cell=float(projInfo['NumPerPostCell'])
                      
                      for syn_ind in range(0,len(synapse_list) ):
                      
                          if projInfo['Type']=='Chem':
                      
                             for net_proj in range(0,len(network.projections) ):
                         
                                 proj=network.projections[net_proj]
                             
                                 if proj.presynaptic_population==pre_pop_id and proj.postsynaptic_population==post_pop_id and proj.synapse==synapse_list[syn_ind]:
                                 
                                    num_of_checked_chemical_projections+=1
                                    
                                    pre_segments=[146, 147, 142, 143]
                                    
                                    post_segments=[16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 24, 25, 22, 23, 20, 21, 18, 19, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 48, 49, 46, 47, 44, 45, 42, 43, 64, 65, 62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 72, 73, 70, 71, 68, 69, 66, 67]
                                    
                                    self.assertTrue( len(proj.connection_wds) ==float(num_per_post_cell) * popDict[post_pop_id][0]    )
                                    
                                    for conn_index in range(0,len(proj.connection_wds) ):
                                    
                                        connection=proj.connection_wds[conn_index]
                                        
                                        self.assertTrue( connection.pre_segment_id in pre_segments )
                                        
                                        self.assertTrue( connection.post_segment_id in post_segments )
                                        
                                        delay_value=float(connection.delay.split(" ")[0] )
                                        
                                        self.assertTrue( delay_value == 0.05)
                                        
                                        if 'AMPA' in proj.synapse:
                                        
                                           self.assertTrue(connection.weight ==2 )
                                    
                                    break
                                    
                          if projInfo['Type']=='Elect':
                                    
                             for net_proj in range(0,len(network.electrical_projections) ):
                             
                                 proj=network.electrical_projections[net_proj]
                                 
                                 check_synapse=proj.electrical_connection_instance_ws[0].synapse==synapse_list[syn_ind]
                                 
                                 if proj.presynaptic_population==pre_pop_id and proj.postsynaptic_population==post_pop_id and check_synapse:
                                 
                                    num_of_checked_electrical_projections+=1
                                    
                                    pre_segments=[136, 137, 138, 139, 144, 145, 140, 141, 146, 147, 142, 143]
                                    
                                    post_segments=[136, 137, 138, 139, 144, 145, 140, 141, 146, 147, 142, 143]
                                    
                                    for conn_index in range(0,len(proj.electrical_connection_instance_ws) ):
                                    
                                        connection= proj.electrical_connection_instance_ws[conn_index]
                                        
                                        self.assertTrue( connection.pre_segment in pre_segments )
                                        
                                        self.assertTrue( connection.post_segment in post_segments )
                                        
                                    break
                                       
          self.assertTrue(num_of_checked_chemical_projections == 8) 
          
      def test_probability_based_connectivity(self):
      
          network = neuroml.Network(id='Net0')
          popDict={}
          popDict['CG3D_L23PyrRS'] = (1000, 'L23','Test','single',None)
          popDict['CG3D_L23PyrFRB']= (50,'L23','Test2','single',None)
          t1=-0
          t2=-250
          t3=-250
          boundaries={}
          boundaries['L1']=[0,t1]
          boundaries['L23']=[t1,t1+t2+t3]
          xs = [0,500]
          zs = [0,500] 
          
          pop_params=oc_utils.add_populations_in_rectangular_layers(net=network,
                                                                    boundaryDict=boundaries,
                                                                    popDict=popDict,
                                                                    x_vector=xs,
                                                                    z_vector=zs)
                                                                    
          prob_matrix=[[0.0,0.5],
                       [0.9,0.0]]
                       
          syn_matrix=[[None,['Syn1','Syn2']],
                      [['Syn1','Syn2'],None]]
                      
          pop_tags=['L23PyrRS','L23PyrFRB']
          
          w_matrix=[[None,2.0],
                    [1.0,None]]
                    
          d_matrix=[[0.05,0.05],
                    [0.05,0.05]]
                                                                    
          proj_array=oc_utils.build_probability_based_connectivity(net=network,
                                                                   pop_params=pop_params,
                                                                   probability_matrix=prob_matrix, 
                                                                   synapse_matrix=syn_matrix,
                                                                   weight_matrix=w_matrix, 
                                                                   delay_matrix=d_matrix,
                                                                   tags_on_populations=pop_tags, 
                                                                   std_weight_matrix=None,
                                                                   std_delay_matrix=None)       
                                                                   
            
          self.assertTrue(len(network.projections)==4)
          
          found_projections=0
          
          count_zero_projections=0
          
          source_L23PyrFRB_projections=[]
          
          source_L23PyrRS_projections=[]
          
          for proj_ind in range(0,len(network.projections) ): 
          
             proj=network.projections[proj_ind]
             
             pre_id=proj.presynaptic_population
             
             post_id=proj.postsynaptic_population
             
             if pre_id =='CG3D_L23PyrFRB':
             
                source_L23PyrFRB_projections.append(proj.synapse)
                
             if post_id == 'CG3D_L23PyrRS':
                    
                source_L23PyrRS_projections.append(proj.synapse)
             
             for pop_tag_index in range(0,len(pop_tags)):
             
                 if pop_tags[pop_tag_index] in pre_id: 
                 
                    pre_pop_index=pop_tag_index
                    
                 if pop_tags[pop_tag_index] in post_id:
                 
                    post_pop_index=pop_tag_index
                
             for row_index in range(0,len(syn_matrix)):
             
                 for col_index in range(0,len(syn_matrix[row_index])):
                     
                     if isinstance(syn_matrix[row_index][col_index], list):
                     
                        for syn_component in range(0,len(syn_matrix[row_index][col_index])):
                        
                            check_component=syn_matrix[row_index][col_index][syn_component] == proj.synapse
                            
                            check_pre_pop=pop_tags[col_index] in pre_id 
                            
                            check_post_pop=pop_tags[row_index] in post_id 
             
                            if check_component and check_pre_pop and check_post_pop:
                               
                               found_projections+=1
             
             for conn_index in range(0,len(network.projections[proj_ind].connection_wds)):
             
                 connection=network.projections[proj_ind].connection_wds[conn_index]
                 
                 self.assertTrue( connection.weight == w_matrix[post_pop_index][pre_pop_index] )
                 
                 self.assertTrue(str( d_matrix[post_pop_index][pre_pop_index]) in connection.delay )
                                 
          self.assertTrue( found_projections ==4)                  
             
          self.assertTrue( set(source_L23PyrFRB_projections) == set (['Syn1','Syn2']) )
          
          self.assertTrue( set( source_L23PyrRS_projections) == set(['Syn1','Syn2'])  )
          
      def test_build_inputs(self):
          
          nml_doc, network = oc.generate_network('Net0')
          popDict={}
          popDict['CG3D_L23PyrRS'] = (2, 'L23','Test','multi',None)
          popDict['CG3D_L23PyrFRB']= (10,'L23','Test2','multi',None)
          popDict['CG3D_PointNeurons']=(1,'L23','SingleCompartment','single',None)
          t1=-0
          t2=-250
          t3=-250
          boundaries={}
          boundaries['L1']=[0,t1]
          boundaries['L23']=[t1,t1+t2+t3]
          xs = [0,50]
          zs = [0,50] 
          
          pop_params=oc_utils.add_populations_in_rectangular_layers(net=network,
                                                                    boundaryDict=boundaries,
                                                                    popDict=popDict,
                                                                    x_vector=xs,
                                                                    z_vector=zs)
                                                                    
          all_synapse_components,proj_array,cached_segment_dicts=oc_utils.build_connectivity(net=network,
                                                                                             pop_objects=pop_params,
                                                                                             path_to_cells=None,
                                                                                             full_path_to_conn_summary='ConnListTest',
                                                                                             pre_segment_group_info=[{'PreSegGroup':"distal_axon",'ProjType':'Chem'}],
                                                                                             synaptic_scaling_params=[{'weight':2.0,
                                                                                                                       'synComp':'AMPA',
                                                                                                                       'synEndsWith':[],
                                                                                                                       'targetCellGroup':[]}],
                                                                                             synaptic_delay_params=[{'delay':0.05,'synComp':'all'}])    
            
                                                                                             
          cell_nml_file = 'Test.cell.nml'
          
          document_cell = neuroml.loaders.NeuroMLLoader.load(cell_nml_file)
                                
          cell_object=document_cell.cells[0]
          
          CG3D_L23PyrRS_target_segs=oc_build.extract_seg_ids(cell_object,
                                                      target_compartment_array=['dendrite_group'],
                                                      targeting_mode='segGroups')
                                                      
         
          CG3D_L23PyrRS_target_segs=CG3D_L23PyrRS_target_segs['dendrite_group']
          
          cell_nml_file = 'Test2.cell.nml'
          
          document_cell = neuroml.loaders.NeuroMLLoader.load(cell_nml_file)
                                
          cell_object=document_cell.cells[0]
          
          CG3D_L23PyrFRB_target_segs=oc_build.extract_seg_ids(cell_object,
                                                      target_compartment_array=['dendrite_group'],
                                                      targeting_mode='segGroups')
                                                      
          CG3D_L23PyrFRB_target_segs=CG3D_L23PyrFRB_target_segs['dendrite_group']                                                                                   
                                                                                                  
          input_params ={'CG3D_L23PyrRS':[{'InputType':'PulseGenerators',
                         'InputName':"DepCurr_L23RS",
                         'Noise':True,
                         'SmallestAmplitudeList':[5.0E-5],
                         'LargestAmplitudeList':[1.0E-4],
                         'DurationList':[20000.0],
                         'DelayList':[0.0],
                         'TimeUnits':'ms',
                         'AmplitudeUnits':'uA',
                         'FractionToTarget':1.0,
                         'LocationSpecific':False,
                         'UniversalTargetSegmentID':0,
                         'UniversalFractionAlong':0.5},
                     
                        {'InputType':'GeneratePoissonTrains',
                         'InputName':"BackgroundL23RS",
                         'TrainType':'persistent',
                         'Synapse':'Syn_AMPA_SupPyr_SupPyr',
                         'AverageRateList':[30.0],
                         'RateUnits':'Hz',
                         'FractionToTarget':1.0,
                         'LocationSpecific':False,
                         'TargetDict':{'dendrite_group':100} } ],
                         
                         'CG3D_L23PyrFRB':[{'InputType':'GeneratePoissonTrains',
                         'InputName':"EctopicStimL23FRB",
                         'TrainType':'transient',
                         'Synapse':'SynForEctStim',
                         'AverageRateList':[150.0],
                         'DurationList':[1000.0],
                         'DelayList':[5000.0],
                         'RateUnits':'Hz',
                         'TimeUnits':'ms',
                         'FractionToTarget':0.5,
                         'LocationSpecific':False,
                         'UniversalTargetSegmentID':143,
                         'UniversalFractionAlong':0.5},
                     
                        {'InputType':'PulseGenerators',
                         'InputName':"DepCurr_L23FRB_spatial",
                         'Noise':False,
                         'AmplitudeList':[2.5E-4,3.0E-4],
                         'DurationList':[20000.0,10000.0],
                         'DelayList':[0.0,20000.0],
                         'TimeUnits':'ms',
                         'AmplitudeUnits':'uA',
                         'FractionToTarget':1.0,
                         'LocationSpecific':True,
                         'TargetRegions':[{'XVector': [0,50],'YVector':[0,-500],'ZVector': [0,50]}],
                         'TargetDict':{'dendrite_group':100}  }],
                         
                         'CG3D_PointNeurons':[{'InputType':'PulseGenerators',
                         'InputName':"DepCurr_CG3D_PointNeurons",
                         'Noise':True,
                         'SmallestAmplitudeList':[5.0E-5],
                         'LargestAmplitudeList':[1.0E-4],
                         'DurationList':[20000.0],
                         'DelayList':[0.0],
                         'TimeUnits':'ms',
                         'AmplitudeUnits':'nA',
                         'FractionToTarget':1.0,
                         'LocationSpecific':False,
                         'TargetDict': {None:100}   },
                         
                        {'InputType':'GenerateSpikeSourcePoisson',
                         'InputName':"SpikeSource",
                         'AverageRateList':[8.0],
                         'DurationList':[300.0],
                         'DelayList':[0.0],
                         'WeightList':[2.5],
                         'Synapse':'exp_curr_syn_all',
                         'RateUnits':'Hz',
                         'TimeUnits':'ms',
                         'FractionToTarget':1.0,
                         'LocationSpecific':False,
                         'TargetDict':{None:10} } ] } 
                         
          input_list_array_final, input_synapse_list=oc_utils.build_inputs(nml_doc=nml_doc,
                                                                           net=network,
                                                                           population_params=pop_params,
                                                                           input_params=input_params,
                                                                           cached_dicts=cached_segment_dicts,
                                                                           path_to_cells=None,
                                                                           path_to_synapses=None)  
                                                                           
          self.assertTrue(isinstance(input_list_array_final,list) and len(input_list_array_final) >=0)
          
          CG3D_L23PyrRS_pulses=[]
          
          CG3D_PointNeurons_spike_sources=[]
          
          CG3D_PointNeurons_spike_source_pops=[]
          
          CG3D_PointNeurons_pulses=[]
          
          CG3D_L23PyrFRB_spatial_pulses0=[]
          
          CG3D_L23PyrFRB_spatial_pulses1=[]
          
          CG3D_L23PyrRS_pulse_input_lists=[]
          
          CG3D_PointNeurons_spike_source_projections=[]
          
          CG3D_PointNeurons_pulse_input_lists=[]
          
          CG3D_PointNeurons_pulse_input_list_instances=[]
          
          CG3D_L23PyrFRB_spatial_pulse0_input_lists=[]
          
          CG3D_L23PyrFRB_spatial_pulse1_input_lists=[]
          
          G3D_L23PyrFRB_spatial_pulse0_input_list_instances=[]
          
          G3D_L23PyrFRB_spatial_pulse1_input_list_instances=[]
          
          CG3D_L23PyrRS_spike_source_proj_connections=[]
          
          CG3D_L23PyrRS_persistent_synapses=[]
          
          CG3D_L23PyrRS_persistent_synapse_input_lists=[]
          
          CG3D_L23PyrRS_persistent_synapse_input_list_instances=[]
          
          CG3D_L23PyrFRB_transient_synapses=[]
          
          CG3D_L23PyrFRB_transient_synapse_input_lists=[]
          
          CG3D_L23PyrFRB_transient_synapse_input_list_instances=[]
          
          for pulse_index in range(0,len(nml_doc.pulse_generators)):
          
              pulse_group=nml_doc.pulse_generators[pulse_index]
              
              delay=float(pulse_group.delay.split(" ")[0])
              
              amplitude=float(pulse_group.amplitude.split(" ")[0])
              
              duration=float(pulse_group.duration.split(" ")[0])
              
              delay_units=pulse_group.delay.split(" ")[1]
              
              amplitude_units=pulse_group.amplitude.split(" ")[1]
              
              duration_units=pulse_group.duration.split(" ")[1]
              
              if "DepCurr_L23RS" in pulse_group.id:
              
                 CG3D_L23PyrRS_pulses.append(pulse_group.id)
                 
                 ###self.assertTrue( amplitude <=1.0E-4 and amplitude >= 5.0E-5)
                 
                 self.assertTrue( duration == 20000.0)
                 
                 self.assertTrue( delay == 0.0 )
                 
                 self.assertTrue( delay_units=="ms")
                 
                 self.assertTrue( amplitude_units=='uA')
                 
                 self.assertTrue( duration_units=="ms" )
                 
              if ("DepCurr_L23FRB_spatial" in pulse_group.id)  and ("Pulse0" in pulse_group.id):
              
                 CG3D_L23PyrFRB_spatial_pulses0.append(pulse_group.id)
                 
                 self.assertTrue( amplitude == 2.5E-4)
                 
                 self.assertTrue( duration == 20000.0)
                 
                 self.assertTrue( delay == 0.0 )
                 
                 self.assertTrue( delay_units=="ms")
                 
                 self.assertTrue( amplitude_units=='uA')
                 
                 self.assertTrue( duration_units=="ms" )
                 
              if ("DepCurr_L23FRB_spatial" in pulse_group.id) and ("Pulse1" in pulse_group.id):
              
                 CG3D_L23PyrFRB_spatial_pulses1.append(pulse_group.id)
                 
                 self.assertTrue( amplitude== 3.0E-4)
                 
                 self.assertTrue( duration == 10000.0)
                 
                 self.assertTrue( delay == 20000.0 )
                 
                 self.assertTrue( delay_units=="ms")
                 
                 self.assertTrue( amplitude_units=='uA')
                 
                 self.assertTrue( duration_units=="ms" )
                 
              if "DepCurr_CG3D_PointNeurons" in pulse_group.id:
              
                 CG3D_PointNeurons_pulses.append(pulse_group.id)
                 
                 ##self.assertTrue(amplitude <=1.0E-4 and amplitude >= 5.0E-5 )
                 
                 self.assertTrue( duration == 20000.0)
                 
                 self.assertTrue( delay == 0.0 )
                 
                 self.assertTrue( delay_units=="ms")
                 
                 self.assertTrue( amplitude_units=='nA')
                 
                 self.assertTrue( duration_units=="ms" )
                 
          for poisson_index in range(0,len(nml_doc.poisson_firing_synapses)):
          
              poisson_synapse_case=nml_doc.poisson_firing_synapses[poisson_index]
              
              if "BackgroundL23RS" in poisson_synapse_case.id:
              
                 CG3D_L23PyrRS_persistent_synapses.append(poisson_synapse_case.id)
             
                 rate=float(poisson_synapse_case.average_rate.split(" ")[0])
              
                 rate_units=poisson_synapse_case.average_rate.split(" ")[1]
                 
                 self.assertTrue( rate == 30.0 )
                 
                 self.assertTrue( rate_units == "Hz")
                 
          for pop_index in range(0,len(network.populations)):
          
              pop=network.populations[pop_index]
              
              if "SpikeSource" in pop.component:
              
                 CG3D_PointNeurons_spike_source_pops.append(pop.id)
                 
          for spike_source_index in range(0,len(nml_doc.SpikeSourcePoisson)):
          
              spike_source_case=nml_doc.SpikeSourcePoisson[spike_source_index]
              
              if "SpikeSource" in spike_source_case.id:
              
                 CG3D_PointNeurons_spike_sources.append(spike_source_case.id)
                 
                 rate=float(spike_source_case.rate.split(" ")[0])
              
                 rate_units=spike_source_case.rate.split(" ")[1]
                 
                 self.assertTrue( rate == 8.0 )
                 
                 self.assertTrue( rate_units == "Hz")
                 
                 start=float(spike_source_case.start.split(" ")[0])
              
                 start_units=spike_source_case.start.split(" ")[1]
                 
                 self.assertTrue( start== 0.0     )
                 
                 self.assertTrue( start_units == "ms")
                 
                 duration=float(spike_source_case.duration.split(" ")[0])
              
                 duration_units=spike_source_case.duration.split(" ")[1]
                 
                 self.assertTrue( duration == 300.0    )
                 
                 self.assertTrue( duration_units == "ms" )
                 
          for poisson_index in range(0,len(nml_doc.transient_poisson_firing_synapses)):
          
              poisson_synapse_case=nml_doc.transient_poisson_firing_synapses[poisson_index]
              
              if "EctopicStimL23FRB" in poisson_synapse_case.id:
              
                 CG3D_L23PyrFRB_transient_synapses.append(poisson_synapse_case.id)
                 
                 rate=float(poisson_synapse_case.average_rate.split(" ")[0])
              
                 rate_units=poisson_synapse_case.average_rate.split(" ")[1]
                 
                 self.assertTrue( rate == 150.0 )
                 
                 self.assertTrue( rate_units == "Hz")
                 
                 duration=float(poisson_synapse_case.duration.split(" ")[0])
                 
                 duration_units=poisson_synapse_case.duration.split(" ")[1]
                 
                 delay=float(poisson_synapse_case.delay.split(" ")[0])
                 
                 delay_units=poisson_synapse_case.delay.split(" ")[1]
                 
                 self.assertTrue( duration == 1000.0 )
                 
                 self.assertTrue( delay ==5000.0 )
                 
                 self.assertTrue( duration_units == "ms")
                 
                 self.assertTrue( delay_units == "ms" )
                 
          for proj_index in range(0,len(network.projections)):
          
              proj=network.projections[proj_index]
              
              if (proj.presynaptic_population in CG3D_PointNeurons_spike_source_pops) and (proj.postsynaptic_population == 'CG3D_PointNeurons' ) \
              and proj.synapse=='exp_curr_syn_all':
              
                 CG3D_PointNeurons_spike_source_projections.append(proj.id)
                 
                 self.assertTrue(len(proj.connection_wds),10)
                 
                 for conn_index in range(0,len(proj.connection_wds)):
                 
                     conn=proj.connection_wds[conn_index]
                     
                     self.assertEqual(conn.weight,2.5 )
                     
                     self.assertEqual( '../CG3D_PointNeurons/0/SingleCompartment', conn.post_cell_id)
                 
          for input_list_index in range(0,len(network.input_lists)):
          
              input_list_instance=network.input_lists[input_list_index]
              
              if "BackgroundL23RS" in input_list_instance.id and "BackgroundL23RS" in input_list_instance.component:
              
                 CG3D_L23PyrRS_persistent_synapse_input_lists.append(input_list_instance.id)
                 
                 for input_index in range(0,len(input_list_instance.input)):
                 
                     input_case=input_list_instance.input[input_index]
                     
                     CG3D_L23PyrRS_persistent_synapse_input_list_instances.append(input_case.id)
                     
                     self.assertTrue(int(input_case.segment_id) in CG3D_L23PyrRS_target_segs)
                     
                     self.assertTrue(float(input_case.fraction_along) >=0 and float(input_case.fraction_along) <=1)
                     
              if "EctopicStimL23FRB" in input_list_instance.id and "EctopicStimL23FRB" in input_list_instance.component:
              
                 CG3D_L23PyrFRB_transient_synapse_input_lists.append(input_list_instance.id)
                 
                 for input_index in range(0,len(input_list_instance.input)):
                 
                     input_case=input_list_instance.input[input_index]
                     
                     CG3D_L23PyrFRB_transient_synapse_input_list_instances.append(input_case.id)
                     
                     self.assertTrue(int(input_case.segment_id) ==143)
                     
                     self.assertTrue(float(input_case.fraction_along) ==0.5)
              
              if "DepCurr_L23FRB_spatial" in input_list_instance.id and ("DepCurr_L23FRB_spatial" in input_list_instance.component and \
               "Pulse0" in input_list_instance.component):
              
                 CG3D_L23PyrFRB_spatial_pulse0_input_lists.append(input_list_instance.id)
                 
                 for input_index in range(0,len(input_list_instance.input)):
                 
                     input_case=input_list_instance.input[input_index]
                     
                     G3D_L23PyrFRB_spatial_pulse0_input_list_instances.append(input_case.id)
                     
                     self.assertTrue(int(input_case.segment_id) in CG3D_L23PyrFRB_target_segs)
                     
                     self.assertTrue(float(input_case.fraction_along) >=0 and float(input_case.fraction_along) <=1)
                     
              if "DepCurr_L23FRB_spatial" in input_list_instance.id and ("DepCurr_L23FRB_spatial" in input_list_instance.component and \
              'Pulse1' in input_list_instance.component):
              
                 CG3D_L23PyrFRB_spatial_pulse1_input_lists.append(input_list_instance.id)
                 
                 for input_index in range(0,len(input_list_instance.input)):
                 
                     input_case=input_list_instance.input[input_index]
                     
                     G3D_L23PyrFRB_spatial_pulse1_input_list_instances.append(input_case.id)
                     
                     self.assertTrue(int(input_case.segment_id) in CG3D_L23PyrFRB_target_segs)
                     
                     self.assertTrue(float(input_case.fraction_along) >=0 and float(input_case.fraction_along) <=1)
              
              if "DepCurr_L23RS" in input_list_instance.id and "DepCurr_L23RS" in input_list_instance.component:
              
                 CG3D_L23PyrRS_pulse_input_lists.append(input_list_instance.id)
                 
                 for input_index in range(0,len(input_list_instance.input)):
                 
                     input_case=input_list_instance.input[input_index]
                 
                     self.assertTrue( 0 ==int(input_case.segment_id) )
                 
                     self.assertTrue( 0.5 == float(input_case.fraction_along) )
                     
              if "DepCurr_CG3D_PointNeurons" in input_list_instance.id and "DepCurr_CG3D_PointNeurons" in input_list_instance.component:
              
                 CG3D_PointNeurons_pulse_input_lists.append(input_list_instance.id)
                 
                 for input_index in range(0,len(input_list_instance.input)):
                 
                     input_case=input_list_instance.input[input_index]
                     
                     CG3D_PointNeurons_pulse_input_list_instances.append(input_case.id)
                 
          self.assertEqual(len(set(CG3D_L23PyrRS_pulses)),1)
          
          self.assertEqual( len(set(CG3D_L23PyrRS_pulses)),len(set(CG3D_L23PyrRS_pulse_input_lists)) )
          
          self.assertEqual( len(set(CG3D_L23PyrFRB_spatial_pulses0)), 1)
          
          self.assertEqual(len(set(CG3D_L23PyrFRB_spatial_pulse0_input_lists)),1 )
          
          self.assertEqual( len(set(CG3D_L23PyrFRB_spatial_pulses1)), 1)
          
          self.assertEqual(len(set(CG3D_L23PyrFRB_spatial_pulse1_input_lists)), 1 )
          
          self.assertEqual(len(set(G3D_L23PyrFRB_spatial_pulse0_input_list_instances)), 1000)
          
          self.assertEqual(len(set(G3D_L23PyrFRB_spatial_pulse1_input_list_instances)), 1000)
          
          self.assertEqual(len(set(CG3D_PointNeurons_pulses)), 1)
         
          self.assertEqual(len(set(CG3D_PointNeurons_pulse_input_lists)), 1)
          
          ##self.assertEqual(len(set(CG3D_PointNeurons_pulse_input_list_instances)), 100)
          
          self.assertEqual(len(set(CG3D_L23PyrRS_persistent_synapses)), 1)
          
          self.assertEqual(len(set(CG3D_L23PyrRS_persistent_synapse_input_lists)), 1)
          
          self.assertEqual(len(set(CG3D_L23PyrRS_persistent_synapse_input_list_instances)), 200)
          
          self.assertEqual(len(set(CG3D_L23PyrFRB_transient_synapses)), 1)
          
          self.assertEqual(len(set(CG3D_L23PyrFRB_transient_synapse_input_lists)), 1)
          ### testing whether 50 % of cells are targeted, see input params
          self.assertEqual(len(set(CG3D_L23PyrFRB_transient_synapse_input_list_instances)), 5)
          
          self.assertTrue(len(set(CG3D_PointNeurons_spike_sources)), 1)
          
          self.assertTrue(len(set(CG3D_PointNeurons_spike_source_pops)),len(set(CG3D_PointNeurons_spike_sources)) )
          
          self.assertTrue(len(set(CG3D_PointNeurons_spike_source_projections)),len(set(CG3D_PointNeurons_spike_sources)) )
          
          
                   

