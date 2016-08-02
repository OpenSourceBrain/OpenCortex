#####################
### Subject to change without notice!!
#####################
##############################################################
### Author : Rokas Stanislovas
###
### GSoC 2016 project: Cortical Networks
###
##############################################################
import opencortex.build as oc
import opencortex.utils as oc_utils
import neuroml
import numpy as np
import os

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
              

      def test_add_populations_in_layers(self): 
      
          network = neuroml.Network(id='Net0')
          popDict={}
          popDict['CG3D_L23PyrRS'] = (1000, 'L23','Test')
          popDict['CG3D_L23PyrFRB']= (50,'L23','Test2')
          t1=-0
          t2=-250
          t3=-250
          boundaries={}
          boundaries['L1']=[0,t1]
          boundaries['L23']=[t1,t1+t2+t3]
          xs = [0,500]
          zs = [0,500] 
          
          pop_params=oc_utils.add_populations_in_layers(net=network,
                                                        boundaryDict=boundaries,
                                                        popDict=popDict,
                                                        x_vector=xs,
                                                        z_vector=zs)
          
          for cell_pop in popDict.keys():
          
             self.assertTrue( cell_pop in pop_params.keys() )
             
             self.assertTrue('PopObj' in pop_params[cell_pop] )
             
             self.assertTrue('Positions' in pop_params[cell_pop] )
             
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
          
          network.populations=[]
          
          pop_params=oc_utils.add_populations_in_layers(net=network,
                                                        boundaryDict=boundaries,
                                                        popDict=popDict,
                                                        x_vector=xs,
                                                        z_vector=zs,
                                                        storeSoma=True)
             
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
          popDict['CG3D_L23PyrRS'] = (1000, 'L23','Test')
          popDict['CG3D_L23PyrFRB']= (50,'L23','Test2')
          t1=-0
          t2=-250
          t3=-250
          boundaries={}
          boundaries['L1']=[0,t1]
          boundaries['L23']=[t1,t1+t2+t3]
          xs = [0,500]
          zs = [0,500] 
          
          pop_params=oc_utils.add_populations_in_layers(net=network,
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
                                 
                                 check_synapse=proj.electrical_connection_instances[0].synapse==synapse_list[syn_ind]
                                 
                                 if proj.presynaptic_population==pre_pop_id and proj.postsynaptic_population==post_pop_id and check_synapse:
                                 
                                    num_of_checked_electrical_projections+=1
                                    
                                    pre_segments=[136, 137, 138, 139, 144, 145, 140, 141, 146, 147, 142, 143]
                                    
                                    post_segments=[136, 137, 138, 139, 144, 145, 140, 141, 146, 147, 142, 143]
                                    
                                    for conn_index in range(0,len(proj.electrical_connection_instances) ):
                                    
                                        connection= proj.electrical_connection_instances[conn_index]
                                        
                                        self.assertTrue( connection.pre_segment in pre_segments )
                                        
                                        self.assertTrue( connection.post_segment in post_segments )
                                        
                                    break
                                       
          self.assertTrue(num_of_checked_chemical_projections == 8) 
          
          
    
             
          
             
             
             
             
             
             
             
             
          
     
          
          
          
          
          
          
          
          
        
          
          
          
         

