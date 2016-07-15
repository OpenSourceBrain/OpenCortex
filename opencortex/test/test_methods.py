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
    
class TestNetMorphMethods(unittest.TestCase):

      #########################################################################
      def test_extract_seg_ids(self):
          
          cell_nml_file = 'Test.cell.nml'
          
          document_cell = neuroml.loaders.NeuroMLLoader.load(cell_nml_file)
                                
          cell_object=document_cell.cells[0]
          
          test_return1=oc.extract_seg_ids(cell_object,
                                          target_compartment_array=['basal_obl_dends','most_prox_bas_dend'],
                                          targeting_mode='segGroups')
          
          
          self.assertEqual(set(['basal_obl_dends','most_prox_bas_dend']), set(test_return1.keys()))
          
          self.assertEqual(set(test_return1['basal_obl_dends']),set([16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 24, 25, 22, 23, 20, 21, 18, 19, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 48, 49, 46, 47, 44, 45, 42, 43, 64, 65, 62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 72, 73, 70, 71, 68, 69, 66, 67]))
          
          self.assertEqual(set(test_return1['most_prox_bas_dend']),set([16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3]) )
          
          test_return2=oc.extract_seg_ids(cell_object,
                                          target_compartment_array=['Seg1_comp_68','Seg1_comp_28'],
                                          targeting_mode='segments')
                                           
          self.assertEqual(set(['Seg1_comp_68','Seg1_comp_28']),set(test_return2.keys()) )
          self.assertEqual(set(test_return2['Seg1_comp_68']),set([135]) )
          self.assertEqual(set(test_return2['Seg1_comp_28']),set([55])  )
          
      ############################################################################################################   
      def test_get_seg_lengths(self):
          
          cell_nml_file = 'Test.cell.nml'
          
          document_cell = neuroml.loaders.NeuroMLLoader.load(cell_nml_file)
                                
          cell_object=document_cell.cells[0]
      
          length_dist1,segment_list1=oc.get_seg_lengths(cell_object,[55])
                                       
          length_dist2,segment_list2=oc.get_seg_lengths(cell_object,[135])        
          
          length_dist3,segment_list3=oc.get_seg_lengths(cell_object,[55,135])    
          
          self.assertEqual(len(length_dist1),len(segment_list1) )
          
          self.assertEqual(len(length_dist3),len(segment_list3) )
          
          self.assertEqual(length_dist1[0],25.00030750752478)
          
          self.assertEqual(length_dist2[0],24.99966139974902)
          
          self.assertEqual(set([55,135]),set(segment_list3))
          
          self.assertEqual(length_dist1[0],length_dist3[0])     
          
          self.assertEqual(length_dist1[0]+length_dist2[0],length_dist3[1])    
                  
      #######################################################################################################################                                
      def test_make_target_dict(self):   
          
          cell_nml_file = 'Test.cell.nml'
          
          document_cell = neuroml.loaders.NeuroMLLoader.load(cell_nml_file)
                                
          cell_object=document_cell.cells[0]           
          
          makeDict1=oc.make_target_dict(cell_object=cell_object,
                                       target_segs={'Seg1_comp_28':[55], 'Seg1_comp_68': [135]})
                               
          self.assertEqual(set(['Seg1_comp_28','Seg1_comp_68']),set(makeDict1.keys()))
          self.assertEqual(set(makeDict1['Seg1_comp_28'].keys()),set(['SegList','LengthDist']))
          
          self.assertEqual(len(makeDict1['Seg1_comp_68']['SegList']),len(makeDict1['Seg1_comp_68']['LengthDist']))
          
          makeDict2=oc.make_target_dict(cell_object=cell_object,
                                        target_segs={'basal_obl_dends': [16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 24, 25, 22, 23, 20, 21, 18, 19, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 48, 49, 46, 47, 44, 45, 42, 43, 64, 65, 62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 72, 73, 70, 71, 68, 69, 66, 67], 'most_prox_bas_dend': [16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3]})
                                        
          print  makeDict2
                                        
          self.assertEqual(len(makeDict2['basal_obl_dends']['SegList']),len(makeDict2['basal_obl_dends']['LengthDist']) )
          
          self.assertEqual(set(makeDict2['most_prox_bas_dend']['SegList']),set([16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3])) 
        
      #######################################################################################################################
      def test_get_target_segments(self):
      
          target_segs1, fractions_list1=oc.get_target_segments(seg_specifications={'Seg1_comp_28': {'SegList': [55], 'LengthDist': [25.00030750752478]}, 'Seg1_comp_68': {'SegList': [135], 'LengthDist': [24.99966139974902]}},subset_dict={'Seg1_comp_x':1,'Seg1_comp_y':1})
          
          self.assertEqual(target_segs1,[])
          self.assertEqual(fractions_list1,[])
          
          target_segs1, fractions_list1=oc.get_target_segments(seg_specifications={'Seg1_comp_28': {'SegList': [55], 'LengthDist': [25.00030750752478]}, 'Seg1_comp_68': {'SegList': [135], 'LengthDist': [24.99966139974902]}},subset_dict={'Seg1_comp_28':1,'Seg1_comp_68':1})
          
          self.assertEqual(len(target_segs1),2)
          self.assertEqual(len(fractions_list1),2)
          
          self.assertEqual(set(target_segs1),set([55,135]))
          self.assertEqual(len(np.unique(fractions_list1)),2)
          self.assertTrue( fractions_list1[0] <=1)
          self.assertTrue( fractions_list1[1] <=1)
          
          
          target_segs, fractions_list=oc.get_target_segments(seg_specifications={'basal_obl_dends': {'SegList': [16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 24, 25, 22, 23, 20, 21, 18, 19, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 48, 49, 46, 47, 44, 45, 42, 43, 64, 65, 62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 72, 73, 70, 71, 68, 69, 66, 67], 'LengthDist': [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0000012807632, 249.99999701896448, 274.9999982997277, 299.9999413836009, 324.99994266436414, 349.99992870308233, 374.99992998384556, 399.9999756051272, 424.9999756051272, 449.9999756051272, 474.9999756051272, 499.9999756051272, 524.9999756051272, 549.9999756051272, 574.9999756051272, 599.9999756051272, 624.9999756051272, 649.9999756051272, 674.9999756051272, 699.9999756051272, 724.9999756051272, 749.9999756051272, 774.9999756051272, 799.9999756051272, 824.9999741146091, 849.9995943443294, 874.9996247235468, 899.9999308454469, 924.9999175769843, 950.0002631896537, 975.0002596206808, 1000.0001576712268, 1025.0001576712268, 1050.0001576712268, 1075.0001576712268, 1100.0001576712268, 1125.0001576712268, 1150.0001576712268, 1175.0001576712268, 1200.0001576712268, 1225.0001576712268, 1250.0001576712268, 1275.0001576712268, 1300.0001576712268, 1325.0001576712268, 1350.0001576712268, 1375.0001576712268, 1400.0001576712268, 1425.0010159713997, 1450.0009008583847, 1475.0002938437874, 1500.000601351312, 1525.0002368231228, 1549.9998501247107, 1575.0001714886703, 1599.9998485298981, 1624.9998485298981, 1649.9998485298981, 1674.9998485298981, 1699.9998485298981, 1724.9998485298981, 1749.9998485298981, 1774.9998485298981, 1799.9998485298981]}, 'most_prox_bas_dend': {'SegList': [16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3], 'LengthDist': [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0000012807632, 249.99999701896448, 274.9999982997277, 299.9999413836009, 324.99994266436414, 349.99992870308233, 374.99992998384556, 399.9999756051272]}},
                                                             subset_dict={'basal_obl_dends':100,'most_prox_bas_dend':50})
                                                             
                                                             
          self.assertEqual(len(target_segs),150)
          self.assertEqual(len(fractions_list),150)
          basal_obl_dends_count=0
          most_prox_bas_dend_count=0
          for ind in range(0,len(target_segs)):
              
              if target_segs[ind] in [16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3]:
                 most_prox_bas_dend_count+=1
              if most_prox_bas_dend_count ==50:
                 break
          
          self.assertEqual(most_prox_bas_dend_count,50)
          
#######################################################################################################################################################################
class TestNetConnectionMethods(unittest.TestCase):

      def test_read_connectivity(self):
      
          proj_info=oc_utils.read_connectivity('L23PyrRS','L23PyrFRB','ConnListTest')
          
          
          self.assertTrue(isinstance(proj_info,list))
          
          for proj_ind in range(0,len(proj_info)):
          
              self.assertEqual(len(proj_info[proj_ind].keys() ), len(proj_info[proj_ind].values() ) )
          
              self.assertTrue( 'SynapseList' in proj_info[proj_ind].keys() )
              self.assertTrue( 'PreCellGroup' in proj_info[proj_ind].keys() )
              self.assertTrue( 'PostCellGroup' in proj_info[proj_ind].keys() )
              self.assertTrue( 'Type' in proj_info[proj_ind].keys() )
              self.assertTrue( 'LocOnPostCell' in proj_info[proj_ind].keys() )
              self.assertTrue( 'NumPerPostCell' in proj_info[proj_ind].keys() or 'NumPerPreCell' in proj_info[proj_ind].keys() )
              

      def test_add_chem_projection(self):
          ######## Test 1 convergent
          network = neuroml.Network(id='Net0')     
          presynaptic_population = neuroml.Population(id="Pop0", component="L23PyrRS", type="populationList", size=1)
          postsynaptic_population=neuroml.Population(id="Pop1", component="L23PyrFRB", type="populationList", size=1)
          
          synapse_list=['AMPA','NMDA']
          
          projection_array=[]
          
          for synapse_element in range(0,len(synapse_list) ):
          
              proj = neuroml.Projection(id="Proj%d"%synapse_element, 
                                        presynaptic_population=presynaptic_population.id, 
                                        postsynaptic_population=postsynaptic_population.id, 
                                        synapse=synapse_list[synapse_element])
                                        
              projection_array.append(proj)
              
              
          parsed_target_dict={'basal_obl_dends': {'SegList': [16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 24, 25, 22, 23, 20, 21, 18, 19, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 48, 49, 46, 47, 44, 45, 42, 43, 64, 65, 62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 72, 73, 70, 71, 68, 69, 66, 67], 'LengthDist': [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0000012807632, 249.99999701896448, 274.9999982997277, 299.9999413836009, 324.99994266436414, 349.99992870308233, 374.99992998384556, 399.9999756051272, 424.9999756051272, 449.9999756051272, 474.9999756051272, 499.9999756051272, 524.9999756051272, 549.9999756051272, 574.9999756051272, 599.9999756051272, 624.9999756051272, 649.9999756051272, 674.9999756051272, 699.9999756051272, 724.9999756051272, 749.9999756051272, 774.9999756051272, 799.9999756051272, 824.9999741146091, 849.9995943443294, 874.9996247235468, 899.9999308454469, 924.9999175769843, 950.0002631896537, 975.0002596206808, 1000.0001576712268, 1025.0001576712268, 1050.0001576712268, 1075.0001576712268, 1100.0001576712268, 1125.0001576712268, 1150.0001576712268, 1175.0001576712268, 1200.0001576712268, 1225.0001576712268, 1250.0001576712268, 1275.0001576712268, 1300.0001576712268, 1325.0001576712268, 1350.0001576712268, 1375.0001576712268, 1400.0001576712268, 1425.0010159713997, 1450.0009008583847, 1475.0002938437874, 1500.000601351312, 1525.0002368231228, 1549.9998501247107, 1575.0001714886703, 1599.9998485298981, 1624.9998485298981, 1649.9998485298981, 1674.9998485298981, 1699.9998485298981, 1724.9998485298981, 1749.9998485298981, 1774.9998485298981, 1799.9998485298981]}}
    
              
          proj_array=oc.add_chem_projection(net=network,
                                            proj_array=projection_array,
                                            presynaptic_population=presynaptic_population,
                                            postsynaptic_population=postsynaptic_population,
                                            targeting_mode='convergent',
                                            synapse_list=synapse_list,
                                            seg_target_dict=parsed_target_dict,
                                            subset_dict={'basal_obl_dends':50},
                                            delays_dict={'NMDA':5},
                                            weights_dict={'AMPA':1.5,'NMDA':2})
                                            
          
          self.assertEqual(len(network.projections),2)
          
          self.assertEqual(len(proj_array[0].connection_wds),50)
          
          pre_cell_AMPA_strings=[]
          
          post_cell_AMPA_strings=[]
          
          pre_cell_NMDA_strings=[]
          
          post_cell_NMDA_strings=[]
          
          for conn_ind in range(0,50):
          
              pre_cell_AMPA_strings.append(proj_array[0].connection_wds[conn_ind].pre_cell_id)
             
              post_cell_AMPA_strings.append(proj_array[0].connection_wds[conn_ind].post_cell_id)
             
              pre_cell_NMDA_strings.append(proj_array[1].connection_wds[conn_ind].pre_cell_id)
             
              post_cell_NMDA_strings.append(proj_array[1].connection_wds[conn_ind].post_cell_id)
              
          
          self.assertEqual(len(set(pre_cell_AMPA_strings)),1)
          
          self.assertEqual(len(set(pre_cell_AMPA_strings)),len(set(pre_cell_NMDA_strings)) )
          
          self.assertEqual(len(set(post_cell_AMPA_strings)),1)
          
          self.assertEqual(len(set(post_cell_AMPA_strings)),len(set(post_cell_NMDA_strings)) )
          
          self.assertEqual(len(proj_array[0].connection_wds),len(proj_array[1].connection_wds) )
          
          self.assertEqual(proj_array[0].synapse,'AMPA')
          
          self.assertEqual(proj_array[1].synapse,'NMDA')
              
          
          ######## Test 2 convergent
          network = neuroml.Network(id='Net0')     
          presynaptic_population = neuroml.Population(id="Pop0", component="L23PyrRS", type="populationList", size=1)
          postsynaptic_population=neuroml.Population(id="Pop0", component="L23PyrRS", type="populationList", size=1)
          
          synapse_list=['AMPA','NMDA']
          
          projection_array=[]
          
          for synapse_element in range(0,len(synapse_list) ):
          
              proj = neuroml.Projection(id="Proj%d"%synapse_element, 
                                        presynaptic_population=presynaptic_population.id, 
                                        postsynaptic_population=postsynaptic_population.id, 
                                        synapse=synapse_list[synapse_element])
                                        
              projection_array.append(proj)
              
              
          parsed_target_dict={'basal_obl_dends': {'SegList': [16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 24, 25, 22, 23, 20, 21, 18, 19, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 48, 49, 46, 47, 44, 45, 42, 43, 64, 65, 62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 72, 73, 70, 71, 68, 69, 66, 67], 'LengthDist': [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0000012807632, 249.99999701896448, 274.9999982997277, 299.9999413836009, 324.99994266436414, 349.99992870308233, 374.99992998384556, 399.9999756051272, 424.9999756051272, 449.9999756051272, 474.9999756051272, 499.9999756051272, 524.9999756051272, 549.9999756051272, 574.9999756051272, 599.9999756051272, 624.9999756051272, 649.9999756051272, 674.9999756051272, 699.9999756051272, 724.9999756051272, 749.9999756051272, 774.9999756051272, 799.9999756051272, 824.9999741146091, 849.9995943443294, 874.9996247235468, 899.9999308454469, 924.9999175769843, 950.0002631896537, 975.0002596206808, 1000.0001576712268, 1025.0001576712268, 1050.0001576712268, 1075.0001576712268, 1100.0001576712268, 1125.0001576712268, 1150.0001576712268, 1175.0001576712268, 1200.0001576712268, 1225.0001576712268, 1250.0001576712268, 1275.0001576712268, 1300.0001576712268, 1325.0001576712268, 1350.0001576712268, 1375.0001576712268, 1400.0001576712268, 1425.0010159713997, 1450.0009008583847, 1475.0002938437874, 1500.000601351312, 1525.0002368231228, 1549.9998501247107, 1575.0001714886703, 1599.9998485298981, 1624.9998485298981, 1649.9998485298981, 1674.9998485298981, 1699.9998485298981, 1724.9998485298981, 1749.9998485298981, 1774.9998485298981, 1799.9998485298981]}}
    
              
          proj_array=oc.add_chem_projection(net=network,
                                            proj_array=projection_array,
                                            presynaptic_population=presynaptic_population,
                                            postsynaptic_population=postsynaptic_population,
                                            targeting_mode='convergent',
                                            synapse_list=synapse_list,
                                            seg_target_dict=parsed_target_dict,
                                            subset_dict={'basal_obl_dends':50},
                                            delays_dict={'NMDA':5},
                                            weights_dict={'AMPA':1.5,'NMDA':2})
                                            
          
          
          self.assertEqual(len(proj_array[0].connection_wds),0)
          
          self.assertEqual(len(proj_array[0].connection_wds),len(proj_array[1].connection_wds) )
          
          self.assertEqual(proj_array[0].synapse,'AMPA')
          
          self.assertEqual(proj_array[1].synapse,'NMDA')
          
          self.assertEqual(len(network.projections),0)
          
          ######## Test 3 convergent
          network = neuroml.Network(id='Net0')     
          presynaptic_population = neuroml.Population(id="Pop0", component="L23PyrRS", type="populationList", size=50)
          postsynaptic_population=neuroml.Population(id="Pop1", component="L23PyrFRB", type="populationList", size=2)
          
          synapse_list=['AMPA','NMDA']
          
          projection_array=[]
          
          for synapse_element in range(0,len(synapse_list) ):
          
              proj = neuroml.Projection(id="Proj%d"%synapse_element, 
                                        presynaptic_population=presynaptic_population.id, 
                                        postsynaptic_population=postsynaptic_population.id, 
                                        synapse=synapse_list[synapse_element])
                                        
              projection_array.append(proj)
              
          parsed_target_dict={'basal_obl_dends': {'SegList': [16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 24, 25, 22, 23, 20, 21, 18, 19, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 48, 49, 46, 47, 44, 45, 42, 43, 64, 65, 62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 72, 73, 70, 71, 68, 69, 66, 67], 'LengthDist': [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0000012807632, 249.99999701896448, 274.9999982997277, 299.9999413836009, 324.99994266436414, 349.99992870308233, 374.99992998384556, 399.9999756051272, 424.9999756051272, 449.9999756051272, 474.9999756051272, 499.9999756051272, 524.9999756051272, 549.9999756051272, 574.9999756051272, 599.9999756051272, 624.9999756051272, 649.9999756051272, 674.9999756051272, 699.9999756051272, 724.9999756051272, 749.9999756051272, 774.9999756051272, 799.9999756051272, 824.9999741146091, 849.9995943443294, 874.9996247235468, 899.9999308454469, 924.9999175769843, 950.0002631896537, 975.0002596206808, 1000.0001576712268, 1025.0001576712268, 1050.0001576712268, 1075.0001576712268, 1100.0001576712268, 1125.0001576712268, 1150.0001576712268, 1175.0001576712268, 1200.0001576712268, 1225.0001576712268, 1250.0001576712268, 1275.0001576712268, 1300.0001576712268, 1325.0001576712268, 1350.0001576712268, 1375.0001576712268, 1400.0001576712268, 1425.0010159713997, 1450.0009008583847, 1475.0002938437874, 1500.000601351312, 1525.0002368231228, 1549.9998501247107, 1575.0001714886703, 1599.9998485298981, 1624.9998485298981, 1649.9998485298981, 1674.9998485298981, 1699.9998485298981, 1724.9998485298981, 1749.9998485298981, 1774.9998485298981, 1799.9998485298981]}}
    
              
          proj_array=oc.add_chem_projection(net=network,
                                            proj_array=projection_array,
                                            presynaptic_population=presynaptic_population,
                                            postsynaptic_population=postsynaptic_population,
                                            targeting_mode='convergent',
                                            synapse_list=synapse_list,
                                            seg_target_dict=parsed_target_dict,
                                            subset_dict={'basal_obl_dends':50},
                                            delays_dict={'NMDA':5},
                                            weights_dict={'AMPA':1.5,'NMDA':2})
                                            
          
          self.assertEqual(len(network.projections),2)
          
          self.assertEqual(len(proj_array[0].connection_wds),100)
          
          self.assertEqual(len(proj_array[0].connection_wds),len(proj_array[1].connection_wds) )
          
          self.assertEqual(proj_array[0].synapse,'AMPA')
          
          self.assertEqual(proj_array[1].synapse,'NMDA')
          
          pre_cell_AMPA_strings=[]
          
          post_cell_AMPA_strings=[]
          
          pre_cell_NMDA_strings=[]
          
          post_cell_NMDA_strings=[]
          
          for conn_ind in range(0,100):
          
              pre_cell_AMPA_strings.append(proj_array[0].connection_wds[conn_ind].pre_cell_id)
             
              post_cell_AMPA_strings.append(proj_array[0].connection_wds[conn_ind].post_cell_id)
             
              pre_cell_NMDA_strings.append(proj_array[1].connection_wds[conn_ind].pre_cell_id)
             
              post_cell_NMDA_strings.append(proj_array[1].connection_wds[conn_ind].post_cell_id)
          
              self.assertEqual(proj_array[0].connection_wds[conn_ind].pre_cell_id,proj_array[1].connection_wds[conn_ind].pre_cell_id)
             
              self.assertEqual(proj_array[0].connection_wds[conn_ind].post_cell_id,proj_array[1].connection_wds[conn_ind].post_cell_id)
             
              self.assertEqual(proj_array[0].connection_wds[conn_ind].post_segment_id,proj_array[1].connection_wds[conn_ind].post_segment_id)
             
              self.assertEqual(proj_array[0].connection_wds[conn_ind].pre_segment_id,proj_array[1].connection_wds[conn_ind].pre_segment_id)
             
              self.assertEqual(proj_array[0].connection_wds[conn_ind].pre_fraction_along,proj_array[1].connection_wds[conn_ind].pre_fraction_along)
             
              self.assertEqual(proj_array[0].connection_wds[conn_ind].post_fraction_along,proj_array[1].connection_wds[conn_ind].post_fraction_along)
             
              self.assertNotEqual(proj_array[0].connection_wds[conn_ind].delay,proj_array[1].connection_wds[conn_ind].delay)
             
              self.assertNotEqual(proj_array[0].connection_wds[conn_ind].weight,proj_array[1].connection_wds[conn_ind].weight)
             
          self.assertEqual(len(set(pre_cell_AMPA_strings)),50)
          
          self.assertEqual(len(set(pre_cell_AMPA_strings)),len(set(pre_cell_NMDA_strings)) )
          
          self.assertEqual(len(set(post_cell_AMPA_strings)),2)
          
          self.assertEqual(len(set(post_cell_AMPA_strings)),len(set(post_cell_NMDA_strings)) )
          
          ######## Test 4 divergent
          network = neuroml.Network(id='Net0')     
          presynaptic_population = neuroml.Population(id="Pop0", component="L23PyrRS", type="populationList", size=3)
          postsynaptic_population=neuroml.Population(id="Pop1", component="L23PyrFRB", type="populationList", size=50)
          
          synapse_list=['AMPA','NMDA']
          
          projection_array=[]
          
          for synapse_element in range(0,len(synapse_list) ):
          
              proj = neuroml.Projection(id="Proj%d"%synapse_element, 
                                        presynaptic_population=presynaptic_population.id, 
                                        postsynaptic_population=postsynaptic_population.id, 
                                        synapse=synapse_list[synapse_element])
                                        
              projection_array.append(proj)
              
          parsed_target_dict={'basal_obl_dends': {'SegList': [16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 24, 25, 22, 23, 20, 21, 18, 19, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 48, 49, 46, 47, 44, 45, 42, 43, 64, 65, 62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 72, 73, 70, 71, 68, 69, 66, 67], 'LengthDist': [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0000012807632, 249.99999701896448, 274.9999982997277, 299.9999413836009, 324.99994266436414, 349.99992870308233, 374.99992998384556, 399.9999756051272, 424.9999756051272, 449.9999756051272, 474.9999756051272, 499.9999756051272, 524.9999756051272, 549.9999756051272, 574.9999756051272, 599.9999756051272, 624.9999756051272, 649.9999756051272, 674.9999756051272, 699.9999756051272, 724.9999756051272, 749.9999756051272, 774.9999756051272, 799.9999756051272, 824.9999741146091, 849.9995943443294, 874.9996247235468, 899.9999308454469, 924.9999175769843, 950.0002631896537, 975.0002596206808, 1000.0001576712268, 1025.0001576712268, 1050.0001576712268, 1075.0001576712268, 1100.0001576712268, 1125.0001576712268, 1150.0001576712268, 1175.0001576712268, 1200.0001576712268, 1225.0001576712268, 1250.0001576712268, 1275.0001576712268, 1300.0001576712268, 1325.0001576712268, 1350.0001576712268, 1375.0001576712268, 1400.0001576712268, 1425.0010159713997, 1450.0009008583847, 1475.0002938437874, 1500.000601351312, 1525.0002368231228, 1549.9998501247107, 1575.0001714886703, 1599.9998485298981, 1624.9998485298981, 1649.9998485298981, 1674.9998485298981, 1699.9998485298981, 1724.9998485298981, 1749.9998485298981, 1774.9998485298981, 1799.9998485298981]}}
         
          proj_array=oc.add_chem_projection(net=network,
                                            proj_array=projection_array,
                                            presynaptic_population=presynaptic_population,
                                            postsynaptic_population=postsynaptic_population,
                                            targeting_mode='divergent',
                                            synapse_list=synapse_list,
                                            seg_target_dict=parsed_target_dict,
                                            subset_dict={'basal_obl_dends':50},
                                            delays_dict={'NMDA':5},
                                            weights_dict={'AMPA':1.5,'NMDA':2})
                                            
          self.assertEqual(len(network.projections),2)
          
          self.assertEqual(len(proj_array[0].connection_wds),150)
          
          self.assertEqual(len(proj_array[0].connection_wds),len(proj_array[1].connection_wds) )
          
          self.assertEqual(proj_array[0].synapse,'AMPA')
          
          self.assertEqual(proj_array[1].synapse,'NMDA')
          
          pre_cell_AMPA_strings=[]
          
          post_cell_AMPA_strings=[]
          
          pre_cell_NMDA_strings=[]
          
          post_cell_NMDA_strings=[]
          
          for conn_ind in range(0,150):
          
              pre_cell_AMPA_strings.append(proj_array[0].connection_wds[conn_ind].pre_cell_id)
             
              post_cell_AMPA_strings.append(proj_array[0].connection_wds[conn_ind].post_cell_id)
             
              pre_cell_NMDA_strings.append(proj_array[1].connection_wds[conn_ind].pre_cell_id)
             
              post_cell_NMDA_strings.append(proj_array[1].connection_wds[conn_ind].post_cell_id)
          
              self.assertEqual(proj_array[0].connection_wds[conn_ind].pre_cell_id,proj_array[1].connection_wds[conn_ind].pre_cell_id)
             
              self.assertEqual(proj_array[0].connection_wds[conn_ind].post_cell_id,proj_array[1].connection_wds[conn_ind].post_cell_id)
             
              self.assertEqual(proj_array[0].connection_wds[conn_ind].post_segment_id,proj_array[1].connection_wds[conn_ind].post_segment_id)
             
              self.assertEqual(proj_array[0].connection_wds[conn_ind].pre_segment_id,proj_array[1].connection_wds[conn_ind].pre_segment_id)
             
              self.assertEqual(proj_array[0].connection_wds[conn_ind].pre_fraction_along,proj_array[1].connection_wds[conn_ind].pre_fraction_along)
             
              self.assertEqual(proj_array[0].connection_wds[conn_ind].post_fraction_along,proj_array[1].connection_wds[conn_ind].post_fraction_along)
             
              self.assertNotEqual(proj_array[0].connection_wds[conn_ind].delay,proj_array[1].connection_wds[conn_ind].delay)
             
              self.assertNotEqual(proj_array[0].connection_wds[conn_ind].weight,proj_array[1].connection_wds[conn_ind].weight)
             
          self.assertEqual(len(set(pre_cell_AMPA_strings)),3)
          
          self.assertEqual(len(set(pre_cell_AMPA_strings)),len(set(pre_cell_NMDA_strings)) )
          
          self.assertEqual(len(set(post_cell_AMPA_strings)),50)
          
          self.assertEqual(len(set(post_cell_AMPA_strings)),len(set(post_cell_NMDA_strings)) )
          
          
          
          
          
          
          
          
          
          
          
             
          
             
             
             
             
             
             
             
             
          
     
          
          
          
          
          
          
          
          
        
          
          
          
         

