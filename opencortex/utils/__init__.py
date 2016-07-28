#####################
### Subject to change without notice!!
#####################
##############################################################
### Author : Rokas Stanislovas
###
### GSoC 2016 project: Cortical Networks
###
##############################################################

import opencortex
import neuroml
import pyneuroml
import pyneuroml.lems

import neuroml.writers as writers
import neuroml.loaders as loaders

from pyneuroml import pynml
from pyneuroml.lems.LEMSSimulation import LEMSSimulation

import random
import sys
import os
import shutil
import numpy as np
import json
import math
import operator


import opencortex.build as oc_build


################################################################################    
    
def add_populations_in_layers(net,boundaryDict,popDict,x_vector,z_vector,storeSoma=False): 

   '''This method distributes the cells in rectangular layers. The input arguments:
   
   net - libNeuroML network object;
                    
   popDict - a dictionary whose keys are unique cell population ids; each key entry stores a tuple of three elements: population size, Layer tag and cell model id; 
   
   layer tags (of type string) must make up the keys() of boundaryDict;
    
   boundaryDict have layer pointers as keys; each entry stores the left and right bound of the layer in the list format , e.g. [L3_min, L3_max]
   
   x_vector - a vector that stores the left and right bounds of the cortical column along x dimension
   
   y_vector - a vector that stores the left and right bounds of the cortical column along y dimension
   
   storeSoma - specifies whether soma positions have to be stored in the output array (default is set to False).
   
   This method returns the dictionary; each key is a unique cell population id and the corresponding value is a dictionary
   which refers to libNeuroML population object (key 'PopObj') and cell position array ('Positions') which by default is None.'''
    
   return_pops={}
   
   for cell_pop in popDict.keys():

       size, layer,cell_model = popDict[cell_pop]
    
       if size>0:
          return_pops[cell_pop]={}
          xl=x_vector[1]-x_vector[0]
          yl=boundaryDict[layer][1]-boundaryDict[layer][0]
          zl=z_vector[1]-z_vector[0]
          
          if storeSoma:
          
             pop, cellPositions=oc_build.add_population_in_rectangular_region(net,cell_pop,cell_model,size,x_vector[0],boundaryDict[layer][0],z_vector[0],xl,yl,zl,storeSoma)
             
          else:
             pop=oc_build.add_population_in_rectangular_region(net,cell_pop,cell_model,size,x_vector[0],boundaryDict[layer][0],z_vector[0],xl,yl,zl,storeSoma)
             
             cellPositions=None
         
          return_pops[cell_pop]={}
          return_pops[cell_pop]['PopObj']=pop
          return_pops[cell_pop]['Positions']=cellPositions
   
   opencortex.print_comment_v("This is a final list of cell population ids: %s"%return_pops.keys())
   
   return return_pops
          

############################################################################################################################################################################

def build_projection(net, 
                     proj_counter,
                     proj_type,
                     presynaptic_population, 
                     postsynaptic_population, 
                     synapse_list,  
                     targeting_mode,
                     seg_length_dict,
                     num_of_conn_dict,
                     distance_dependent_rule=None,
                     pre_cell_positions=None,
                     post_cell_positions=None,
                     delays_dict=None,
                     weights_dict=None):
                         
    
    '''This method calls the appropriate methods that construct chemical or electrical projections. The input arguments are as follows:
   
    net - the network object created using libNeuroML ( neuroml.Network() );
    
    proj_counter - stores the number of projections at any given moment;
    
    proj_type - 'Chem' or 'Elect' depending on whether the projection is chemical or electrical.
    
    presynaptic_population - object corresponding to the presynaptic population in the network;
    
    postsynaptic_population - object corresponding to the postsynaptic population in the network;
    
    synapse_list - the list of synapse ids that correspond to the individual receptor components on the physical synapse, e.g. the first element is
    the id of the AMPA synapse and the second element is the id of the NMDA synapse; these synapse components will be mapped onto the same location of the target segment;
    
    targeting_mode - specifies the type of projection: divergent or convergent;
    
    seg_length_dict - a dictionary whose keys are the ids of target segment groups and the values are dictionaries in the format returned by make_target_dict();
    
    num_of_conn_dict - a dictionary whose keys are the ids of target segment groups with the corresponding values of type 'int' specifying the number of connections per 
    tarrget segment group per each cell.
    
    distance_dependent_rule - optional string which defines the distance dependent rule of connectivity - soma to soma distance must be represented by the string character 'r';
    
    pre_cell_positions- optional array specifying the cell positions for the presynaptic population; the format is an array of [ x coordinate, y coordinate, z coordinate];
    
    post_cell_positions- optional array specifying the cell positions for the postsynaptic population; the format is an array of [ x coordinate, y coordinate, z coordinate];
    
    delays_dict - optional dictionary that specifies the delays (in ms) for individual synapse components, e.g. {'NMDA':5.0} or {'AMPA':3.0,'NMDA':5};
    
    weights_dict - optional dictionary that specifies the weights (in ms) for individual synapse components, e.g. {'NMDA':1} or {'NMDA':1,'AMPA':2}.'''  
               
    
    if presynaptic_population.size==0 or postsynaptic_population.size==0:
       return None
    
    proj_array=[]
    syn_counter=0
    
    synapse_list=list(set(synapse_list))
    
    for synapse_id in synapse_list:
       
        if proj_type=='Elect':
        
           proj = neuroml.ElectricalProjection(id="Proj%dsyn%d_%s_%s"%(proj_counter,syn_counter,presynaptic_population.id, postsynaptic_population.id),
                                               presynaptic_population=presynaptic_population.id,
                                               postsynaptic_population=postsynaptic_population.id)
                                               
           syn_counter+=1
           proj_array.append(proj)
           
        if proj_type=='Chem':
                                            
           proj = neuroml.Projection(id="Proj%dsyn%d_%s_%s"%(proj_counter,syn_counter,presynaptic_population.id, postsynaptic_population.id), 
                                     presynaptic_population=presynaptic_population.id, 
                                     postsynaptic_population=postsynaptic_population.id, 
                                     synapse=synapse_id)
                                     
           syn_counter+=1              
           proj_array.append(proj)
 
       
    if distance_dependent_rule==None:
       
       if proj_type=='Chem':
          proj_array              =oc_build.add_chem_projection(net,
                                                                proj_array,
                                                                presynaptic_population,
                                                                postsynaptic_population,
                                                                targeting_mode,
                                                                synapse_list,
                                                                seg_length_dict,
                                                                num_of_conn_dict,
                                                                delays_dict,
                                                                weights_dict) 
                                                                
       if proj_type=='Elect':
          proj_array              =oc_build.add_elect_projection(net,
                                                                 proj_array,
                                                                 presynaptic_population,
                                                                 postsynaptic_population,
                                                                 targeting_mode,
                                                                 synapse_list,
                                                                 seg_length_dict,
                                                                 num_of_conn_dict)
    else:
    
       if proj_type=='Chem':
          proj_array              =oc_build.add_chem_spatial_projection(net,
                                                                        proj_array,
                                                                        presynaptic_population,
                                                                        postsynaptic_population,
                                                                        targeting_mode,
                                                                        synapse_list,
                                                                        seg_length_dict,
                                                                        num_of_conn_dict,
                                                                        distance_dependent_rule,
                                                                        pre_cell_positions,
                                                                        post_cell_positions,
                                                                        delays_dict,
                                                                        weights_dict)
                                                                        
       if proj_type=='Elect':
          proj_array              =oc_build.add_elect_spatial_projection(net,
                                                                         proj_array,
                                                                         presynaptic_population,
                                                                         postsynaptic_population,
                                                                         targeting_mode,
                                                                         synapse_list,
                                                                         seg_length_dict,
                                                                         num_of_conn_dict,
                                                                         distance_dependent_rule,
                                                                         pre_cell_positions,
                                                                         post_cell_positions)
       
          
    

    return proj_array, proj_counter



#######################################################################################################################################

def build_connectivity(net,
                       pop_objects,
                       path_to_cells,
                       full_path_to_conn_summary,
                       return_cached_dicts=True,
                       synaptic_scaling_params=None,
                       synaptic_delay_params=None,
                       distance_dependence_params=None):

    final_synapse_list=[]
    
    final_proj_array=[]
    
    cached_target_dict={}
    
    proj_counter=0
    
    for prePop in pop_objects.keys():
        
        preCellObject=pop_objects[prePop]
            
        for postPop in pop_objects.keys():
                
            postCellObject=pop_objects[postPop]
                    
            if preCellObject['PopObj'].size !=0 and postCellObject['PopObj'].size !=0:
                       
               proj_summary=read_connectivity(prePop,postPop,full_path_to_conn_summary)
               
               if proj_summary !=[]:
                       
                  for proj_ind in range(0,len(proj_summary)):
                          
                      projInfo=proj_summary[proj_ind]
                          
                      target_comp_groups=projInfo['LocOnPostCell']
                      
                      synapseList=projInfo['SynapseList']   
                                                                           
                      final_synapse_list.extend(projInfo['SynapseList'])
                           
                      if 'NumPerPostCell' in projInfo:
                    
                         targetingMode='convergent'
                       
                         mode_string='NumPerPostCell'
                              
                      if 'NumPerPreCell' in projInfo:
                    
                         targetingMode='divergent'
                       
                         mode_string='NumPerPreCell'
                           
                      if synaptic_scaling_params != None:
                      
                         weights=parse_weights(synaptic_scaling_params,postPop,synapseList)
                         
                      else:
                         
                         weights=None
                         
                      if synaptic_delay_params != None:
                      
                         delays=parse_delays(synaptic_delay_params,postPop,synapseList)
                         
                      else:
                        
                         delays=None
                          
                      if distance_dependence_params != None:
                      
                         dist_par=parse_distance_dependent_rule(distance_dependence_params,prePop,postPop)
                         
                      else:
                      
                         dist_par=None
                         
                      ### assumes one target segment group per given projection in the format of netConnLists   ### 
                      subset_dict={}
                                    
                      subset_dict[target_comp_groups]=float(projInfo[mode_string])
                         
                      target_comp_groups=[target_comp_groups]
                      #############################################################################################
                      cell_component=postCellObject['PopObj'].component
                                 
                      if cell_component not in cached_target_dict.keys():
                           
                         cell_nml_file = '%s.cell.nml'%cell_component
                           
                         if path_to_cells != None:
                       
                            document_cell = neuroml.loaders.NeuroMLLoader.load(os.path.join(path_to_cells,cell_nml_file))
                          
                         else:
                       
                            document_cell=neuroml.loaders.NeuroMLLoader.load(cell_nml_file)
                              
                         cellObject=document_cell.cells[0]
                              
                         target_segments=oc_build.extract_seg_ids(cell_object=cellObject,target_compartment_array=target_comp_groups,targeting_mode='segGroups')
                              
                         segLengthDict=oc_build.make_target_dict(cell_object=cellObject,target_segs=target_segments) 
                                 
                         cached_target_dict[cell_component]={}
                                 
                         cached_target_dict[cell_component]['CellObject']=cellObject
                              
                         cached_target_dict[cell_component]['TargetDict']=segLengthDict
                              
                      else:
                      
                         target_groups_to_include=[]
                         
                         new_segment_groups=False
                         
                         segLengthDict={}
                      
                         for target_group in target_comp_groups:
                              
                             if target_group not in cached_target_dict[cell_component]['TargetDict'].keys():
                             
                                target_groups_to_include.append(target_group)
                                
                                new_segment_groups=True
                                
                             else:
                             
                                segLengthDict[target_group]=cached_target_dict[cell_component]['TargetDict'][target_group]
                                
                         if new_segment_groups:
                                 
                            cellObject=cached_target_dict[cell_component]['CellObject']
                                 
                            target_segments=oc_build.extract_seg_ids(cell_object=cellObject,target_compartment_array=target_groups_to_include,targeting_mode='segGroups') 
                                 
                            new_seg_length_dict=oc_build.make_target_dict(cell_object=cellObject,target_segs=target_segments) 
                              
                            for new_target_group in new_seg_length_dict.keys():
                            
                                cached_target_dict[cell_component]['TargetDict'][new_target_group]=new_seg_length_dict[new_target_group]
                                
                                segLengthDict[new_target_group]=new_seg_length_dict[new_target_group]
                           
                      compound_proj              =build_projection(net=net, 
                                                                   proj_counter=proj_counter,
                                                                   proj_type=projInfo['Type'],
                                                                   presynaptic_population=preCellObject['PopObj'], 
                                                                   postsynaptic_population=postCellObject['PopObj'], 
                                                                   synapse_list=synapseList,  
                                                                   targeting_mode=targetingMode,
                                                                   seg_length_dict=segLengthDict,
                                                                   num_of_conn_dict=subset_dict,
                                                                   distance_dependent_rule=dist_par,
                                                                   pre_cell_positions=preCellObject['Positions'],
                                                                   post_cell_positions=postCellObject['Positions'],
                                                                   delays_dict=delays,
                                                                   weights_dict=weights)
                                                        
                                                        
                      proj_counter+=1                      
                      final_proj_array.extend(compound_proj)
                              
                              
    final_synapse_list=np.unique(final_synapse_list)
    
    final_synapse_list=list(final_synapse_list)
    
    if return_cached_dicts:
                          
       return final_synapse_list, final_proj_array, cached_target_dict
       
    else:
    
       return final_synapse_list, final_proj_array
    
    
################################################################################################################################################################

def read_connectivity(pre_pop,
                      post_pop,
                      path_to_txt_file):
                      
    '''Method that reads the txt file in the format of netConnList found in: Thalamocortical/neuroConstruct/pythonScripts/netbuild.'''                   

    proj_summary=[]

    with open(path_to_txt_file, 'r') as file:
    
         lines = file.readlines()
         
    num_per_post_cell=False
    
    num_per_pre_cell=False
         
    for line in lines:
    
        if 'NumPerPostCell' in line:
        
           num_per_post_cell=True
           break
           
        if 'NumPerPreCell' in line:
           
           num_per_pre_cell=True
           break
         
    for line in lines:
       
        split_line=line.split(" ")
        
        extract_info=[]
        
        for string_index in range(0,len(split_line)):
        
            if split_line[string_index]!='':
            
               extract_info.append(split_line[string_index])
        
        counter=0    
          
        if pre_pop in extract_info[0] and post_pop in extract_info[1]:
           
           proj_info={}
           
           proj_info['PreCellGroup']=pre_pop
           
           proj_info['PostCellGroup']=post_pop
           
           synapse_list=[]
           
           if ',' in extract_info[2]:
               synapse_list_string=extract_info[2].split(',')
           else:
               synapse_list_string=[extract_info[2]]
              
           for synapse_string in synapse_list_string:
            
               if '[' and ']' in synapse_string:
               
                  left=synapse_string.find('[')
                  
                  right=synapse_string.find(']')
                  
                  synapse=synapse_string[left+1:right]
                  
                  synapse_list.append(synapse)
                  
                  continue
                  
               if '[' in synapse_string:
               
                  left=synapse_string.find('[')
                  
                  synapse=synapse_string[left+1:]
                  
                  synapse_list.append(synapse)
                  
                  continue
                  
               if ']' in synapse_string:
               
                  right=synapse_string.find(']')
                  
                  synapse=synapse_string[0:right]
                  
                  synapse_list.append(synapse)
                                  
                  continue  
           
           proj_info['SynapseList']=synapse_list
             
           if 'Elect' in extract_info[2]:
              proj_info['Type']='Elect'       
           else:
              proj_info['Type']='Chem'
              
           if num_per_post_cell:
              
              proj_info['NumPerPostCell']=extract_info[3]
             
           if num_per_pre_cell:
           
              proj_info['NumPerPreCell']=extract_info[3]
           
           if '\n' in extract_info[4]:
           
              proj_info['LocOnPostCell']=extract_info[4][0:-1]
              
           else:
           
              proj_info['LocOnPostCell']=extract_info[4]
                
           proj_summary.append(proj_info)   
           
       
    return proj_summary
        
################################################################################################################################################################   
def check_cached_dicts(cell_component,cached_dicts,input_group_params,path_to_nml2=None):

    segLengthDict={}
               
    if cell_component in cached_dicts.keys():
            
       target_groups_to_include=[]
                         
       new_segment_groups=False
                     
       segLengthDict={}
                      
       for target_group in input_group_params['TargetDict'].keys():
                              
           if target_group not in cached_dicts[cell_component]['TargetDict'].keys():
                             
               target_groups_to_include.append(target_group)
                                
               new_segment_groups=True
                                
           else:
                             
               segLengthDict[target_group]=cached_dicts[cell_component]['TargetDict'][target_group]
                                
           if new_segment_groups:
                                 
              cellObject=cached_dicts[cell_component]['CellObject']
                                 
              target_segments=oc_build.extract_seg_ids(cell_object=cellObject,target_compartment_array=target_groups_to_include,targeting_mode='segGroups') 
                                 
              new_seg_length_dict=oc_build.make_target_dict(cell_object=cellObject,target_segs=target_segments) 
                              
              for new_target_group in new_seg_length_dict.keys():
                            
                  cached_dicts[cell_component]['TargetDict'][new_target_group]=new_seg_length_dict[new_target_group]
                                
                  segLengthDict[new_target_group]=new_seg_length_dict[new_target_group]
                            
    else:
               
       cell_nml_file = '%s.cell.nml'%pop.component
                           
       if path_to_nml2 != None:
                       
          document_cell = neuroml.loaders.NeuroMLLoader.load(os.path.join(path_to_nml2,cell_nml_file))
                          
       else:
                       
          document_cell=neuroml.loaders.NeuroMLLoader.load(cell_nml_file)
                              
       cellObject=document_cell.cells[0]
               
       target_segments=oc_build.extract_seg_ids(cell_object=cellObject,target_compartment_array=input_group_params['TargetDict'].keys(),targeting_mode='segGroups')
                              
       segLengthDict=oc_build.make_target_dict(cell_object=cellObject,target_segs=target_segments) 
                  
       cached_dicts[cell_component]={}
                                 
       cached_dicts[cell_component]['CellObject']=cellObject
                              
       cached_dicts[cell_component]['TargetDict']=segLengthDict
                  
    return segLengthDict, cached_dicts
##################################################################################################################################################################
def build_inputs(nml_doc,net,population_params,input_params,cached_dicts=None,path_to_cells=None,path_to_synapses=None):     

    '''
    a wrapper method that calls appropriate methods to build inputs to the NeuroML2 network. Input arguments:
    
    nml_doc - a libNeuroML doc object;
    
    net - a libNeuroML net object;
    
    population_params - a dictionary that stores population parameters in the format returned by the method add_populations_in_layers;
    
    input_params -a dictionary that specifies input parameters for any given cell model. The format can be checked by the method check_inputs. Dictionary values must
    
    be of type 'list' and can thus define multiple input groups on any given cell type. Examples where lists contain only one input group but differ in other parameters:
    
    Example 1: input_params={'TCR':[{'InputType':'GeneratePoissonTrains',
                  'InputName':'TransPoiInputs',
                  'TrainType':'transient',
                  'Synapse':'Syn_AMPA_L6NT_TCR',
                  'AverageRateList':[0.05],
                  'DurationList':[200.0],
                  'DelayList':[20.0],
                  'FractionToTarget':1.0,
                  'LocationSpecific':False,
                  'TargetRegions':[{'XVector':[2,12],'YVector':[3,5],'ZVector':[0,5]}],
                  'TargetDict':{'dendrite_group':1000 }       }]              }     
                  
    Example 2: input_params={'TCR':[{'InputType':'PulseGenerators',
                     'InputName':'PulseGenerator0',
                     'Noise':False,
                     'AmplitudeList':[20.0,-20.0],
                     'DurationList':[100.0,50.0],
                     'DelayList':[50.0,200.0],
                     'FractionToTarget':1.0,
                     'LocationSpecific':False,
                     'TargetDict':{'dendrite_group':2 }       }]              }   
                     
    Example 3: input_params={'CG3D_L23PyrRS':[{'InputType':'PulseGenerators',
                             'InputName':'PulseGenerator0',
                             'Noise':True,
                             'SmallestAmplitudeList':[5.0E-5],
                             'LargestAmplitudeList':[1.0E-4],
                             'DurationList':[20000.0],
                             'DelayList':[0.0],
                             'TimeUnits':'ms',
                             'AmplitudeUnits':'uA',
                             'FractionToTarget':1.0,
                             'LocationSpecific':False,
                             'UniqueTargetSegmentID':0,
                             'UniqueFractionAlong':0.5  }]             }                            ;
    
    path_to_nml2 - dir where NeuroML2 files are found.                                   '''
    
    passed_inputs=check_inputs(input_params,population_params,path_to_cells,path_to_synapses)
    
    if passed_inputs:
    
       opencortex.print_comment_v("Input parameters were specified correctly.")
       
    else:
    
      opencortex.print_comment_v("Input parameters were specified incorrectly; execution will terminate.")
      
      quit()
                                   
    input_list_array_final=[]        
                        
    input_synapse_list=[]                    
                        
    for cell_population in input_params.keys():
    
        pop=population_params[cell_population]['PopObj']
        
        cell_component=pop.component
    
        for input_group_ind in range(0,len(input_params[cell_population])):
        
            input_group_params=input_params[cell_population][input_group_ind]
        
            input_group_tag=input_group_params['InputName']
            
            popID=cell_population
            
            cell_positions=population_params[cell_population]['Positions']
            
            pop_size=pop.size
            
            fraction_to_target=input_group_params['FractionToTarget']
            
            if not input_group_params['LocationSpecific']:
             
               target_cell_ids=oc_build.get_target_cells(pop_size,fraction_to_target)
               
            else:
            
               list_of_regions=input_group_params['TargetRegions']
            
               target_cell_ids=oc_build.get_target_cells(pop_size,fraction_to_target,cell_positions, list_of_regions)
               
            if target_cell_ids !=[]:
            
               input_ids_final=[]
               
               condition1='TargetDict' in input_group_params.keys()
               
               condition2='UniversalTargetSegmentID' not in input_group_params.keys()
               
               condition3='UniversalFractionAlong' not in input_group_params.keys()
               
               if condition1 and condition2 and condition3:
               
                  subset_dict=input_group_params['TargetDict']
                  
                  target_segment=None
                  
                  fraction_along=None
               
                  if cached_dicts !=None:
            
                     segLengthDict, cached_dicts =check_cached_dicts(cell_component,cached_dicts,input_group_params,path_to_nml2=path_to_cells)
              
                  else:
            
                     target_segments=oc_build.extract_seg_ids(cell_object=cellObject,target_compartment_array=input_group_params['TargetDict'].keys(),targeting_mode='segGroups')
                              
                     segLengthDict=oc_build.make_target_dict(cell_object=cellObject,target_segs=target_segments) 
                     
               else:
               
                  target_segment=input_group_params['UniversalTargetSegmentID']
                  
                  fraction_along=input_group_params['UniversalFractionAlong']
               
                  segLengthDict=None
                  
                  subset_dict=None
                  
               if input_group_params['InputType']=='GeneratePoissonTrains':
               
                  list_of_input_ids=[]
            
                  if input_group_params['TrainType']=='transient':
               
                     for input_index in range(0,len(input_group_params['AverageRateList']) ):
                  
                         tpfs=oc_build.add_transient_poisson_firing_synapse(nml_doc=nml_doc, 
                                                                id=input_group_tag+"_TransPoiSyn%d"%input_index, 
                                                                average_rate="%f %s"%(input_group_params['AverageRateList'][input_index],input_group_params['RateUnits']),
                                                                delay="%f %s"%(input_group_params['DelayList'][input_index],input_group_params['TimeUnits']),
                                                                duration="%f %s"%(input_group_params['DurationList'][input_index],input_group_params['TimeUnits']), 
                                                                synapse_id=input_group_params['Synapse'])
                                
                         input_synapse_list.append(input_group_params['Synapse'])                                          
                         list_of_input_ids.append(tpfs.id)
                         
                  
                  if input_group_params['TrainType']=='persistent':
               
                     for input_index in range(0,len(input_group_params['AverageRateList']) ):
                       
                         pfs=oc_build.add_poisson_firing_synapse(nml_doc=nml_doc, 
                                                     id=input_group_tag+"_PoiSyn%d"%input_index, 
                                                     average_rate="%f %s"%(input_group_params['AverageRateList'][input_index],input_group_params['RateUnits']), 
                                                     synapse_id=input_group_params['Synapse'])
                                                     
                                                     
                         input_synapse_list.append(input_group_params['Synapse'])                           
                         list_of_input_ids.append(pfs.id)
                         
                         
                  input_ids_final.append(list_of_input_ids)
                      
               if input_group_params['InputType']=='PulseGenerators':
            
                  if not input_group_params['Noise']:
                  
                     list_of_input_ids=[]
               
                     for input_index in range(0,len(input_group_params['AmplitudeList']) ):
               
                         pg=oc_build.add_pulse_generator(nml_doc=nml_doc, 
                                          id=input_group_tag+"_Pulse%d"%input_index, 
                                          delay="%f %s"%(input_group_params['DelayList'][input_index],input_group_params['TimeUnits']),
                                          duration="%f %s"%(input_group_params['DurationList'][input_index],input_group_params['TimeUnits']), 
                                          amplitude="%f %s"%(input_group_params['AmplitudeList'][input_index],input_group_params['AmplitudeUnits']) )
                                          
                         list_of_input_ids.append(pg.id)
                         
                     input_ids_final.append(list_of_input_ids)
                         
                  else:
                  
                     for cell in target_cell_ids:
                     
                         list_of_input_ids=[]
                     
                         for input_index in range(0,len(input_group_params['SmallestAmplitudeList'])):
                         
                             random_amplitude=random.uniform(input_group_params['SmallestAmplitudeList'][input_index],input_group_params['LargestAmplitudeList'][input_index])
               
                             pg=oc_build.add_pulse_generator(nml_doc=nml_doc, 
                                                             id=input_group_tag+"_Pulse%d_Cell%d"%(input_index,cell), 
                                                             delay="%f %s"%(input_group_params['DelayList'][input_index],input_group_params['TimeUnits']),
                                                             duration="%f %s"%(input_group_params['DurationList'][input_index],input_group_params['TimeUnits']), 
                                                             amplitude="%f %s"%(random_amplitude,input_group_params['AmplitudeUnits']) )
                                          
                             list_of_input_ids.append(pg.id)
                             
                         input_ids_final.append(list_of_input_ids)
                         
            
               input_list_array=oc_build.add_advanced_inputs_to_population(net=net, 
                                                                           id=input_group_tag, 
                                                                           population=pop,
                                                                           input_id_list=input_ids_final,
                                                                           seg_length_dict=segLengthDict,
                                                                           subset_dict=subset_dict,
                                                                           universal_target_segment=target_segment,
                                                                           universal_fraction_along=fraction_along,
                                                                           only_cells=target_cell_ids)
                                                                             
                                                                                                                                   
               input_list_array_final.append(input_list_array)
     
    input_synapse_list=list(set(input_synapse_list))  
    
    return input_list_array_final, input_synapse_list  

####################################################################################################################################################################    
    
def replace_cell_types(net_file_name,
                       path_to_net,
                       new_net_file_name,
                       cell_types_to_be_replaced,
                       cell_types_replaced_by,
                       dir_to_new_components,
                       reduced_to_single_compartment=True,
                       compartment_targeting_params=None):

    nml2_file_path=os.path.join(path_to_net,net_file_name+".net.nml")      
    
    net_doc = pynml.read_neuroml2_file(nml2_file_path)
    
    if len(cell_types_to_be_replaced)==len(cell_types_replaced_by):
    
       net_doc.includes=[]
       
       net=net_doc.networks[0]
       
       for cell_index in range(0,len(cell_types_replaced_by)):
        
           oc_build.add_cell_and_channels(net_doc, os.path.join(dir_to_new_components,"%s.cell.nml"%cell_types_replaced_by[cell_index]), cell_types_replaced_by[cell_index] )
       
           for pop_counter in range(0,len(net.populations) ):
           
               pop=net.populations[pop_counter]
               
               if pop.component==cell_types_to_be_replaced[cell_index]:
               
                  pop.component=cell_types_replaced_by[cell_index]
                  
               if cell_types_to_be_replaced[cell_index] in pop.id:
               
                  pop.id= pop.id.replace(cell_types_to_be_replaced[cell_index],cell_types_replaced_by[cell_index])
                  
           if hasattr(net, 'projections'):
                  
              for proj_counter in range(0,len(net.projections)):
            
                  proj=net.projections[proj_counter]
                  
                  wd_indicator=False
                  
                  if hasattr(proj,'connection_wds'):
                  
                     connections=proj.connection_wds
                     
                     wd_indicator=True
                  
                  if hasattr(proj,'connections'):
                  
                     connections=proj.connections
                  
                  for conn_counter in range(0,len(connections)):
                      
                      connection=proj.connection_wds[conn_counter]
                      
                      if cell_types_to_be_replaced[cell_index] in connection.pre_cell_id or cell_types_to_be_replaced[cell_index] in connection.post_cell_id:
                      
                         #TODO
                      
                         if hasattr(connection,'post_segment_id'):
                            
                         if hasattr(connection,'pre_segment_id'):
                       
                         if hasattr(connection,'pre_fraction_along'):
                   
                         if hasattr(connection,'post_fraction_along'):
                  
                         if hasattr(connection,'delay'):
                  
                         if hasattr(connection,'weight'):
                  
                          
       
           
           
       
       
    else:
    
      opencortex.print_comment_v("Error: the number of cell types in the list cell_types_to_be_replaced is not equal to the number of new cell types in the list"
      " cell_types_replaced_by.") 
      
      quit()
        
##############################################################################################################################################
def parse_distance_dependence_params(distance_dependence_params,pre_pop,post_pop):

    dist_rule=None

    for distance_param in range(0,len(distance_dependence_params)):
    
        if distance_dependence_params[distance_param]['prePopID']==pre_pop and distance_dependence_params[distance_param]['postPopID']==post_pop:
        
           dist_rule=distance_dependence_params['DistDependConn']
           
           break
           
    return dist_rule

def parse_delays(delays_params,post_pop,synapse_list):

    delays={}
    
    for syn_ind in range(0,len(synapse_list)):
    
        for delay_param in range(0,len(delays_params)):
        
            if delays_params[ delay_param]['synComp']=='all':
            
               delays[synapse_list[syn_ind]]=float(delays_params[delay_param]['delay'])
               
            else:
            
               passed_synComp=False
               
               if delays_params[delay_param]['synComp'] in synapse_list[syn_ind]:
               
                  passed_synComp=True
                  
               passed_synEndsWith=False
               
               if delays_params[delay_param]['synEndsWith']==[]:
               
                  passed_synEndsWith=True
                  
               else:
               
                 for syn_end in delays_params[delay_param]['synEndsWith']:
               
                     if synapse_list[syn_ind].endswith(syn_end):
                   
                        passed_synEndsWith=True
                      
                        break
                      
               passed_targetCellGroup=False
               
               if delays_params[delay_param]['targetCellGroup']==[]:
               
                  passed_targetCellGroup=True
                  
               else:
               
                  for target_cell_group in delays_params[delay_param]['targetCellGroup']:
               
                      if target_cell_group in post_pop:
                   
                         passed_targetCellGroup=True
                      
                         break
                      
               if passed_synComp and passed_synEndsWith and passed_targetCellGroup:
               
                  delays[synapse_list[syn_ind]]=float(delays_params[delay_param]['delay'])
               
    
    if delays.keys()==[]:
       
       delays=None
          
    return delays
    
def parse_weights(weights_params,post_pop,synapse_list):

    weights={}
    
    for syn_ind in range(0,len(synapse_list)):
    
        for weight_param in range(0,len(weights_params)):
        
            if weights_params[weight_param]['synComp']=='all':
            
               weights[synapse_list[syn_ind]]=float(weights_params[weight_param]['weight'])
               
            else:
            
               passed_synComp=False
               
               if weights_params[weight_param]['synComp'] in synapse_list[syn_ind]:
               
                  passed_synComp=True
                  
               passed_synEndsWith=False
               
               if weights_params[weight_param]['synEndsWith']==[]:
               
                  passed_synEndsWith=True
                  
               else:
               
                  for syn_end in weights_params[weight_param]['synEndsWith']:
               
                      if synapse_list[syn_ind].endswith(syn_end):
                   
                         passed_synEndsWith=True
                      
                         break
                      
               passed_targetCellGroup=False
               
               if weights_params[weight_param]['targetCellGroup']==[]:

                  passed_targetCellGroup=True
                  
               else:
               
                  for target_cell_group in weights_params[weight_param]['targetCellGroup']:
                
                      if target_cell_group in post_pop:
                   
                         passed_targetCellGroup=True
                      
                         break
                      
               if passed_synComp and passed_synEndsWith and passed_targetCellGroup:
               
                  weights[synapse_list[syn_ind]]=float(weights_params[weight_param]['weight'])
               
    
    if weights.keys()==[]:
       
       weights=None
          
    return weights

##############################################################################################################################
def check_includes_in_cells(dir_to_cells,
                            list_of_cell_ids,
                            extra_channel_tags=None):
                            
    passed=True
    
    list_of_cell_file_names=[]
    
    for cell_id in list_of_cell_ids:
    
        list_of_cell_file_names.append(cell_id+".cell.nml")
    
    all_src_files=os.listdir(dir_to_cells)
    
    target_files=list_of_cell_file_names
    
    for src_file in all_src_files:
    
        if src_file not in target_files:
        
           target_files.append(src_file)
           
    for cell_file_name in target_files:

        full_path_to_cell=os.path.join(dir_to_cells,cell_file_name)
    
        if not os.path.exists(full_path_to_cell):
    
           passed=False
           
           opencortex.print_comment_v("Error: path %s does not exist; use method copy_nml2_source to copy nml2 files from the source directory to the appropriate NeuroML2 component directories."%full_path_to_cell)
           
           break
           
        else:
        
           nml2_doc_cell=pynml.read_neuroml2_file(full_path_to_cell,include_includes=False)
        
           for included in nml2_doc_cell.includes:
        
               if '.channel.nml' in included.href:
               
                  if ('../channels/' not in included.href) or ('..\channels\'' not in included.href):
                  
                     channel_dir=os.path.join("..","channels")
                     
                     included.href=os.path.join(channel_dir,included.href)
                     
                     continue
                     
               else:
            
                  if extra_channel_tags != None:
               
                     for channel_tag in included.href:
                  
                         if channel_tag in included.href:
                      
                            if ('../channels/' not in included.href) or ('..\channels\'' not in included.href):
                  
                               channel_dir=os.path.join("..","channels")
                     
                               included.href=os.path.join(channel_dir,included.href)
                            
                               break
           
           pynml.write_neuroml2_file(nml2_doc_cell,full_path_to_cell)    
              
    return passed
########################################################################################################################################
def check_pop_dict_and_layers(pop_dict,boundary_dict):

    error_counter=0
    
    passed=False
    
    for cell_population in pop_dict.keys():
        
        if not isinstance(pop_dict[cell_population],tuple):
           print("TypeError in population parameters: the values stored in the population dictionary must be tuples.")
           print("The current type is %s"%(type(pop_dict[cell_population])  ) )
           error_counter+=1
           
        else:
        
           if not isinstance(pop_dict[cell_population][0],int):
              print("TypeError in population parameters: the first element in tuples in the population dictionary must be of type 'int'")
              print(" as it specifies the size of cell population. The current type of the first element is %s"%( type(pop_dict[cell_population][0] )  )  )
              error_counter+=1
              
           if not isinstance(pop_dict[cell_population][1],str):
              print("TypeError in population parameters: the second element in tuples in the population dictionary must be of type 'string'")
              print(" as it specifies the layer of cell population. The current type of the second element is %s"%( type(pop_dict[cell_population][1]) ) )
              error_counter+=1
              
           else:
           
              try:
              
                 test_layer=boundary_dict[pop_dict[cell_population][1]]
                 
              except KeyError:
              
                 print("KeyError in the layer boundary dictionary: cell population id '%s' is not in the keys of the layer boundary dictionary"%cell_population)
                 error_counter+=1
                 
                    
           
           if not isinstance(pop_dict[cell_population][2],str):
              print("TypeError in population parameters: the third element in tuples in the population dictionary must be of type 'string'")
              print(" as it specifies the cell model for a given population. The current type of the third element is %s"%( type(pp_dict[cell_population][2]) ) )
              error_counter+=1
           
    if error_counter==0:
    
       passed=True       
        
           
    return passed
    
 
def check_synapse_location(synapse_id,pathToSynapses):
    
    found=False
    
    if pathToSynapses ==None:
    
       path="./"
       
    else:
    
       path=pathToSynapses
    
    src_files=os.listdir(path)
    
    for file_name in src_files:
        if synapse_id in file_name:
           found=True   
           
           
    return found  
    
def get_segment_groups(cell_id,path_to_cells):

    if path_to_cells !=None:
   
       cell_nml_file =os.path.join(path_to_cells,'%s.cell.nml'%cell_id)
       
    else:
    
       cell_nml_file='%s.cell.nml'%cell_id
       
    document_cell=neuroml.loaders.NeuroMLLoader.load(cell_nml_file)
    
    cell_object=document_cell.cells[0]
    
    segment_groups=[]
    
    for segment_group in cell_object.morphology.segment_groups:
        
        segment_groups.append(segment_group.id)
        
    return segment_groups
    
def check_segment_group(segment_groups,target_segment_group):
    segment_group_exist=False
    if target_segment_group in segment_groups:
       segment_group_exist=True
    return segment_group_exist
    
    
def check_weight_params(weight_params):

    error_counter=0
    
    if not isinstance(weight_params,list):
    
       print("TypeError in weight parameters: weight parameters must be of type 'list'. The current type is '%s'."%(type(weight_params)))
       
       error_counter+=1
       
    else:
    
       for weight_param in range(0,len(weight_params)):
       
           if not isinstance(weight_params[weight_param],dict):
             
              print("TypeError in weight parameters: list elements in weight parameters must be of type 'dict'. The current type is '%s'."%(type(weight_params[weight_param])))
              
              error_counter+=1
              
           else:
           
              try:
              
                 test_weight_field=weight_params[weight_param]['weight']
                 
              except KeyError:
                 
                 print("KeyError in weight parameters: the key 'weight' is not in the keys of weight parameter dictionary.")
                 
                 error_counter+=1
                 
              try:
              
                 test_syn_comp_field=weight_params[weight_param]['synComp']
                 
                 if not isinstance(test_syn_comp_field,str):
                 
                    print("TypeError in weight parameters: the value of the key 'synComp' must be of type 'str'. The current type is '%s'."%(type(test_syn_comp_field)))
                    
                    error_counter+=1
                    
                 else:
                 
                    if test_syn_comp_field != 'all':
                    
                       try:
              
                         test_syn_ends_with_field=weight_params[weight_param]['synEndsWith']
                 
                         if not isinstance(test_syn_ends_with_field,list):
                 
                            print("TypeError in weight parameters: the value of the key 'synEndsWith' must be of type 'list'.")
                            print("The current type is '%s'."%(type(test_syn_ends_with_field)))
                 
                            error_counter+=1
                    
                       except KeyError:
              
                         print("KeyError in weight parameters: the key 'synEndsWith' is not in the keys of weight parameter dictionary.")
                 
                         error_counter+=1
                 
                       try:
              
                         test_target_cell_group=weight_params[weight_param]['targetCellGroup']
                 
                         if not isinstance(test_target_cell_group,list):
                 
                            print("TypeError in weight parameters: the value of the key 'targetCellGroup' must be of type 'list'.")
                            print("The current type is '%s'."%(type(test_target_cell_group)))
                    
                            error_counter+=1
                    
                       except KeyError:
              
                         print("KeyError in weight parameters: the key 'targetCellGroup' is not in the keys of weight parameter dictionary.")
                 
                         error_counter+=1
               
              except KeyError:
                 
                 print("KeyError in weight parameters: the key 'synComp' is not in the keys of weight parameter dictionary.")
                    
                 error_counter+=1
                 
    if error_counter==0:
    
       return True
       
    else:
    
       return False              
     
def check_delay_params(delay_params):

    error_counter=0
    
    if not isinstance(delay_params,list):
    
       print("TypeError in delay parameters: delay parameters must be of type 'list'. The current type is '%s'."%(type(delay_params)))
       
       error_counter+=1
       
    else:
    
       for delay_param in range(0,len(delay_params)):
       
           if not isinstance(delay_params[delay_param],dict):
             
              print("TypeError in delay parameters: list elements in delay parameters must be of type 'dict'. The current type is '%s'."%(type(delay_params[delay_param])))
              
              error_counter+=1
              
           else:
           
              try:
              
                 test_weight_field=delay_params[delay_param]['delay']
                 
              except KeyError:
                 
                 print("KeyError in delay parameters: the key 'delay' is not in the keys of delay parameter dictionary.")
                 
                 error_counter+=1
                 
              try:
              
                 test_syn_comp_field=delay_params[delay_param]['synComp']
                 
                 if not isinstance(test_syn_comp_field,str):
                 
                    print("TypeError in delay parameters: the value of the key 'synComp' must be of type 'str'. The current type is '%s'."%(type(test_syn_comp_field)))
                    
                    error_counter+=1
                    
                 else:
                 
                    if test_syn_comp_field != 'all':
                    
                       try:
              
                         test_syn_ends_with_field=delay_params[delay_param]['synEndsWith']
                 
                         if not isinstance(test_syn_ends_with_field,list):
                 
                            print("TypeError in delay parameters: the value of the key 'synEndsWith' must be of type 'list'.")
                            print("The current type is '%s'."%(type(test_syn_ends_with_field)))
                 
                            error_counter+=1
                    
                       except KeyError:
              
                         print("KeyError in delay parameters: the key 'synEndsWith' is not in the keys of delay parameter dictionary.")
                 
                         error_counter+=1
                 
                       try:
              
                         test_target_cell_group=delay_params[delay_param]['targetCellGroup']
                 
                         if not isinstance(test_target_cell_group,list):
                 
                            print("TypeError in delay parameters: the value of the key 'targetCellGroup' must be of type 'list'.")
                            print("The current type is '%s'."%(type(test_target_cell_group)))
                    
                            error_counter+=1
                    
                       except KeyError:
              
                         print("KeyError in delay parameters: the key 'targetCellGroup' is not in the keys of delay parameter dictionary.")
                 
                         error_counter+=1
               
              except KeyError:
                 
                 print("KeyError in delay parameters: the key 'synComp' is not in the keys of delay parameter dictionary.")
                    
                 error_counter+=1
                 
    if error_counter==0:
    
       return True
       
    else:
    
       return False              
       
def check_inputs(input_params,popDict,path_to_cells,path_to_synapses):
    
    error_counter=0
    
    for cell_receiver in input_params.keys():
    
        try:
           test_cell_component=popDict[cell_receiver]
           
           segment_groups=get_segment_groups(test_cell_component['PopObj'].component,path_to_cells)
           
           cell_type=test_cell_component['PopObj'].component
           
        except KeyError:
           opencortex.print_comment_v("KeyError in input parameters: cell population id '%s' is not in the keys of population dictionary"%cell_receiver)
           error_counter+=1
           cell_type=None
           
        if not isinstance(input_params[cell_receiver],list):
       
           opencortex.print_comment_v("TypeError in input parameters: the dictionary value for '%s' must be a list."
           " The current type is %s"%(cell_receiver,type(input_params[cell_receiver])))
           
           error_counter+=1
           
        else:
        
           for input_group_ind in range(0,len(input_params[cell_receiver])):
           
               input_group_params=input_params[cell_receiver][input_group_ind]
               
               try:
               
                 test_key=input_group_params['InputType']
                 
                 if not isinstance(test_key,str):
                 
                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'InputType' must be of type 'string'. The current type is %s"%type(test_key))
                    
                    error_counter+=1 
                    
                 if test_key not in ['GeneratePoissonTrains','PulseGenerators']:
                 
                    opencortex.print_comment_v("ValueError in input parameters: the value of the key 'InputType' must be one of the following: "   
                    "'GeneratePoissonTrains','PulseGenerators'")
                    
                    error_counter+=1
                    
                 else:
                 
                    if test_key=="GeneratePoissonTrains":
                    
                       try:
                          test_train_type=input_group_params['TrainType']
                          
                          if not isinstance(test_train_type,str):
                          
                             opencortex.print_comment_v("TypeError in input parameters: the value of the key 'TrainType' must be of type 'string'. "
                             "The current type is %s"%type(test_train_type))
                             
                             error_counter+=1
                          else:
                          
                             if test_train_type not in ['persistent','transient']:
                             
                                opencortex.print_comment_v("ValueError in input parameters: the value of the key 'TrainType' when 'InputType' is 'GeneratePoissonTrains' must be"
                                " one of the following: 'persistent' or 'transient'")
                                
                                error_counter+=1
                                
                             else:
                             
                                if test_train_type=="persistent":
                                
                                   try:
                                   
                                      test_rates=input_group_params['AverageRateList']
                                      
                                      if not isinstance(test_rates,list):
                                      
                                         opencortex.print_comment_v("TypeError in input parameters: the value of the key 'AverageRateList' must be of type 'list'."
                                         " The current type is %s"%type(test_rates) )
                                         
                                         error_counter+=1
                                         
                                      else:
                                         for r in range(0,len(test_rates)):
                                         
                                             if not isinstance(test_rates[r],float):
                                             
                                                opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'AverageRateList' must be of type 'float'."
                                                " The current type is %s"%type(test_rates[r]) )
                                                
                                                error_counter+=1
                                      
                                   except KeyError:
                                   
                                      opencortex.print_comment_v("KeyError in input parameters: the key 'AverageRateList' is not in the keys of input parameters.")
                                      
                                      error_counter+=1
                                      
                                if test_train_type=="transient":
                                
                                   try:
                                   
                                      test_time_units=input_group_params['TimeUnits']
                                      
                                      if not isinstance(test_time_units,str):
                                      
                                         opencortex.print_comment_v("TypeError in input parameters: the value of the key 'TimeUnits' must be of type 'str'."
                                         " The current type is %s."%type(test_time_units)) 
                                         
                                         error_counter+=1
                                         
                                   except KeyError:
                                   
                                      opencortex.print_comment_v("KeyError in input parameters: the key 'TimeUnits' is not in the keys of input parameters.")
                                      
                                      error_counter+=1
                                
                                
                                   try:
                                   
                                      test_rates=input_group_params['AverageRateList']
                                      
                                      if not isinstance(test_rates,list):
                                      
                                         opencortex.print_comment_v("TypeError in input parameters: the value of the key 'AverageRateList' must be of type 'list'."
                                         " The current type is %s."%type(test_rates) )
                                         
                                         error_counter+=1
                                         
                                      else:
                                      
                                         for r in range(0,len(test_rates)):
                                         
                                             if not isinstance(test_rates[r],float):
                                             
                                                opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'AverageRateList' must be of type 'float'."
                                                " The current type is %s."%type(test_rates[r]) )
                                                
                                                error_counter+=1
                                      
                                   except KeyError:
                                   
                                      opencortex.print_comment_v("KeyError in input parameters: the key 'AverageRateList' is not in the keys of input parameters.")
                                      
                                      error_counter+=1
                                   
                                   try:
                                   
                                      test_rates=input_group_params['DelayList']
                                      
                                      if not isinstance(test_rates,list):
                                      
                                         opencortex.print_comment_v("TypeError in input parameters: the value of the key 'DelayList' must be of type 'list'."
                                         " The current type is %s."%type(test_rates) )
                                         
                                         error_counter+=1
                                         
                                      else:
                                      
                                         for r in range(0,len(test_rates)):
                                         
                                             if not isinstance(test_rates[r],float):
                                             
                                                opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'DelayList' must be of type 'float'."
                                                " The current type is %s."%type(test_rates[r])  )
                                                
                                                error_counter+=1
                                         
                                   except KeyError:
                                   
                                      opencortex.print_comment_v("KeyError in input parameters: the key 'DelayList' is not in the keys of input parameters.")
                                      
                                      error_counter+=1
                                   
                                   try:
                                   
                                      test_rates=input_group_params['DurationList']
                                      
                                      if not isinstance(test_rates,list):
                                      
                                         opencortex.print_comment_v("TypeError in input parameters: the value of the key 'DurationList' must be of type 'list'."
                                         " The current type is %s."%type(test_rates) )
                                         
                                         error_counter+=1
                                         
                                      else:
                                      
                                         for r in range(0,len(test_rates)):
                                         
                                             if not isinstance(test_rates[r],float):
                                             
                                                opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'DurationList' must be of type 'float'."
                                                " The current type is %s."%type(test_rates[r]) )
                                                
                                                error_counter+=1
                                      
                                   except KeyError:
                                   
                                      opencortex.print_comment_v("KeyError in input parameters: the key 'DurationList' is not in the keys of input parameters.")
                                      
                                      error_counter+=1
                                     
                       except KeyError:
                       
                          opencortex.print_comment_v("KeyError in input parameters: the key 'TrainType' is not in the keys of input parameters.")
                          
                          error_counter+=1
                          
                       try:
                       
                          test_rate_units=input_group_params['RateUnits']
                           
                          if not isinstance(test_rate_units,str):
                          
                             opencortex.print_comment_v("TypeError in input parameters: the value of the key 'RateUnits' must be of type 'str'."
                             "The current type is %s."%type(test_rate_units))
                             
                             error_counter+=1
                             
                       except KeyError:
                       
                              opencortex.print_comment_v("KeyError in input parametres: the key 'RateUnits' is not in the keys of inputs parameters.")
                              
                              error_counter+=1
                           
                       try:
                       
                          test_synapse=input_group_params['Synapse']
                          
                          if not isinstance(test_synapse,str):
                          
                             opencortex.print_comment_v("TypeError in input parameters: the value of the key 'Synapse' must be of type 'str'."
                             " The current type is %s."%type(test_synapse) )
                             
                             error_counter+=1
                             
                          else:
                          
                             found=check_synapse_location(test_synapse,path_to_synapses)
                             
                             if not found:
                             
                                opencortex.print_comment_v("ValueError in input parameters: the value '%s' of the key 'Synapse' is not found in %s"%(test_synapse,path_to_synapses))
                                
                                error_counter+=1
                                
                       except KeyError:
                       
                           opencortex.print_comment_v("KeyError in input parameters: the key 'Synapse' is not in the keys of input parameters.")  
                            
                           error_counter+=1
                           
                    if test_key=='PulseGenerators':
                    
                       try:
                                   
                          test_time_units=input_group_params['TimeUnits']
                                      
                          if not isinstance(test_time_units,str):
                                      
                             opencortex.print_comment_v("TypeError in input parameters: the value of the key 'TimeUnits' must be of type 'str'."
                             " The current type is %s."%type(test_time_units)) 
                                         
                             error_counter+=1
                                         
                       except KeyError:
                                   
                          opencortex.print_comment_v("KeyError in input parameters: the key 'TimeUnits' is not in the keys of input parameters.")
                                      
                          error_counter+=1
                    
                       try:
                       
                          test_amplitude_units=input_group_params['AmplitudeUnits']
                          
                          if not isinstance(test_amplitude_units,str):
                          
                             opencortex.print_comment_v("TypeError in input parameters: the value of the key 'AmplitudeUnits' must be of type 'str'."
                             " The current type is %s."%type(test_amplitude_units) )
                             
                             error_counter+=1
                             
                       except KeyError:
                       
                          opencortex.print_comment_v("KeyError in input parametres: the key 'AmplitudeUnits' is not in the keys of inputs parameters.")
                          
                          error_counter+=1
                       
                       try:
                        
                          test_noise=input_group_params['Noise']
                          
                          if not isinstance(test_noise,bool):
                             
                             opencortex.print_comment_v("TypeError in input parameters: the value of the key 'Noise' must be of type 'bool'."
                             " The current type is %s."%type(test_noise) )
                             
                             error_counter+=1
                             
                          else:
                          
                             if test_noise:
                             
                                try:
                                
                                   test_smallest_amplitudes=input_group_params['SmallestAmplitudeList']
                                  
                                   if not isinstance(test_smallest_amplitudes,list):
                                  
                                      opencortex.print_comment_v("TypeError in input parameters: the value of the key 'SmallestAmplitudeList' must be of type 'list'."
                                      " The current type is %s."%type(test_smallest_amplitudes))
                             
                                      error_counter+=1
                                     
                                except KeyError:
                                 
                                   opencortex.print_comment_v("KeyError in input parameters: the key 'SmallestAmplitudeList' is not in the keys of input parameters when "
                                   "'Noise' is set to True.")
                                  
                                   error_counter+=1    
                                  
                                try:
                                
                                   test_largest_amplitudes=input_group_params['LargestAmplitudeList']
                                   
                                   if not isinstance(test_largest_amplitudes,list):
                                   
                                      opencortex.print_comment_v("TypeError in input parameters: the value of the key 'LargestAmplitudeList' must be of type 'list'."
                                      " The current type is %s."%type(test_largest_amplitudes) )
                                      
                                      error_counter+=1
                                      
                                except KeyError:
                                
                                   opencortex.print_comment_v("KeyError in input parameters: the key 'LargestAmplitudeList' is not in the keys of input parameters when "
                                   "'Noise' is set to True.")
                                   
                                   error_counter+=1
                             
                             else:
                             
                                try:
                       
                                   test_amplitudes=input_group_params['AmplitudeList']
                          
                                   if not isinstance(test_amplitudes,list):
                          
                                      opencortex.print_comment_v("TypeError in input parameters: the value of the key 'AmplitudeList' must be of type 'list'."
                                      " The current type is %s."%type(test_amplitudes))
                             
                                      error_counter+=1
                             
                                   else:
                          
                                      for r in range(0,len(test_amplitudes)):
                             
                                          if not isinstance(test_amplitudes[r],float):
                                 
                                             opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'AverageRateList' must be of type 'float'."
                                             " The current type is %s."%type(test_amplitudes[r]) )
                                    
                                             error_counter+=1
                                      
                                except KeyError:
                       
                                   opencortex.print_comment_v("KeyError in input parameters: the key 'AmplitudeList' is not in the keys of input parameters.")
                              
                                   error_counter+=1
                          
                       except KeyError:
                       
                          opencortex.print_comment_v("KeyError in input parameters: the key 'Noise' is not in the keys of input parameters.")
                          
                          error_counter+=1
                                
                       try:
                          test_delays=input_group_params['DelayList']
                                      
                          if not isinstance(test_delays,list):
                          
                             opencortex.print_comment_v("TypeError in input parameters: the value of the key 'DelayList' must be of type 'list'."
                             " The current type is %s."%type(test_delays))
                             
                             error_counter+=1
                             
                          else:
                          
                             for r in range(0,len(test_delays)):
                             
                                 if not isinstance(test_delays[r],float):
                                 
                                    opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'DelayList' must be of type 'float'."
                                    " The current type is %s."%type(test_delays[r]) )
                                    
                                    error_counter+=1
                                         
                       except KeyError:
                       
                              opencortex.print_comment_v("KeyError in input parameters: the key 'DelayList' is not in the keys of input parameters.")
                             
                              error_counter+=1
                                   
                       try:
                       
                          test_durations=input_group_params['DurationList']
                          
                          if not isinstance(test_durations,list):
                          
                             opencortex.print_comment_v("TypeError in input parameters: the value of the key 'DurationList' must be of type 'list'."
                             " The current type is %s."%type(test_durations) )
                             
                             error_counter+=1
                             
                          else:
                          
                             for r in range(0,len(test_durations)):
                             
                                 if not isinstance(test_durations[r],float):
                                 
                                    opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'DurationList' must be of type 'float'."
                                    " The current type is %s."%type(test_durations[r]) )
                                    
                                    error_counter+=1
                                      
                       except KeyError:
                       
                              opencortex.print_comment_v("KeyError in input parameters: the key 'DurationList' is not in the keys of input parameters.")
                              
                              error_counter+=1
                          
                    
               except KeyError:
               
                      opencortex.print_comment_v("KeyError in input parameters: the key 'InputType' is not in input parameters.")
                      
                      error_counter+=1
             
               try:
               
                  test_input_name=input_group_params['InputName']
                  
                  if not isinstance(test_input_name,str):
                  
                     opencortex.print_comment_v("TypeError in input parameters: the value of the key 'InputName' must be of type 'str'."
                     
                     "The current type is %s."%type(test_input_name))
                     
                     error_counter+=1
                     
               except KeyError:
               
                      opencortex.print_comment_v("KeyError in input parameters: the key 'InputName' is not in input parameters.")   
                         
                      error_counter+=1
                      
               if ('UniversalTargetSegmentID' not in input_group_params.keys()) and ('UniversalFractionAlong' not in input_group_params.keys()):
               
                  try:
               
                     test_key=input_group_params['TargetDict']
                 
                     if not isinstance(test_key,dict):
                 
                        opencortex.print_comment_v("TypeError in input parameters: the value of the key 'TargetDict' in input parameters must be of type 'dict'."
                        " The current type is %s."%type(test_key) ) 
                    
                        error_counter+=1
                    
                     else:
                    
                        if cell_type != None:
                    
                           for target_segment_group in test_key.keys():
                       
                               if not check_segment_group(segment_groups,target_segment_group):
                           
                                  opencortex.print_comment_v("ValueError in input parameters: '%s' is not a segment group of the cell type '%s'"%(target_segment_group,cell_receiver) )
                              
                                  error_counter+=1
                              
                               else:
                           
                                  if not isinstance(test_key[target_segment_group],int):
                              
                                     opencortex.print_comment_v("TypeError in input parameters: the value of the key '%s' must be of type 'int'."
                                     " The current type is %s"%(target_segment_group,type(test_key[target_segment_group]) ) )
                                
                                     error_counter+=1
                                
                  except KeyError:
              
                       opencortex.print_comment_v("KeyError in input parameters: the key 'TargetDict' is not in input parameters.")
                 
                       error_counter+=1
              
               if 'TargetDict' in input_group_params.keys():
               
                  if 'UniversalTargetSegmentID' in input_group_params.keys():
                  
                     opencortex.print_comment_v("KeyError in input parameters: the key 'UniversalTargetSegmentID' cannot be specified together with the key 'TargetDict'.")
                     
                     error_counter+=1
                  
                  if 'UniversalFractionAlong' in input_group_params.keys():
                  
                     opencortex.print_comment_v("KeyError in input parameters: the key 'UniversalFractionAlong' cannot be specified together with the key 'TargetDict'.")
                     
                     error_counter+=1
                 
               else:
               
                  try:
                  
                     test_target_seg_id=input_group_params['UniversalTargetSegmentID']
                     
                  except KeyError:
                  
                     opencortex.print_comment_v("KeyError in input parameters: the key 'UniversalTargetSegmentID' must be specified when the key 'TargetDict' is not in "
                     "input parameters.")
                     
                     error_counter+=1
                     
                  try:
                   
                     test_fraction_along=input_group_params['UniversalFractionAlong']
                     
                  except KeyError:
                   
                     opencortex.print_comment_v("KeyError in input parameters: the key 'UniversalFractionAlong' must be specified when the key 'TargetDict' is not in "
                     "input parameters.")
                     
                     error_counter+=1
               
               try:
                 
                 test_key=input_group_params['FractionToTarget']
                 
                 if not isinstance(test_key,float):
                 
                    opencortex.print_comment_v("TypeError: the value of the key 'FractionToTarget' must be of type 'float'. The current type is %s."%type(test_key) )
                    
                    error_counter+=1
                    
               except KeyError:
               
                 opencortex.print_comment_v("KeyError: the key 'FractionToTarget' is not in input parameters.")
                 
                 error_counter+=1
               
                 
               try:
                 
                 test_key=input_group_params['LocationSpecific']
                 
                 
                 if not isinstance(test_key,bool):
                 
                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'LocationSpecific' must be of the type 'bool'."
                    " The current type is %s."%type(test_key) ) 
                    
                    error_counter+=1
                    
                 else:
                    
                    if test_key:
                    
                       try:
                         
                         test_region_key=input_group_params['TargetRegions']
                         
                         if not isinstance(test_region_key,list):
                         
                            opencortex.print_comment_v("TypeError in input parameters: the value of the key 'TargetRegions' must be of the type 'list'."
                            " The current type is %s."%type(test_region_key) )
                            
                            error_counter+=1
                            
                         else:
                         
                            for region in range(0,len(test_region_key)):
                            
                                if not isinstance(test_region_key[region],dict):
                                
                                   opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'TargetRegions' must be of the type 'dict'."
                                   " The current type is %s."%type(test_region_key[region]) )
                                   
                                   error_counter+=1
                                   
                                else:
                                
                                    for dim_key in ['XVector','YVector','ZVector']:
                                     
                                        if dim_key not in test_region_key[region].keys():
                                        
                                            opencortex.print_comment_v("ValueError in input parameters: the list values of the key 'TargetRegions' must be dictionaries "
                                            "with the following keys: 'XVector', 'YVector', 'ZVector'.")
                                            
                                            error_counter+=1 
                                            
                                        else:
                                        
                                            if not isinstance(test_region_key[region][dim_key],list):
                                            
                                               opencortex.print_comment_v("TypeError in input parametres: the 'X/Y/ZVector' must store the value of type 'list'."
                                               " The current type is %s."%type(test_region_key[region][dim_key]) )
                                               
                                               error_counter+=1
                                               
                                            else:
                                            
                                               if len(test_region_key[region][dim_key]) !=2:
                                               
                                                  opencortex.print_comment_v("ValueError in input parameters: the lists stored by 'XVector', 'YVector' and 'ZVector'"
                                                  " must contain two values.")
                                                  
                                                  error_counter+=1
                                                  
                                               else:
                                               
                                                  if (test_region_key[region][dim_key][0]-test_region_key[region][dim_key][1]) ==0:
                                                  
                                                     opencortex.print_comment_v("ValueError in input parameters: the lists stored by 'XVector', 'YVector' and 'ZVector'"
                                                     " must contain two different values.")
                                                     
                                                     error_counter+=1
                         
                       except KeyError:
                         
                         opencortex.print_comment_v("KeyError in input parameters: 'LocationSpecific' is True but the key 'TargetRegions' is not in input parameters.")
                         
                         error_counter+=1
                         
                      
                    
               except KeyError:
              
                  opencortex.print_comment_v("KeyError in input parameters: the key 'LocationSpecific' is not in input parameters.")
                  
                  error_counter+=1
                 
    if error_counter==0:
    
       return True
       
    else:
    
       return False       
#########################################################################################################################################           
    
