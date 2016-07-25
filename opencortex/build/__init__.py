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

all_cells = {}
all_included_files = []

    
def add_connection(projection, 
                   id, 
                   presynaptic_population, 
                   pre_cell_id, 
                   pre_seg_id, 
                   postsynaptic_population, 
                   post_cell_id, 
                   post_seg_id,
                   delay,
                   weight,
                   pre_fraction=0.5,
                   post_fraction=0.5):
    
    connection = neuroml.ConnectionWD(id=id, \
                            pre_cell_id="../%s/%i/%s"%(presynaptic_population.id, pre_cell_id, presynaptic_population.component), \
                            pre_segment_id=pre_seg_id, \
                            pre_fraction_along=pre_fraction,
                            post_cell_id="../%s/%i/%s"%(postsynaptic_population.id, post_cell_id, postsynaptic_population.component), \
                            post_segment_id=post_seg_id,
                            post_fraction_along=post_fraction,
                            delay = '%s ms'%delay,
                            weight = weight)

    projection.connection_wds.append(connection)
    

#############################################################################################################################
def add_elect_connection(projection, 
                         id, 
                         presynaptic_population, 
                         pre_cell_id, 
                         pre_seg_id, 
                         postsynaptic_population, 
                         post_cell_id, 
                         post_seg_id,
                         gap_junction_id,
                         pre_fraction=0.5,
                         post_fraction=0.5):
    
    connection =neuroml.ElectricalConnectionInstance(id=id,\
                                                     pre_cell="../%s/%i/%s"%(presynaptic_population.id, pre_cell_id, presynaptic_population.component),\
                                                     post_cell="../%s/%i/%s"%(postsynaptic_population.id, post_cell_id,postsynaptic_population.component),\
                                                     synapse=gap_junction_id,\
                                                     pre_segment=pre_seg_id,\
                                                     post_segment=post_seg_id,\
                                                     pre_fraction_along=pre_fraction,\
                                                     post_fraction_along=post_fraction)
                                                     
    projection.electrical_connection_instances.append(connection)
    
    
########################################################################################################################################        


def add_probabilistic_projection(net, 
                                 prefix, 
                                 presynaptic_population, 
                                 postsynaptic_population, 
                                 synapse_id,  
                                 connection_probability,
                                 delay = 0,
                                 weight = 1):
    
    if presynaptic_population.size==0 or postsynaptic_population.size==0:
        return None

    proj = neuroml.Projection(id="%s_%s_%s"%(prefix,presynaptic_population.id, postsynaptic_population.id), 
                      presynaptic_population=presynaptic_population.id, 
                      postsynaptic_population=postsynaptic_population.id, 
                      synapse=synapse_id)


    count = 0

    for i in range(0, presynaptic_population.size):
        for j in range(0, postsynaptic_population.size):
            if i != j or presynaptic_population.id != postsynaptic_population.id:
                if connection_probability>= 1 or random.random() < connection_probability:
                    add_connection(proj, 
                                   count, 
                                   presynaptic_population, 
                                   i, 
                                   0, 
                                   postsynaptic_population, 
                                   j, 
                                   0,
                                   delay = delay,
                                   weight = weight)
                    count+=1

    net.projections.append(proj)

    return proj

###################################################################################################################################################################   

def add_chem_projection(net,
                        proj_array,
                        presynaptic_population,
                        postsynaptic_population,
                        targeting_mode,
                        synapse_list,
                        seg_target_dict,
                        subset_dict,
                        delays_dict=None,
                        weights_dict=None):
    
    
    '''This method adds the divergent or convergent chemical projection depending on the input argument targeting_mode. The input arguments are as follows:
    
    net - the network object created using libNeuroML API ( neuroml.Network() );
    
    proj_array - dictionary which stores the projections of class neuroml.Projection; each projection has unique synapse component (e.g. AMPA , NMDA or GABA);
    thus the keys of proj_array must be equal to the synapse ids in the synapse_list;
    
    presynaptic_population - object corresponding to the presynaptic population in the network;
    
    postsynaptic_population - object corresponding to the postsynaptic population in the network;
    
    targeting_mode - a string that specifies the targeting mode: 'convergent' or 'divergent';
    
    synapse_list - the list of synapse ids that correspond to the individual receptor components on the physical synapse, e.g. the first element is
    the id of the AMPA synapse and the second element is the id of the NMDA synapse; these synapse components will be mapped onto the same location of the target segment;
    
    seg_target_dict - a dictionary whose keys are the ids of target segment groups and the values are dictionaries in the format returned by make_target_dict();
    
    subset_dict - a dictionary whose keys are the ids of target segment groups; interpretation of the corresponding dictionary values depends on the targeting mode:
    
    Case I, targeting mode = 'divergent' - the number of synaptic connections made by each presynaptic cell per given target segment group of postsynaptic cells;
    
    Case II, targeting mode = 'convergent' - the number of synaptic connections per target segment group per each postsynaptic cell;
    
    delays_dict - optional dictionary that specifies the delays (in ms) for individual synapse components, e.g. {'NMDA':5.0} or {'AMPA':3.0,'NMDA':5};
    
    weights_dict - optional dictionary that specifies the weights (in ms) for individual synapse components, e.g. {'NMDA':1} or {'NMDA':1,'AMPA':2}.'''    
    
    
    if targeting_mode=='divergent':
       
       pop1_size=presynaptic_population.size
       
       pop1_id=presynaptic_population.id
       
       pop2_size=postsynaptic_population.size
       
       pop2_id=postsynaptic_population.size
       
    if targeting_mode=='convergent':
       
       pop1_size=postsynaptic_population.size
       
       pop1_id=postsynaptic_population.id
       
       pop2_size=presynaptic_population.size
       
       pop2_id=presynaptic_population.id
                 
    total_given=int(sum(subset_dict.values() ))
    
    count=0
    
    for i in range(0, pop1_size):
    
        pop2_cell_ids=range(0,pop2_size)
        
        if pop1_id == pop2_id:
           
           pop2_cell_ids.remove(i)
           
        if pop2_cell_ids != []:
           
           if len(pop2_cell_ids) >= total_given:
              ##### get unique set of cells
              pop2_cells=random.sample(pop2_cell_ids,total_given)
           
           else:
              #### any cell might appear several times
              pop2_cells=[]
              
              for value in range(0,total_given):
           
                  cell_id=random.sample(pop2_cell_ids,1)
               
                  pop2_cells.extend(cell_id)
        
           target_seg_array, target_fractions=get_target_segments(seg_target_dict,subset_dict)
        
           for j in pop2_cells:
        
               post_seg_id=target_seg_array[0]
            
               del target_seg_array[0]
            
               fraction_along=target_fractions[0]
            
               del target_fractions[0]  
                
               if targeting_mode=='divergent':
                   
                  pre_cell_id=i
                      
                  post_cell_id=j
                      
               if targeting_mode=='convergent':
                   
                  pre_cell_id=j
                      
                  post_cell_id=i  
               
               syn_counter=0
                        
               for synapse_id in synapse_list:
            
                   delay=0
                
                   weight=1
                
                   if delays_dict !=None:
                      for synapseComp in delays_dict.keys():
                          if synapseComp in synapse_id:
                             delay=delays_dict[synapseComp]
                          
                   if weights_dict !=None:
                      for synapseComp in weights_dict.keys():
                          if synapseComp in synapse_id:
                             weight=weights_dict[synapseComp]
                     
                   add_connection(proj_array[syn_counter], 
                                  count, 
                                  presynaptic_population, 
                                  pre_cell_id, 
                                  0, 
                                  postsynaptic_population, 
                                  post_cell_id, 
                                  post_seg_id,
                                  delay = delay,
                                  weight = weight,
                                  post_fraction=fraction_along)
                    
                                  
                   syn_counter+=1               
                     
               count+=1
               
               
    if count !=0:   
                   
       for synapse_ind in range(0,len(synapse_list)):
    
           net.projections.append(proj_array[synapse_ind])

    return proj_array  
    
#####################################################################################################################################################
    
def add_elect_projection(net,
                         proj_array,
                         presynaptic_population,
                         postsynaptic_population,
                         targeting_mode,
                         synapse_list,
                         seg_target_dict,
                         subset_dict):
    
    '''This method adds the divergent or convergent electrical projection depending on the input argument targeting_mode. The input arguments are as follows:
    
    net - the network object created using libNeuroML API ( neuroml.Network() );
    
    proj_array - dictionary which stores the projections of class neuroml.Projection; each projection has unique gap junction component;
    thus the keys of proj_array must be equal to the gap junction ids in the synapse_list;
    
    presynaptic_population - object corresponding to the presynaptic population in the network;
    
    postsynaptic_population - object corresponding to the postsynaptic population in the network;
    
    targeting_mode - a string that specifies the targeting mode: 'convergent' or 'divergent';
    
    synapse_list - the list of gap junction (synapse) ids that correspond to the individual gap junction components on the physical contact;
    these components will be mapped onto the same location of the target segment;
    
    seg_target_dict - a dictionary whose keys are the ids of target segment groups and the values are dictionaries in the format returned by make_target_dict();
    
    subset_dict - a dictionary whose keys are the ids of target segment groups; interpretation of the corresponding dictionary values depends on the targeting mode:
    
    Case I, targeting mode = 'divergent' - the number of synaptic connections made by each presynaptic cell per given target segment group of postsynaptic cells;
    
    Case II, targeting mode = 'convergent' - the number of synaptic connections per target segment group per each postsynaptic cell.'''    
    
    if targeting_mode=='divergent':
       
       pop1_size=presynaptic_population.size
       
       pop1_id=presynaptic_population.id
       
       pop2_size=postsynaptic_population.size
       
       pop2_id=postsynaptic_population.size
       
    if targeting_mode=='convergent':
       
       pop1_size=postsynaptic_population.size
       
       pop1_id=postsynaptic_population.id
       
       pop2_size=presynaptic_population.size
       
       pop2_id=presynaptic_population.id
                     
    count=0
    
    numberConnections={}
    
    for subset in subset_dict.keys():
        
        numberConnections[subset]=int(subset_dict[subset])
      
    for i in range(0, pop1_size):
        
        total_conns=0
        
        conn_subsets={}
    
        for subset in subset_dict.keys():
            
            if subset_dict[subset] != numberConnections[subset]:
               
               if random.random() < subset_dict[subset] - numberConnections[subset]:
               
                  conn_subsets[subset]=numberConnections[subset]+1
                  
               else:
               
                  conn_subsets[subset]=numberConnections[subset]
                  
            else:
               
               conn_subsets[subset]=numberConnections[subset]
            
            total_conns=total_conns+conn_subsets[subset]
            
        if total_conns != 0:    
        
           pop2_cell_ids=range(0,pop2_size)
        
           if pop1_id == pop2_id:
           
              pop2_cell_ids.remove(i)
           
           if pop2_cell_ids !=[]:
        
              if len(pop2_cell_ids) >= total_conns:
        
                  pop2_cells=random.sample(pop2_cell_ids,total_conns)
           
              else:
        
                  pop2_cells=[]
           
                  for value in range(0,total_conns):
              
                      cell_id=random.sample(pop2_cell_ids,1)
                  
                      pop2_cells.extend(cell_id)
                 
              target_seg_array, target_fractions=get_target_segments(seg_target_dict,conn_subsets)
                 
              for j in pop2_cells:
           
                  post_seg_id=target_seg_array[0]
               
                  del target_seg_array[0]
               
                  fraction_along=target_fractions[0]
               
                  del target_fractions[0]  
                
                  if targeting_mode=='divergent':
                   
                     pre_cell_id=i
                      
                     post_cell_id=j
                      
                  if targeting_mode=='convergent':
                   
                     pre_cell_id=j
                      
                     post_cell_id=i  
                     
                  syn_counter=0   
                         
                  for synapse_id in synapse_list:
                         
                      add_elect_connection(proj_array[syn_counter], 
                                           count, 
                                           presynaptic_population, 
                                           pre_cell_id, 
                                           0, 
                                           postsynaptic_population, 
                                           post_cell_id, 
                                           post_seg_id,
                                           synapse_id,
                                           pre_fraction=0.5,
                                           post_fraction=fraction_along)
                                           
                      syn_counter+=1                      
                                           
                  count+=1
                  
    if count !=0:
                   
       for synapse_ind in range(0,len(synapse_list)):
    
           net.electrical_projections.append(proj_array[synapse_ind])

    return proj_array   
                       
    
#########################################################################################
    
def add_chem_spatial_projection(net,
                                proj_array,
                                presynaptic_population,
                                postsynaptic_population,
                                targeting_mode,
                                synapse_list,
                                seg_target_dict,
                                subset_dict,
                                distance_rule,
                                pre_cell_positions,
                                post_cell_positions,
                                delays_dict,
                                weights_dict):
    
    '''This method adds the divergent distance-dependent chemical projection. The input arguments are as follows:
    
    
    net - the network object created using libNeuroML API ( neuroml.Network() );
    
    proj_array - dictionary which stores the projections of class neuroml.Projection; each projection has unique synapse component (e.g. AMPA , NMDA or GABA);
    thus the keys of proj_array must be equal to the synapse ids in the synapse_list;
    
    presynaptic_population - object corresponding to the presynaptic population in the network;
    
    postsynaptic_population - object corresponding to the postsynaptic population in the network;
    
    targeting_mode - a string that specifies the targeting mode: 'convergent' or 'divergent';
    
    synapse_list - the list of synapse ids that correspond to the individual receptor components on the physical synapse, e.g. the first element is
    the id of the AMPA synapse and the second element is the id of the NMDA synapse; these synapse components will be mapped onto the same location of the target segment;
    
    seg_target_dict - a dictionary whose keys are the ids of target segment groups and the values are dictionaries in the format returned by make_target_dict();
    
    subset_dict - a dictionary whose keys are the ids of target segment groups; interpretation of the corresponding dictionary values depends on the targeting mode:
    
    Case I, targeting mode = 'divergent' - the desired number of synaptic connections made by each presynaptic cell per given target segment group of postsynaptic cells;
    
    Case II, targeting mode = 'convergent' - the desired number of synaptic connections per target segment group per each postsynaptic cell;
    
    Note: the chemical connection is made only if distance-dependent probability is higher than some random number random.random(); thus, the actual numbers of connections made
    
    according to the distance-dependent rule might be smaller than the numbers of connections specified by subset_dict; subset_dict defines the upper bound for the 
    
    number of connections.
    
    distance_rule - string which defines the distance dependent rule of connectivity - soma to soma distance must be represented by the string character 'r';
    
    pre_cell_positions- array specifying the cell positions for the presynaptic population; the format is an array of [ x coordinate, y coordinate, z coordinate];
    
    post_cell_positions- array specifying the cell positions for the postsynaptic population; the format is an array of [ x coordinate, y coordinate, z coordinate];
    
    delays_dict - optional dictionary that specifies the delays (in ms) for individual synapse components, e.g. {'NMDA':5.0} or {'AMPA':3.0,'NMDA':5};
    
    weights_dict - optional dictionary that specifies the weights (in ms) for individual synapse components, e.g. {'NMDA':1} or {'NMDA':1,'AMPA':2}.'''   
    
    if targeting_mode=='divergent':
       
       pop1_size=presynaptic_population.size
       
       pop1_id=presynaptic_population.id
       
       pop1_cell_positions=pre_cell_positions
       
       pop2_size=postsynaptic_population.size
       
       pop2_id=postsynaptic_population.size
       
       pop2_cell_positions=post_cell_positions
       
    if targeting_mode=='convergent':
       
       pop1_size=postsynaptic_population.size
       
       pop1_id=postsynaptic_population.id
       
       pop1_cell_positions=post_cell_positions
       
       pop2_size=presynaptic_population.size
       
       pop2_id=presynaptic_population.id
       
       pop2_cell_positions=pre_cell_positions
                 
    total_given=int( sum(subsets.values()) )
    
    count=0
    for i in range(0, pop1_size):
    
        pop2_cell_ids=range(0,pop2_size)
        
        if pop1_id == pop2_id:
           
           pop2_cell_ids.remove(i)
           
        if pop2_cell_ids !=[]:
        
           if len(pop2_cell_ids) >= total_given:
        
              pop2_cells=random.sample(pop2_cell_ids,total_given)
           
           else:
        
              pop2_cells=[]
           
              for value in range(0,total_given):
           
                  cell_id=random.sample(pop2_cell_ids,1)
               
                  pop2_cells.extend(cell_id)
    
           cell1_position=pop1_cell_positions[i]
        
           target_seg_array,fractions_along=get_target_segments(seg_target_dict,subset_dict)
        
           conn_counter=0
        
           for j in pop2_cells:
        
               cell2_position=pop2_cell_positions[j]
        
               r=math.sqrt(sum([(a - b)**2 for a,b in zip(cell1_position,cell2_position)]))
            
               if eval(distance_rule) >= 1 or random.random() < eval(distance_rule):
                  
                  conn_counter+=1
                     
                  post_seg_id=target_seg_array[0]
                     
                  del target_seg_array[0]
                     
                  fraction_along=fractions_along[0]
                     
                  del fractions_along[0]
                       
                  if targeting_mode=='divergent':
                   
                     pre_cell_id=i
                      
                     post_cell_id=j
                      
                  if targeting_mode=='convergent':
                   
                     pre_cell_id=j
                      
                     post_cell_id=i
                     
                  syn_counter=0
                                
                  for synapse_id in synapse_list:
                  
                      delay=0
                      
                      weight=1
                      
                      if delays_dict !=None:
                         for synapseComp in delays_dict.keys():
                             if synapseComp in synapse_id:
                                delay=delays_dict[synapseComp]
                                
                      if weights_dict !=None:
                         for synapseComp in weights_dict.keys():
                             if synapseComp in synapse_id:
                                weight=weights_dict[synapseComp]
                       
                        
                      add_connection(proj_array[syn_counter], 
                                     count, 
                                     presynaptic_population, 
                                     pre_cell_id, 
                                     0, 
                                     postsynaptic_population, 
                                     post_cell_id, 
                                     post_seg_id,
                                     delay = delay,
                                     weight = weight,
                                     post_fraction=fraction_along)
                                     
                                     
                      syn_counter+=1
                                     
                  count+=1
                     
               if conn_counter==total_given:
                  break
               
    if count !=0:
                   
       for synapse_ind in range(0,len(synapseList)):
    
           net.projections.append(proj_array[synapse_ind])

    return proj_array               


############################################################################################################################
def make_target_dict(cell_object,
                     target_segs):
    '''This method constructs the dictionary whose keys are the names of target segment groups or individual segments and the corresponding values are dictionaries
    with keys 'LengthDist' and 'SegList', as returned by the get_seg_lengths. Input arguments are as follows:
    
    cell_object - object created using libNeuroML API which corresponds to the target cell;
    target_segs - a dictionary in the format returned by the method extract_seg_ids(); the keys are the ids of target segment groups or names of individual segments and 
    the values are lists of corresponding target segment ids.'''
    
    targetDict={}
    for target in target_segs.keys():
        targetDict[target]={}
        lengths,segment_list=get_seg_lengths(cell_object,target_segs[target])
        targetDict[target]['LengthDist']=lengths
        targetDict[target]['SegList']=segment_list
    return targetDict
    
############################################################################################################################

def get_target_cells(pop_size,
                     fraction_to_target,
                     cell_positions=None,
                     list_of_xvectors=None,
                     list_of_yvectors=None,
                     list_of_zvectors=None):
                     
    '''This method returns the list of target cells according to which fraction of randomly selected cells is targeted and whether these cells are localized in the specific 
    rectangular regions of the network. These regions are specified by list_of_xvectors, list_of_yvectors and list_of_zvectors. These lists must have the same length.
    
    The input variable list_of_xvectors stores the lists whose elements define the left and right margins of the target rectangular regions along the x dimension.
    
    Similarly, the input variables list_of_yvectors and list_of_zvectors store the lists whose elements define the left and right margins of the target rectangular regions along
    the y and z dimensions, respectively.'''
    
    if cell_positions==None:
    
       target_cells=random.sample(range(pop_size),int(round(fraction_to_target*pop_size)   )   )
       
    else:
       
       region_specific_targets_per_cell_group=[]
       
       for region in range(0,len(list_of_xvectors)):
       
           for cell in range(0,pop_size):
           
               if (list_of_xvectors[region][0] <  cell_positions[cell,0]) and \
                  (cell_positions[cell,0] < list_of_xvectors[region][1]):
               
                   if (list_of_yvectors[region][0] <  cell_positions[cell,1]) and \
                      (cell_positions[cell,1] <  list_of_yvectors[region][1]) :
                
                      if (list_of_zvectors[region][0] <  cell_positions[cell,2]) and \
                         (cell_positions[cell,2] < list_of_zvectors[region][1]):
                     
                         region_specific_targets_per_cell_group.append(cell)
                                                                        
       target_cells=random.sample(region_specific_targets_per_cell_group,int(round(fraction_to_target*len(region_specific_targets_per_cell_group))))
                                                                   

    return target_cells

################################################################################################################################

def get_seg_lengths(cell_object,
                    target_segments):
                    
    '''This method constructs the cumulative distribution of target segments and the corresponding list of target segment ids.
      Input arguments: cell_object - object created using libNeuroML API which corresponds to the target cell; target_segments - the list of target segment ids. '''
    
    cumulative_length_dist=[]
    segment_list=[]
    totalLength=0
    for seg in cell_object.morphology.segments:
        for target_seg in target_segments:
            if target_seg==seg.id:
            
               if seg.distal !=None:
                  xd=seg.distal.x
                  yd=seg.distal.y
                  zd=seg.distal.z
                  
               if seg.proximal !=None:
                  xp=seg.proximal.x
                  yp=seg.proximal.y
                  zp=seg.proximal.z
               else:
                  if seg.parent != None:
                     get_segment_parent=seg.parent
                     get_segment_parent_id=get_segment_parent.segments
                     for segment_parent in cell_object.morphology.segments:
                         if segment_parent.id==get_segment_parent_id:
                            xp=segment_parent.distal.x
                            yp=segment_parent.distal.y
                            zp=segment_parent.distal.z
               dist=[xd,yd,zd]
               prox=[xp,yp,zp] 
               length=math.sqrt(sum([(a - b)**2 for a,b in zip(dist,prox)])) 
               
               segment_list.append(target_seg)
               totalLength=totalLength+length
               cumulative_length_dist.append(totalLength)
 
    return cumulative_length_dist, segment_list
############################################################################################################################
def extract_seg_ids(cell_object,
                    target_compartment_array,
                    targeting_mode):
                    
    '''This method extracts the segment ids that map on the target segment groups or individual segments. 
       cell_object is the loaded cell object using neuroml.loaders.NeuroMLLoader, target_compartment_array is an array of target compartment names (e.g. segment group ids or individual segment names) and targeting_mode is one of the strings: "segments" or "segGroups". '''
    
    segment_id_array=[]
    segment_group_array={}
    cell_segment_array=[]
    for segment in cell_object.morphology.segments:
        segment_id_array.append(segment.id)   
        segment_name_and_id=[]
        segment_name_and_id.append(segment.name)
        segment_name_and_id.append(segment.id)
        cell_segment_array.append(segment_name_and_id)
    for segment_group in cell_object.morphology.segment_groups:
        pooled_segment_group_data={}
        segment_list=[]
        segment_group_list=[]
        for member in segment_group.members:
            segment_list.append(member.segments)
        for included_segment_group in segment_group.includes:
            segment_group_list.append(included_segment_group.segment_groups)
                   
           
        pooled_segment_group_data["segments"]=segment_list
        pooled_segment_group_data["groups"]=segment_group_list
        segment_group_array[segment_group.id]=pooled_segment_group_data  
               
    
    target_segment_array={}

    if targeting_mode=="segments":
       
       for segment_counter in range(0,len(cell_segment_array)):
           for target_segment in range(0,len(target_compartment_array)):
               if cell_segment_array[segment_counter][0]==target_compartment_array[target_segment]: 
                  target_segment_array[target_compartment_array[target_segment]]=[cell_segment_array[segment_counter][1]]
          
                          
    if targeting_mode=="segGroups":
       
       for segment_group in segment_group_array.keys():
           for target_group in range(0,len(target_compartment_array)):
               if target_compartment_array[target_group]==segment_group:
                  segment_target_array=[]
                  if segment_group_array[segment_group]["segments"] !=[]:
                     for segment in segment_group_array[segment_group]["segments"]:
                         segment_target_array.append(segment)
                  if segment_group_array[segment_group]["groups"] !=[]:
                     for included_segment_group in segment_group_array[segment_group]["groups"]:
                         for included_segment_group_segment in segment_group_array[included_segment_group]["segments"]:
                             segment_target_array.append(included_segment_group_segment)
                  target_segment_array[target_compartment_array[target_group]]=segment_target_array
          
    

    return target_segment_array        
######################################################################################
def get_target_segments(seg_specifications,
                        subset_dict):
    
    '''This method generates the list of target segments and target fractions per cell according to two types of input dictionaries:
    seg_specifications - a dictionary in the format returned by make_target_dict(); keys are target group names or individual segment names
    and the corresponding values are dictionaries with keys 'LengthDist' and 'SegList', as returned by the get_seg_lengths;
    subset_dict - a dictionary whose keys are target group names or individual segment names; each key stores the corresponding number of connections per target group.'''
  
    
    target_segs_per_cell=[]
    target_fractions_along_per_cell=[]
    
    for target_group in subset_dict.keys():
   
        no_per_target_group=subset_dict[target_group]
        
        if target_group in seg_specifications.keys():
        
           target_segs_per_group=[]
           
           target_fractions_along_per_group=[]
           
           cumulative_length_dist=seg_specifications[target_group]['LengthDist']
           
           segment_list=seg_specifications[target_group]['SegList']
           not_selected=True
           while not_selected:
           
                 p=random.random()
                 
                 loc=p*cumulative_length_dist[-1]
                 
                 if len(segment_list)==len(cumulative_length_dist):
                    
                    for seg_index in range(0,len(segment_list)):
                    
                        current_dist_value=cumulative_length_dist[seg_index]
                        
                        if seg_index ==0:
                           
                           previous_dist_value=0
                           
                        else:
                        
                           previous_dist_value=cumulative_length_dist[seg_index-1]
                        
                        if loc > previous_dist_value and loc <  current_dist_value:
                        
                           segment_length=current_dist_value-previous_dist_value
                           
                           length_within_seg=loc-previous_dist_value
                           
                           post_fraction_along=float(length_within_seg)/segment_length
                           
                           target_segs_per_group.append(segment_list[seg_index])
                           
                           target_fractions_along_per_group.append(post_fraction_along)
                           
                           break
                           
                 if len(target_segs_per_group)==no_per_target_group:
                    not_selected=False
                    break
                              
           target_segs_per_cell.extend(target_segs_per_group)
           
           target_fractions_along_per_cell.extend(target_fractions_along_per_group)
             
    return target_segs_per_cell, target_fractions_along_per_cell

###########################################################################################################################
        
def include_cell_prototype(nml_doc,cell_nml2_path):
    
    nml_doc.includes.append(neuroml.IncludeType(cell_nml2_path)) 
    
# Helper method which will be made redundant with a better generated Python API...
def _get_cells_of_all_known_types(nml_doc):
    
    all_cells_known = []
    all_cells_known.extend(nml_doc.cells)
    all_cells_known.extend(nml_doc.izhikevich_cells)
    all_cells_known.extend(nml_doc.izhikevich2007_cells)
    all_cells_known.extend(nml_doc.iaf_cells)
    all_cells_known.extend(nml_doc.iaf_ref_cells)
    
    return all_cells_known

# Helper method which will be made redundant with a better generated Python API...
def _get_channels_of_all_known_types(nml_doc):
    
    all_channels = []
    all_channels.extend(nml_doc.ion_channel)
    all_channels.extend(nml_doc.ion_channel_hhs)
    all_channels.extend(nml_doc.ion_channel_kses)
    all_channels.extend(nml_doc.decaying_pool_concentration_models)
    all_channels.extend(nml_doc.fixed_factor_concentration_models)
    all_channels.extend(nml_doc.ComponentType)
    
    return all_channels

# Helper method which will be made redundant with a better generated Python API...
def _add_to_neuroml_doc(nml_doc, element):
    
    if isinstance(element, neuroml.Cell):
        nml_doc.cells.append(element)
    elif isinstance(element, neuroml.IzhikevichCell):
        nml_doc.izhikevich_cells.append(element)
    elif isinstance(element, neuroml.Izhikevich2007Cell):
        nml_doc.izhikevich2007_cells.append(element)
    elif isinstance(element, neuroml.IafRefCell):
        nml_doc.iaf_ref_cells.append(element)
    elif isinstance(element, neuroml.IafCell):
        nml_doc.iaf_cells.append(element)
        
    elif isinstance(element, neuroml.IonChannelKS):
        nml_doc.ion_channel_kss.append(element)
    elif isinstance(element, neuroml.IonChannelHH):
        nml_doc.ion_channel_hhs.append(element)
    elif isinstance(element, neuroml.IonChannel):
        nml_doc.ion_channel.append(element)
    elif isinstance(element, neuroml.FixedFactorConcentrationModel):
        nml_doc.fixed_factor_concentration_models.append(element)
    elif isinstance(element, neuroml.ComponentType):
        nml_doc.ComponentType.append(element)
        
    
def _copy_to_dir_for_model(nml_doc,file_name):
    
    dir_for_model = nml_doc.id
    if not os.path.isdir(dir_for_model):
        os.mkdir(dir_for_model)
    
    shutil.copy(file_name, dir_for_model)
    
##########################################################################################   
def copy_nml2_source(dir_to_project_nml2,
                      primary_nml2_dir,
                      electrical_synapse_tags,
                      chemical_synapse_tags,
                      extra_channel_tags=[]):
    
    full_path_to_synapses=os.path.join(dir_to_project_nml2,"synapses")
    
    if not os.path.exists(full_path_to_synapses):
    
       os.makedirs(full_path_to_synapses)
       
    full_path_to_gap_junctions=os.path.join(dir_to_project_nml2,"gapJunctions")
    
    if not os.path.exists(full_path_to_gap_junctions):
    
       os.makedirs(full_path_to_gap_junctions)
       
    full_path_to_channels=os.path.join(dir_to_project_nml2,"channels")
    
    if not os.path.exists(full_path_to_channels):
    
       os.makedirs(full_path_to_channels)
       
    full_path_to_cells=os.path.join(dir_to_project_nml2,"cells")
   
    if not os.path.exists(full_path_to_cells):
   
       os.makedirs(full_path_to_cells)
       
    opencortex.print_comment_v("Will be copying cell component files from %s to %s"%(primary_nml2_dir,full_path_to_cells) )
    
    opencortex.print_comment_v("Will be copying channel component files from %s to %s"%(primary_nml2_dir,full_path_to_channels) )
     
    opencortex.print_comment_v("Will be copying synapse component files from %s to %s"%(primary_nml2_dir,full_path_to_synapses) )
    
    opencortex.print_comment_v("Will be copying gap junction component files from %s to %s"%(primary_nml2_dir,full_path_to_gap_junctions))
     
    src_files=os.listdir(primary_nml2_dir)
   
    for file_name in src_files:
   
       full_file_name = os.path.join(primary_nml2_dir,file_name)
   
       if '.cell.nml' in file_name:
        
          shutil.copy(full_file_name,full_path_to_cells)
          
          continue
          
       for elect_tag in electrical_synapse_tags:
       
           if elect_tag in file_name:
           
              shutil.copy(full_file_name,full_path_to_gap_junctions)
              
              continue
              
       for chem_tag in chemical_synapse_tags:
       
           if chem_tag in file_name:
           
              shutil.copy(full_file_name,full_path_to_synapses)
              
              continue
              
       if '.channel.nml' in file_name:
       
          shutil.copy(full_file_name,full_path_to_channels)
          
          continue
       
       if extra_channel_tags !=[]:  
       
          for channel_tag in extra_channel_tags:
       
              if channel_tag in file_name:
           
                 shutil.copy(full_file_name,full_path_to_channels)
              
                 
                 
#########################################################################################
def add_cell_and_channels(nml_doc,cell_nml2_path, cell_id):
    
    nml2_doc_cell = pynml.read_neuroml2_file(cell_nml2_path, include_includes=False)
    
    for cell in _get_cells_of_all_known_types(nml2_doc_cell):
        if cell.id == cell_id:
            all_cells[cell_id] = cell
            
            _copy_to_dir_for_model(nml_doc,cell_nml2_path)
            new_file = '%s/%s.cell.nml'%(nml_doc.id,cell_id)
            nml_doc.includes.append(neuroml.IncludeType(new_file)) 
            if not new_file in all_included_files:
                all_included_files.append(new_file)
            
            for included in nml2_doc_cell.includes:
                
                if '../channels/' in included.href:
                
                   path_included=included.href.split("/")
                   
                   channel_file=path_included[-1]
                   
                   old_loc='../../channels/%s'%channel_file
                   
                elif '..\channels\'' in included.href:
                
                   path_included=included.href.split("\"")
                   
                   channel_file=path_included[-1]
                   
                   old_loc="..\..\channels\%s'"%channel_file
                
                else:
                
                   channel_file=included.href
                
                   old_loc = '%s/%s'%(os.path.dirname(os.path.abspath(cell_nml2_path)), channel_file)
                
                _copy_to_dir_for_model(nml_doc,old_loc)
                new_loc = '%s/%s'%(nml_doc.id,channel_file)
                nml_doc.includes.append(neuroml.IncludeType(new_loc))
                if not new_loc in all_included_files:
                    all_included_files.append(new_loc)

#######################################################################################################################################                    
def remove_component_dirs(dir_to_project_nml2,
                          list_of_cell_ids,
                          extra_channel_tags=None):
                            
    list_of_cell_file_names=[]
    
    for cell_id in list_of_cell_ids:
    
        list_of_cell_file_names.append(cell_id+".cell.nml")
           
    for cell_file_name in list_of_cell_file_names:

        full_path_to_cell=os.path.join(dir_to_project_nml2,cell_file_name)
    
        nml2_doc_cell=pynml.read_neuroml2_file(full_path_to_cell,include_includes=False)
        
        for included in nml2_doc_cell.includes:
        
            if '.channel.nml' in included.href:
               
               if '../channels/' in included.href:
                  
                  split_href=included.href.split("/")
                     
                  included.href=split_href[-1]
                     
                  continue
                  
               if '..\channels\'' in included.href:
               
                  split_href=included.href.split("\'")
                  
                  included.href=split_href[-1]
                  
                  continue
                     
            else:
            
               if extra_channel_tags != None:
               
                  for channel_tag in included.href:
                  
                         if channel_tag in included.href:
                      
                            if '../channels/' in included.href:
                  
                               split_href=included.href.split("/")
                     
                               included.href=split_href[-1]
                     
                               break
                  
                            if '..\channels\'' in included.href:
               
                               split_href=included.href.split("\'")
                  
                               included.href=split_href[-1]
                  
                               break
           
        pynml.write_neuroml2_file(nml2_doc_cell,full_path_to_cell)  
                                 
#######################################################################################################################################
def add_synapses(nml_doc,nml2_path,synapse_list,synapse_tag=True):

    for synapse in synapse_list: 
   
        if synapse_tag:
       
           _copy_to_dir_for_model(nml_doc,os.path.join(nml2_path,"%s.synapse.nml"%synapse))
          
           new_file = '%s/%s.synapse.nml'%(nml_doc.id,synapse)
          
        else:
       
           _copy_to_dir_for_model(nml_doc,os.path.join(nml2_path,"%s.nml"%synapse))
          
           new_file = '%s/%s.nml'%(nml_doc.id,synapse)
          
        nml_doc.includes.append(neuroml.IncludeType(new_file)) 
            
#########################################
                    
    
def add_exp_two_syn(nml_doc, id, gbase, erev, tau_rise, tau_decay):
    # Define synapse
    syn0 = neuroml.ExpTwoSynapse(id=id, gbase=gbase,
                                 erev=erev,
                                 tau_rise=tau_rise,
                                 tau_decay=tau_decay)
                                 
    nml_doc.exp_two_synapses.append(syn0)
    
    return syn0

def add_poisson_firing_synapse(nml_doc, id, average_rate, synapse_id):

    pfs = neuroml.PoissonFiringSynapse(id=id,
                                       average_rate=average_rate,
                                       synapse=synapse_id, 
                                       spike_target="./%s"%synapse_id)
                                       
    nml_doc.poisson_firing_synapses.append(pfs)

    return pfs
    
    
#########################################################################
def add_transient_poisson_firing_synapse(nml_doc, id, average_rate,delay,duration, synapse_id):

    pfs = neuroml.TransientPoissonFiringSynapse(id=id,
                                       average_rate=average_rate,
                                       delay=delay,
                                       duration=duration,
                                       synapse=synapse_id, 
                                       spike_target="./%s"%synapse_id)
                                       
    nml_doc.transient_poisson_firing_synapses.append(pfs)

    return pfs
################################################################################    

def add_pulse_generator(nml_doc, id, delay, duration, amplitude):

    pg = neuroml.PulseGenerator(id=id,
                                delay=delay,
                                duration=duration,
                                amplitude=amplitude)
                                       
    nml_doc.pulse_generators.append(pg)

    return pg
    
    
def add_single_cell_population(net, pop_id, cell_id, x=0, y=0, z=0, color=None):
    
    pop = neuroml.Population(id=pop_id, component=cell_id, type="populationList", size=1)
    if color is not None:
        pop.properties.append(Property("color",color))
    net.populations.append(pop)

    inst = neuroml.Instance(id=0)
    pop.instances.append(inst)
    inst.location = neuroml.Location(x=x, y=y, z=z)

    return pop
    
    
##############################################################################################################################    
def add_population_in_rectangular_region(net, pop_id, cell_id, size, x_min, y_min, z_min, x_size, y_size, z_size,storeSoma=False, color=None):
    
    pop = neuroml.Population(id=pop_id, component=cell_id, type="populationList", size=size)
    if color is not None:
        pop.properties.append(Property("color",color))
    net.populations.append(pop)
    
    if storeSoma:
       cellPositions=np.zeros([size,3])
    
       
    for i in range(0, size) :
            index = i
            inst = neuroml.Instance(id=index)
            pop.instances.append(inst)
            X=x_min +(x_size)*random.random()
            Y=y_min +(y_size)*random.random()
            Z=z_min +(z_size)*random.random()
            inst.location = neuroml.Location(x=str(X), y=str(Y), z=str(Z) )
            if storeSoma==True:
               cellPositions[i,0]=X
               cellPositions[i,1]=Y
               cellPositions[i,2]=Z
            
    
    if storeSoma:
       return pop, cellPositions
    else:
       return pop

###############################################################################################

def add_inputs_to_population(net, id, population, input_comp_id, all_cells=False, only_cells=None):
    
    if all_cells and only_cells is not None:
        opencortex.print_comment_v("Error! Method opencortex.build.%s() called with both arguments all_cells and only_cells set!"%sys._getframe().f_code.co_name)
        exit(-1)
        
    cell_ids = []
    
    if all_cells:
        cell_ids = range(population.size)
    if only_cells is not None:
        if only_cells == []:
            return
        cell_ids = only_cells
        
    input_list = neuroml.InputList(id=id,
                         component=input_comp_id,
                         populations=population.id)
    count = 0
    for cell_id in cell_ids:
        input = neuroml.Input(id=count, 
                      target="../%s/%i/%s"%(population.id, cell_id, population.component), 
                      destination="synapses")  
        input_list.input.append(input)
        count+=1
        
                         
    net.input_lists.append(input_list)
    
    return input_list
########################################################################################################    

def add_advanced_inputs_to_population(net, 
                                      id, 
                                      population, 
                                      input_id_list, 
                                      seg_length_dict,
                                      subset_dict,
                                      universal_target_segment,
                                      universal_fraction_along, 
                                      all_cells=False, 
                                      only_cells=None):

    ''' This method distributes the poisson input synapses on the specific segment groups of target cells. Input arguments to this method:
    
    net- libNeuroML network object;
    
    id - unique string that tags the input group created by the method;
    
    population - libNeuroML population object;
    
    input_id_list - this is a list that stores lists of poisson synapse ids or pulse generator ids; 
    if len(input_id_list)== (num of target cells) then each target cell, specified by only_cells or all_cells, has a unique list input components;
    if len(input_id_list != num, then add_advanced_inputs_to_population assumes that all cells share the same list of input components and thus uses input_id_list[0].
    Note that all of the input components (e.g. differing in delays) per given list of input components are mapped on the same membrane point on the target segment of a given cell.
    
    seg_length_dict - a dictionary whose keys are the ids of target segment groups and the values are the segment length dictionaries in the format returned by make_target_dict(); 
    
    subset_dict - a dictionary whose keys are the ids of target segment groups and the corresponding dictionary values define the desired number of synaptic connections per target    segment group per each postsynaptic cell;
    
    universal_target_segment - this should be set to None if subset_dict and seg_length_dict are used; alternatively, universal_target_segment specifies a single target segment on
    all of the target cells for all input components; then seg_length_dict and subset_dict must be set to None.
    
    universal_fraction_along - this should be set to None if subset_dict and seg_length_dict are used; alternatively, universal_target_fraction specifies a single value of 
    fraction along on all of the target segments for all target cells and all input components; then seg_length_dict and subset_dict must bet set to None;
    
    all_cells - default value is set to False; if all_cells==True then all cells in a given population will receive the inputs;
    
    only_cells - optional variable which stores the list of ids of specific target cells; cannot be set together with all_cells. '''
    
    if all_cells and only_cells is not None:
        opencortex.print_comment_v("Error! Method opencortex.build.%s() called with both arguments all_cells and only_cells set!"%sys._getframe().f_code.co_name)
        exit(-1)
        
    cell_ids = []
    
    if all_cells:
        cell_ids = range(population.size)
    if only_cells is not None:
        if only_cells == []:
            return
        cell_ids = only_cells
        
    input_list_array_final=[]
    
    input_counters_final=[]
    
    for input_cell in range(0,len(input_id_list) ):
    
        input_list_array=[]
        
        input_counters=[]
    
        for input_index in range(0,len(input_id_list[input_cell]) ):
    
            input_list = neuroml.InputList(id=id+"_%d_%d"%(input_cell,input_index),
                                           component=input_id_list[input_cell][input_index],
                                           populations=population.id)
                                       
                                       
            input_list_array.append(input_list)
            
            input_counters.append(0)
            
        input_list_array_final.append(input_list_array)
        
        input_counters_final.append(input_counters)
        
    cell_counter=0
        
    for cell_id in cell_ids:
    
        if len(input_id_list)==len(cell_ids):
               
           cell_index=cell_counter
           
        else:
        
           cell_index=0
    
        if seg_length_dict!=None and subset_dict !=None and universal_target_segment==None and universal_fraction_along==None:
            
           target_seg_array, target_fractions=get_target_segments(seg_length_dict,subset_dict)
           
           for target_point in range(0,len(target_seg_array)):
           
               for input_index in range(0,len(input_list_array_final[cell_index]) ):
            
                    input = neuroml.Input(id=input_counters_final[cell_index][input_index], 
                                      target="../%s/%i/%s"%(population.id, cell_id, population.component), 
                                      destination="synapses",segment_id="%d"%target_seg_array[target_point],fraction_along="%f"%target_fractions[target_point])
                                        
                    input_list_array_final[cell_index][input_index].input.append(input)
                    
                    input_counters_final[cell_index][input_index]+=1
           
        else:
        
           for input_index in range(0,len(input_list_array_final[cell_index])):
            
               input = neuroml.Input(id=input_counters_final[cell_index][input_index], 
                                     target="../%s/%i/%s"%(population.id, cell_id, population.component), 
                                     destination="synapses",segment_id="%d"%universal_target_segment,fraction_along="%f"%universal_fraction_along)
                                        
               input_list_array_final[cell_index][input_index].input.append(input)
               
               input_counters_final[cell_index][input_index]+=1
        
        cell_counter+=1
               
    for input_cell in range(0,len(input_list_array_final)):
                
        for input_index in range(0,len(input_list_array_final[input_cell]) ):
        
            net.input_lists.append(input_list_array_final[input_cell][input_index])
        
        
    return input_list_array_final
    
#######################################################################################################
def generate_network(reference, seed=1234, temperature='32degC'):

    del all_included_files[:]
    all_cells.clear()
    
    nml_doc = neuroml.NeuroMLDocument(id='%s'%reference)
    
    random.seed(seed)
    
    nml_doc.properties.append(neuroml.Property("Python random seed",seed))
    
    # Create network
    network = neuroml.Network(id='%s'%reference, type='networkWithTemperature', temperature=temperature)
    nml_doc.networks.append(network)

    opencortex.print_comment_v("Created NeuroMLDocument containing a network with id: %s"%reference)
    
    return nml_doc, network


def save_network(nml_doc, nml_file_name, validate=True, comment=True, format='xml'):

    info = "\n\nThis NeuroML 2 file was generated by OpenCortex v%s using: \n"%(opencortex.__version__)
    info += "    libNeuroML v%s\n"%(neuroml.__version__)
    info += "    pyNeuroML v%s\n\n    "%(pyneuroml.__version__)
    
    if nml_doc.notes:
        nml_doc.notes += info
    else:
        nml_doc.notes = info
    
    if format == 'xml':
        writers.NeuroMLWriter.write(nml_doc, nml_file_name)
    elif format == 'hdf5':
        writers.NeuroMLHdf5Writer.write(nml_doc, nml_file_name)
    
    opencortex.print_comment_v("Saved NeuroML with id: %s to %s"%(nml_doc.id, nml_file_name))
    
    if validate:
        from pyneuroml.pynml import validate_neuroml2

        passed = validate_neuroml2(nml_file_name)
        
        if passed:
            opencortex.print_comment_v("Generated NeuroML file is valid")
        else:
            opencortex.print_comment_v("Generated NeuroML file is NOT valid!")
            
            
def generate_lems_simulation(nml_doc, 
                             network, 
                             nml_file_name, 
                             duration, 
                             dt, 
                             target_dir = '.',
                             include_extra_lems_files = [],
                             gen_plots_for_all_v = True,
                             plot_all_segments = False,
                             gen_plots_for_quantities = {},   #  Dict with displays vs lists of quantity paths
                             gen_plots_for_only_populations = [],   #  List of populations, all pops if = []
                             gen_saves_for_all_v = True,
                             save_all_segments = False,
                             gen_saves_for_only_populations = [],  #  List of populations, all pops if = []
                             gen_saves_for_quantities = {},   #  Dict with file names vs lists of quantity paths
                             seed=12345):
                                 
    lems_file_name = "LEMS_%s.xml"%network.id
    
    include_extra_lems_files.extend(all_included_files)
    
    pyneuroml.lems.generate_lems_file_for_neuroml("Sim_%s"%network.id, 
                                   nml_file_name, 
                                   network.id, 
                                   duration, 
                                   dt, 
                                   lems_file_name,
                                   target_dir,
                                   include_extra_files = include_extra_lems_files,
                                   gen_plots_for_all_v = gen_plots_for_all_v,
                                   plot_all_segments = plot_all_segments,
                                   gen_plots_for_quantities = gen_plots_for_quantities, 
                                   gen_plots_for_only_populations = gen_plots_for_only_populations,  
                                   gen_saves_for_all_v = gen_saves_for_all_v,
                                   save_all_segments = save_all_segments,
                                   gen_saves_for_only_populations = gen_saves_for_only_populations,
                                   gen_saves_for_quantities = gen_saves_for_quantities,
                                   seed=seed)
                                   
    del include_extra_lems_files[:]

    return lems_file_name
    
def simulate_network(lems_file_name,
                     simulator,
                     max_memory='400M',
                     nogui=True,
                     load_saved_data=False,
                     plot=False,
                     verbose=True):
                     
    
    if simulator=="jNeuroML":
       results = pynml.run_lems_with_jneuroml(lems_file_name,max_memory=max_memory,nogui=nogui,load_saved_data=load_saved_data,plot=plot,verbose=verbose)
    if simulator=="jNeuroML_NEURON":
       results = pynml.run_lems_with_jneuroml_neuron(lems_file_name,max_memory=max_memory, nogui=nogui, load_saved_data=load_saved_data, plot=plot,verbose=verbose)
    
    
    
    
        

