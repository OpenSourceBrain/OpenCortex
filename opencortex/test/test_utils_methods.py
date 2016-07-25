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
              

      
             
          
             
             
             
             
             
             
             
             
          
     
          
          
          
          
          
          
          
          
        
          
          
          
         

