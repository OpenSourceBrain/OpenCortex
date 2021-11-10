###############################################################
###
### Note: OpenCortex is under active development, the API is subject to change without notice!!
###
### Authors: Padraig Gleeson, Rokas Stanislovas
###
### This software has been funded by the Wellcome Trust, as well as a GSoC 2016 project
### on Cortical Network develoment
###
##############################################################

print("\n*********************************************************************************************")
print("          Please note that OpenCortex is in a preliminary state ");
print("          and the API is subject to change without notice!  ")
print("*********************************************************************************************\n")

__version__ = '0.1.18'


verbose = False

def print_comment_v(text):
    """
    Always print the comment
    """
    print_comment(text, True)


def print_comment(text, print_it=verbose):
    """
    Print a comment only if print_it == True
    """
    prefix = "OpenCortex >>> "
    if not isinstance(text, str): text = text.decode('ascii')
    if print_it:

        print("%s%s"%(prefix, text.replace("\n", "\n"+prefix)))

def set_verbose(value=True):
    global verbose
    verbose = value
