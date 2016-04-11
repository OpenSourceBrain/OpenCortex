print("\n*********************************************************************************************")
print("          Please note that OpenCortex is in a very preliminary state ");
print("          and the API is subject to change without notice!  ")
print("*********************************************************************************************\n")

__version__ = '0.0.3'


verbose = False
    
def print_comment_v(text):
    print_comment(text, True)
    
    
def print_comment(text, print_it=verbose):
    prefix = "OpenCortex >>> "
    if not isinstance(text, str): text = text.decode('ascii')
    if print_it:
        
        print("%s%s"%(prefix, text.replace("\n", "\n"+prefix)))