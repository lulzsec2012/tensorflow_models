#!/usr/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : (>>>FILE<<<)
## Authors    : (>>>AUTHOR<<<)
## Create Time: (>>>ISO_DATE<<<):(>>>TIME<<<)
## Description:
## 
##
import os
import sys

__DEBUG__ON__="NO"

def __DEBUG__(msg):
    if __DEBUG__ON__ == "YES":
        msg="__DEBUG__:"+msg+"\n"    
        sys.stdout.write(msg)

def usage():
    usage_info=\
"""
Usage: %s [option]
"""%(os.path.basename(__file__))
    sys.stdout.write(usage_info)
    sys.exit()

if __name__=="__main__":
    if len(sys.argv) != 2:
        usage()
    elif(sys.argv[1]=="--help" or sys.argv[1]=="-h" ):
        usage()
    else:
        sys.stdout.write("Hello Python!")
