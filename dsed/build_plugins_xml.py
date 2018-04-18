'''
Build a Plugins.xml file for use in Source
'''
import sys
import os

TEMPLATE='''<?xml version="1.0" encoding="utf-8"?>
<ArrayOfPlugin xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Plugin>
    <IsEnabled>true</IsEnabled>
    <Path>%s\Dynamic_SedNet.dll</Path>
  </Plugin>
  <Plugin>
    <IsEnabled>true</IsEnabled>
    <Path>%s\GBR_DynSed_Extension.dll</Path>
  </Plugin>
  <Plugin>
    <IsEnabled>true</IsEnabled>
    <Path>%s\ReefHydroCalModels.dll</Path>
  </Plugin>
  <Plugin>
    <IsEnabled>true</IsEnabled>
    <Path>%s\DERMTools.dll</Path>
  </Plugin>
'''

def build_plugins_xml(dest,src):
  txt = TEMPLATE%(src,src,src,src)
  # write file to dest
  dest_fn = os.path.join(dest,'Plugins.xml')
  f = open(dest_fn,'w')
  f.write(txt)
  f.close()

if __name__ == "__main__":
  dest = os.path.abspath(sys.argv[1])
  src = os.path.abspath(sys.argv[2])
  print("Building Plugins.xml in %s using plugins in %s"%(dest,src))
  build_plugins_xml(dest,src)