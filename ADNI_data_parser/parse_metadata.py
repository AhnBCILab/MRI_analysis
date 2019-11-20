from parse_xml import XMLParser

import os
import glob

xml_parser = XMLParser()


metadata_dir = "/home/mri-any/datasets/adni_data/metaFiles"
output_file = "metadata_output.csv"

xml_file = metadata_dir + "/*.xml"
xml_path = glob.glob(xml_file)

output_file = open(output_file, "w")
is_header = True
for xml in xml_path:
	xml_header, xml_parsed = xml_parser.parse(xml)
	
	if is_header:
		output_file.write(",".join(xml_header))
		output_file.write("\n")
		is_header = not is_header
	
	output_file.write(",".join(xml_parsed))
	output_file.write("\n")

output_file.close()

