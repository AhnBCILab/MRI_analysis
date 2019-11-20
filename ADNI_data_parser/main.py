from parse_xml import XMLParser
from parse_aseg import AsegParser
from recon_checker import FsChecker

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", dest="input_file", metavar="FILE",
				  help="Input file which contain list of "
				       "freesurfer subjects")
parser.add_option("-o", "--output", dest="output_file", metavar="FILE",
				  help="Output file which contain the results of "
				       "seperated data as csv format")
parser.add_option("-e", "--error-output", dest="error_output_file", metavar="FILE",
				   help="Output file which contain the results of "
						"error file list (in freesurfer)")
parser.add_option("-m", "--metadata_dir", dest="metadata_dir", metavar="DIR",
				   help="Directory which contains metadata file list")
parser.add_option("-v", "--verbose", action="store_true", default=False,
				   help="Print current checking files")

def check_args():
	import glob
	import os

	(options, args) = parser.parse_args()
	input_file = options.input_file
	output_file = options.output_file
	error_file = options.error_output_file
	metadata_dir = options.metadata_dir
	is_verbose = options.verbose

	input_files = []
	if input_file is None:
		sub_path = os.path.expandvars("$SUBJECTS_DIR")
		input_files = glob.glob(sub_path + "/*.nii")
		print("Using %s to input file directory" % sub_path)
	else:
		with open(input_file) as f:
			line = list(l.strip() for l in f.readlines())
			input_files = line

	if output_file is None:
		import sys
		print("Please input the output file using -o option")
		sys.exit()

	return (input_files, output_file, error_file, metadata_dir, is_verbose)

def main():
	import os
	import re
	import glob
	import datetime

	input_files, output_file, error_file, metadata_dir, is_verbose = check_args()
	file_list = []
	error_list = []

	file_list = input_files
	
	xml_parser = XMLParser()
	aseg_parser = AsegParser()
	is_header = True

	output_file = open(output_file, "w")
	error_file = open(error_file, "w")

	xml_before = []
	for index, filename in enumerate(file_list):
		if is_verbose:
			print("%4d Checking %s" % (index, filename))
		if not FsChecker.is_fine(filename):
			error_list.append(filename)
			continue
		
		image_id = re.split('[._]', filename)[-2]
		xml_file = metadata_dir + "/*" + image_id + ".xml"
		xml_path = glob.glob(xml_file)
		aseg_path = os.path.join(filename, "stats", "aseg.stats")
		
		aseg_parsed = aseg_parser.parse(aseg_path)
		xml_header = xml_parsed = []
		if len(xml_path) == 0:
			print("There is no metadata xml file for %s" % xml_file)
			output_file.write(os.path.basename(xml_file) + ",")
			output_file.write(",".join(["None" for i in xml_before][:-1]))
		else:
			xml_header, xml_parsed = xml_parser.parse(xml_path[0])
			xml_before = xml_parsed
		
		if is_header:
			output_file.write(",".join(xml_header))
			output_file.write(",")
			output_file.write(",".join(aseg_parsed.keys()))
			output_file.write("\n")
			is_header = not is_header

		output_file.write(",".join(xml_parsed))
		output_file.write(",")
		output_file.write(",".join(map(str, aseg_parsed.values())))
		output_file.write("\n")
	
	error_file.write("\n".join(error_list))
	error_file.close()
	output_file.close()

	print("Parsing is ended with %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	
if __name__ == "__main__":
	main()
