class Parser:
	def parse(self, source_file):
		raise NotImplementedError
	
	def write_as_csv(self, output_file, content, header=None):
		import csv
		with open(output_file, "w") as f:
			csv_writer = csv.writer(output_file)
			if header:
				csv_writer.writerow(header)
			csv_writer.writerow(content)
	
	def read_from_csv(self, input_file):
		import csv
		result = []
		with open(input_file, "r") as f:
			reader_obj = csv.reader(input_file)
			for row in reader_obj:
				result.append(row)
		return result
