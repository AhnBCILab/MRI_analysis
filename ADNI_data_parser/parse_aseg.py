from parser import Parser

class AsegParser(Parser):
	MARKER = 1
	KEY_NAME = 2
	REAL_DATA = -2
	VOLUME = 3
	STRUCT_NAME = 4

	def _set_variables(self, filename):
		self.source_file = filename

	def _parse(self):
		import re
		parser = re.compile("\S+")
		result = {}
		with open(self.source_file) as f:
			lines = f.readlines()
			for line in lines:
				elements = parser.findall(line)
				if len(elements) < 2:
					continue

				if elements[self.MARKER] == "Measure":
					result[elements[self.KEY_NAME].split(',')[0]] = float(elements[self.REAL_DATA].split(',')[0])
				elif elements[self.MARKER].isdigit():
					result[elements[self.STRUCT_NAME]] = float(elements[self.VOLUME])
		return result

	def parse(self, filename):
		self._set_variables(filename)
		result = self._parse()
		return result

