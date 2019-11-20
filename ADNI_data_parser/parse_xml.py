from parser import Parser

class XMLParser(Parser):
	def __init__(self):
		self.default_seg_metadata = {
			"Filename": self._get_filename,
			"SubjectID": 'subjectIdentifier',
			"Visit": 'visitIdentifier',
			"Group": 'reseaRchgroup',
			"Age": 'subjectAge',
			"Sex": 'subjectSex',
			"MMSE": { "name":'assessmentScore', "attrs":{'attribute': 'MMSCORE'}},
			"GDSCALE": { "name":'assessmentScore', "attrs":{'attribute': 'GDTOTAL'}},
			"CDR": { "name":'assessmentScore', "attrs":{'attribute': 'CDGLOBAL'}},
			"NPI-Q": { "name":'assessmentScore', "attrs":{'attribute': 'NPISCORE'}},
			"FAQ": { "name":'assesmentScore', "attrs":{'attribute': 'FAQTOTAL'}},
			"APOE A1": { "name":'subjectInfo', "attrs":{'item': 'APOE A1'}},
			"APOE A2": { "name":'subjectInfo', "attrs":{'item': 'APOE A2'}},
			'Study Identifer': { "name":'studyIdentifier' },
			"Weight": { "name":'weightKg' },
			"Series Identifer": 'seriesIdentifier',
			"Acqusisition Type": { "name":'protocol', "attrs":{'term': 'Acquisition Type'}},
			"Weighting": { "name":'protocol', "attrs":{'term': 'Weighting'}},
			"Pulse Sequence": { "name":'protocol', "attrs":{'term': 'Pulse Sequence'}},
			"Slice Thickness": { "name":'protocol', "attrs":{'term': 'Slice Thickness'}},
			"TE": { "name":'protocol', "attrs":{'term': 'TE'}},
			"TR": { "name":'protocol', "attrs":{'term': 'TR'}},
			"TI": { "name":'protocol', "attrs":{'term': 'TI'}},
			"Coil": { "name":'protocol', "attrs":{'term': 'Coil'}},
			"Flip Angle": { "name":'protocol', "attrs":{'term': 'Flip Angle'}},
			"Acquistion Plane": { "name":'protocol', "attrs":{'term': 'Acquisition Plane'}},
			"Matrix X": { "name":'protocol', "attrs":{'term': 'Matrix X'}},
			"Matrix Y": { "name":'protocol', "attrs":{'term': 'Matrix Y'}},
			"Matrix Z": { "name":'protocol', "attrs":{'term': 'Matrix Z'}},
			"Pixel Spacing X": { "name":'protocol', "attrs":{'term': 'Pixel Spacing X'}},
			"Pixel Spacing Y": { "name":'protocol', "attrs":{'term': 'Pixel Spacing Y'}},
			"Field Strength": {"name":'protocol', "attrs":{'term': 'Field Strength'}},
		}

	def _get_filename(self):
		import os
		return os.path.basename(self.source_file)

	def _set_variables(self, filename):
		self.source_file = filename
		self.seg_metadata = self.default_seg_metadata

	def _parse(self):
		from bs4 import BeautifulSoup
		output_dict = {}
		with open(self.source_file, "r") as f:
			src_content = f.read()		
			soup = BeautifulSoup(src_content, 'lxml-xml')
			for header in self.seg_metadata:
				seg_by = self.seg_metadata[header]
				if callable(seg_by):
					output_dict[header] = seg_by()
				else:
					seperated_data = None
					if isinstance(seg_by, dict):
						seperated_data = soup.find(**seg_by)
					else:
						seperated_data = soup.find(seg_by)
					output_dict[header] = seperated_data.get_text() if seperated_data else "None"
		return output_dict.values()

	def parse(self, source_file):
		self._set_variables(source_file)
		headers = list(self.seg_metadata.keys())
		result = list(self._parse())
		return (headers, result)

