import os

class FsChecker:
	@staticmethod
	def is_fine(directory):
		result = True
		filename = ""
		
		log_file_path = os.path.join(directory, "scripts", "recon-all.log")
		with open(log_file_path) as f:
			lines = f.readlines()
			if lines[-1].split(' ')[0] != "recon-all":
				result = False
	
		return result
