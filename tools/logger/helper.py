import re


def tokenize(filename: str):
	"""Helper to sort the log files alphanumerically.

	Args:
		filename (str): Name of run.
	"""
	digits = re.compile(r"(\d+)")
	return tuple(
		int(token) if match else token
		for token, match in (
			(fragment, digits.search(fragment))
			for fragment in digits.split(filename)
		)
	)
