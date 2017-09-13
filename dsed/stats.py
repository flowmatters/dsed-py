"""
GBR Statistics Library...
"""

def nse(obs,pred):
	numerator = ((obs-pred)**2).sum()
	denominator = ((obs-obs.mean())**2).sum()
	return 1 - numerator/denominator
