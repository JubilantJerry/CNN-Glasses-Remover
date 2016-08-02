import numpy as np
from PIL import Image
import cv2, normalize

def generate_eface_basis(db_path, st_path, num_efaces, faces_paths_coords):
	if (db_path and db_path[-1] not in ['/', '\\']):
			db_path += '\\'
	if (st_path and st_path[-1] not in ['/', '\\']):
			st_path += '\\'
	n = len(faces_paths_coords)
	faces_flat = np.zeros((n, normalize.FINAL_HEIGHT * normalize.FINAL_WIDTH))
	for i in range(len(faces_paths_coords)):
		(_, face_path, eye_left, eye_right) = faces_paths_coords[i]
		face_raw = Image.open(db_path + face_path).convert('L')
		if (eye_left == None or eye_right == None):
			normalized = normalize.NormalizeFace(face_raw)
		else:
			normalized = normalize.CropFace(face_raw, eye_left, eye_right)
		faces_flat[i] = np.array(normalized).flatten()
	(evalues, efaces, mu) = pca(faces_flat, num_efaces)
	efaces = efaces.transpose(1,0)
	efaces = efaces.reshape((-1, normalize.FINAL_HEIGHT, normalize.FINAL_WIDTH))
	mu = mu.reshape((normalize.FINAL_HEIGHT, normalize.FINAL_WIDTH))
	kwds = dict(zip(('evalues', 'efaces', 'mu'), (evalues, efaces, mu)))
	np.savez(st_path + 'basis.npz', **kwds)

# From ByteFish
def pca(X, num_components=0):
	[n,d] = X.shape
	if (num_components <= 0) or (num_components>n):
		num_components = n
	mu = X.mean(axis=0)
	X = X - mu
	if n>d:
		C = np.dot(X.T,X)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
	else:
		C = np.dot(X,X.T)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
		eigenvectors = np.dot(X.T,eigenvectors)
		for i in range(n):
			eigenvectors[:,i] = \
					eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
	# or simply perform an economy size decomposition
	# eigenvectors, eigenvalues, 
	# 	variance = np.linalg.svd(X.T, full_matrices=False)
	# sort eigenvectors descending by their eigenvalue
	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]
	# select only num_components
	eigenvalues = eigenvalues[0:num_components].copy()
	eigenvectors = eigenvectors[:,0:num_components].copy()
	return (eigenvalues, eigenvectors, mu)

def project(face, efaces, mu):
	X = (face - mu).flatten()
	Q = efaces.reshape(-1, normalize.FINAL_HEIGHT * normalize.FINAL_WIDTH)
	return np.dot(X, Q.T)

def reconstruct(projection, efaces, mu):
	Q = efaces.reshape(-1, normalize.FINAL_HEIGHT * normalize.FINAL_WIDTH)
	X = np.dot(projection, Q)
	face = X.reshape(normalize.FINAL_HEIGHT, normalize.FINAL_WIDTH) + mu
	return face

def proj_reconst(face, efaces, mu):
	return reconstruct(project(face, efaces, mu), efaces, mu)

def make_valid_image(face):
	m = face.min()
	if (m < 0):
		face -= m
	m = face.max()
	if (m > 255):
		face *= 255 / m
