import numpy as np
from PIL import Image
import os, re, itertools
import pca, normalize

(TRAIN, VERIFY) = (.8, .1)

root = 'J:\\Applications & Tools\\CAS-PEAL-R1\\FRONTAL\\'
cache = 'storage/cache/'
normal = 'NormalJPG\\'
aging = 'AgingJPG\\'
bg = 'BackgroundJPG\\'
glasses = 'AccessoryJPG\\'
eye_data = 'FaceFP_2.txt'

try:
	if (not os.path.exists('storage/data.npz')):
		raise EnvironmentError("No data array!")
	elif (not os.path.exists('storage/mask.png')):
		raise LookupError("No mask image!")
	elif (not os.path.exists('storage/datapairs.npz')):
		raise LookupError("No data pairs!")
	elif (not os.path.isdir(cache)):
		raise LookupError("No cached faces!")
	elif (not os.path.exists('storage/basis.npz')):
		raise LookupError("No pca basis!")
	mask_img = Image.open('storage/mask.png')
	mask_img = (np.asarray(mask_img, dtype='float64') - 127.5) / 127.5
	pca_basis_files = np.load('storage/basis.npz')
	pca_basis = {'evalues': pca_basis_files['evalues'], 
				 'efaces': pca_basis_files['efaces'],
				 'mu': pca_basis_files['mu']}
	pca_basis_files.close()
except LookupError as e:
	print(e)
	print("Mask image, data pairs, cached faces and/or PCA basis not found. "
			"Generate these before preparing any image pairs.")
except EnvironmentError as e:
	print(e)
	print("Data array not found. Generate it before generating the "
			"other prerequisites")

def fixline(add, line):
	line = line.strip()
	(filename, coords_string) = line.split('	 ')
	face = int(re.search('([0-9]{6})', filename).group(1))
	(xL, yL, xR, yR) = (int(i) for i in coords_string.split(' '))
	return (face, add + filename + '.jpg', (xL, yL), (xR, yR))

def split3_prop(arr):
	break1 = int(round(len(arr) * TRAIN))
	break2 = int(round(len(arr) * (TRAIN + VERIFY)))
	return [arr[:break1], arr[break1:break2], arr[break2:]]

def split3_breaks(arr, last_train, last_veri):
	[arr1, arr2, arr3] = [[], [], []]
	for i in range(len(arr)):
		face = arr[i][0]
		if (face <= last_train):
			arr1.append(arr[i])
		elif (face <= last_veri):
			arr2.append(arr[i])
		else:
			arr3.append(arr[i])
	return [arr1, arr2, arr3]

def create_dataset():
	(normal_d, bg_d, aging_d, glasses_d) = \
		(open(root + i + eye_data, 'r') for i in (normal, bg, aging, glasses))

	(normal_path_coords, bg_path_coords, 
			aging_path_coords, glasses_path_coords) = ([], [], [], [])

	for line in normal_d:
		normal_path_coords.append(fixline(normal, line))
	for line in aging_d:
		aging_path_coords.append(fixline(aging, line))
	for line in bg_d:
		if ('T0_BR' in line):
			bg_path_coords.append(fixline(bg, line))
	for line in glasses_d:
		if ('EN_A1' in line or 'EN_A2' in line):
			glasses_path_coords.append(fixline(glasses, line))

	normal_final = split3_prop(normal_path_coords)
	last_train = normal_final[0][-1][0]
	last_veri = normal_final[1][-1][0]
	bg_final = split3_breaks(bg_path_coords, last_train, last_veri)
	aging_final = split3_breaks(aging_path_coords, last_train, last_veri)
	glasses_final = split3_breaks(glasses_path_coords, last_train, last_veri)

	kwds = dict(zip(('normal', 'background', 'aging', 'glasses'), 
					(normal_final, bg_final, aging_final, glasses_final)))
	np.savez('storage/data.npz', **kwds)

	normal_d.close()
	bg_d.close()
	aging_d.close()
	glasses_d.close()

def bisect_left(a, x, lo=0, hi=None):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid][0] < x: lo = mid+1
        else: hi = mid
    return lo

def bisect_right(a, x, lo=0, hi=None):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x < a[mid][0]: hi = mid
        else: lo = mid+1
    return lo

def find_all(arr, x):
	return arr[bisect_left(arr, x):bisect_right(arr, x)]

def pset(arr): # Excluding empty set
	double_gen = (itertools.combinations(arr, r)
			for r in range(1, len(arr) + 1))
	return [item for sublist in double_gen for item in sublist]

def create_datapairs():
	files = np.load('storage/data.npz')
	datapairs = [[], [], []]
	for i in range(3):
		for (face, path, eL, eR) in files['normal'][i]:
			normalfaces = [(face, path, eL, eR)]
			normalfaces += find_all(files['background'][i], face)
			normalfaces += find_all(files['aging'][i], face)
			normal_pset = pset(normalfaces)
			glasses_pset = pset(find_all(files['glasses'][i], face))
			datapairs_s = list(itertools.product(normal_pset, normal_pset))
			datapairs_s += list(itertools.product(glasses_pset, normal_pset))
			datapairs[i] += datapairs_s
	kwds = dict(zip(('training', 'validation', 'testing'), datapairs))
	np.savez('storage/datapairs.npz', **kwds)
	files.close()

def create_cache():
	files = np.load('storage/data.npz')
	for collection in ['background', 'glasses','aging', 'normal']:
		for i in range(3):
			for (face, filepath, eyeLeft, eyeRight) in files[collection][i]:
				img = Image.open(root + filepath).convert('L')
				img = normalize.CropFace(img, eyeLeft, eyeRight)
				img.save(cache + filepath)
	files.close()

def prepare_imagepair(datapair):
	(in_data, out_data) = datapair
	in_img = np.zeros(
			(normalize.FINAL_HEIGHT, normalize.FINAL_WIDTH),
			 dtype='float32')
	out_img = np.zeros(
			(normalize.FINAL_HEIGHT, normalize.FINAL_WIDTH),
			 dtype='float32')
	for (_, filepath, eyeLeft, eyeRight) in in_data:
		img = Image.open(cache + filepath)
		img = np.asarray(img, dtype='float32') / (len(in_data))
		in_img += img
	for (_, filepath, eyeLeft, eyeRight) in out_data:
		img = Image.open(cache + filepath)
		img = np.asarray(img, dtype='float32') / (len(out_data))
		out_img += img
	in_pca = pca.proj_reconst(in_img, pca_basis['efaces'], pca_basis['mu'])
	return (np.asarray([
				(in_img - 127.5) / 127.5,
				(in_pca - 127.5) / 127.5, mask_img], dtype='float32'),
			np.asarray([(out_img - 127.5) / 127.5], dtype='float32'))

def encode_raw_input_image(raw_input_img):
	in_img = np.asarray(raw_input_img)
	in_pca = pca.proj_reconst(in_img, pca_basis['efaces'], pca_basis['mu'])
	return np.asarray([
				(in_img - 127.5) / 127.5,
				(in_pca - 127.5) / 127.5, mask_img], dtype='float32')

def decode_output_image(encoded_output_img):
	return Image.fromarray(encoded_output_img[0] * 127.5 + 127.5)
