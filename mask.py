import numpy as np
import math, png, argparse

# radial_sin: (float [0, inf), int) -> float [-1, 1]
# R: d and s are in pixels
# E: Gives the radial component of pixels distance d from the center
def radial_sin(d, s, offset):
	scaled = (d / s) % math.log(s, 1.5)
	cycles = 1.5**scaled - 1
	return 0.5 * math.sin (offset + math.pi * cycles) + \
		0.5 * math.sin(2 * math.pi * cycles)

# angular_sin: (float [0, inf), float [-pi, pi],
#			   float [0, inf), int, float [-pi, pi]) -> float [-1, 1])
# R: d, max_d and s are in pixels; offset and t is in radians
# E: Gives the angular component of pixels at polar coordinates (d, t)
def angular_sin(d, t, max_d, s, offset):
	norm_d = max_d
	if (d != 0):
		norm_d = max_d / (s ** math.floor(math.log(max_d/d, s)))
	big_cycle_num = math.ceil(0.5 * math.pi * norm_d / s)
	small_cycle_num = 2 * round(0.5 * math.pi * norm_d / s) + 1
	big_unit = math.pi / big_cycle_num
	small_unit = math.pi / small_cycle_num
	return 0.5 * (math.sin(offset + math.pi * t / big_unit) + \
				  math.sin(math.pi * t / small_unit))

# main: (int, int, int, string)
# R: height, width are positive, scope > 1, output ends in '.png'
# E: Creates the position mask as an image file
def generate(width, height, s, output):
	canvas = np.zeros((height, width), dtype='float64')
	max_d = math.sqrt(width**2 + height**2) / 2
	offset_angular = 0
	if (width >= height):
		offset_angular = math.pi / 4
	for (i, j), _ in np.ndenumerate(canvas):
		y = height // 2 - i
		x = j - width // 2
		d = math.sqrt(x**2 + y**2)
		t = math.atan2(y, x)
		canvas[i,j] = (255 / 4) * \
			(2 + radial_sin(d, s, t) + angular_sin (
				d, t, max_d, s, offset_angular))
	f = open(output, 'wb')
	w = png.Writer(width, height, greyscale=True)
	w.write(f, canvas)
	f.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Creates a position mask for convolutional neural networks")
	parser.add_argument('-o', '--output', help="Name of output file (png)",
							default='out.png')
	parser.add_argument('width', help="The width of the canvas", type=int)
	parser.add_argument('height', help="The height of the canvas", type=int)
	parser.add_argument('scope', help="The scope size of the kernel", type=int)
	args = parser.parse_args()
	if (args.width <= 0 or args.height <= 0 or args.scope <= 1):
		print("Arguments are too small.\nUse positive canvas dimensions"
					  " and scope size greater than 1")
	else:
		generate(args.width, args.height, args.scope, args.output)
