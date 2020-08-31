/*
 * draw_bitmap.cpp
 *
 *  Created on: Jun 7, 2020
 *      Author: betten
 */


#include "foundations.h"
#include "EasyBMP.h"

using namespace std;


namespace orbiter {
namespace foundations {


std::vector<int> get_color(int bit_depth, int max_value, int loopCount, int f_invert_colors, int verbose_level);
void fillBitmap(BMP &image, int i, int j, std::vector<int> color);





void draw_bitmap(std::string &fname, int *M, int m, int n,
		int f_partition, int part_width,
		int nb_row_parts, int *Row_part, int nb_col_parts, int *Col_part,
		int f_box_width, int box_width,
		int f_invert_colors, int bit_depth,
		int verbose_level)
// bit_depth should be either 8 or 24.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "draw_bitmap" << endl;
	}

	int i;
	int max_value;

	max_value = int_vec_maximum(M, m * n);
	cout << "max_value=" << max_value << endl;

	//max_value += 5;
	//cout << "max_value after adjustment=" << max_value << endl;


	string fname_out;

	fname_out.assign(fname);
	replace_extension_with(fname_out, "_draw.bmp");

	//int bit_depth = 8;

	BMP image;

	//int bit_depth = 24;


	if (max_value > 10000) {
		cout << "draw_bitmap max_value > 10000" << endl;
		exit(1);
	}
	for (i = max_value; i >= 0; i--) {
		std::vector<int> color = get_color(bit_depth, max_value, i, f_invert_colors, 1);

		cout << i << " : " << color[0] << "," << color[1] << "," << color[2] << endl;
		}


	int width, height;
	//int *Table;
	geometry_global Gg;

	width = n;
	height = m;

	cout << "width=" << width << endl;

	if (f_box_width) {
		image.SetSize(width * box_width, height * box_width);
	}
	else {
		image.SetSize(width, height);
	}

	image.SetBitDepth(bit_depth);

	int j, d;
	int N, N100, cnt;

	N = height * width;
	N100 = N / 100 + 1;

	cout << "N100=" << N100 << endl;

	cnt = 0;
	for (i = 0; i < height; i++) {



		for (j = 0; j < width; j++, cnt++) {


			if ((cnt % N100) == 0) {
				cout << "we are at " << ((double) cnt / (double) N) * 100. << " %" << endl;
			}
			d = M[i * width + j];
			//std::vector<int> color = getColor(M[idx_x * width + idx_z]);
			std::vector<int> color = get_color(bit_depth, max_value, d, f_invert_colors, 0);

			// Here the pixel is set on the image.
			if (f_box_width) {
				int I, J, u, v;

				I = i * box_width;
				J = j * box_width;
				for (u = 0; u < box_width; u++) {
					for (v = 0; v < box_width; v++) {
						fillBitmap(image, J + v, I + u, color);
					}
				}

			}
			else {
				fillBitmap(image, j, i, color);
			}
		}
	}
	if (f_partition) {

		cout << "drawing the partition" << endl;
		int i0, j0;
		int h, t, I, J;
		std::vector<int> color = get_color(bit_depth, max_value, 1, f_invert_colors, 0);

		// row partition:
		i0 = 0;
		for (h = 0; h <= nb_row_parts; h++) {
			for (t = 0; t < part_width; t++) {
				if (f_box_width) {
					for (j = 0; j < width * box_width; j++) {
						I = i0 * box_width;
						if (h == nb_row_parts) {
							fillBitmap(image, j, I - 1 - t, color);
						}
						else {
							fillBitmap(image, j, I + t, color);
						}
					}
				}
			}
			if (h < nb_row_parts) {
				i0 += Row_part[h];
			}
		}

		// col partition:
		j0 = 0;
		for (h = 0; h <= nb_col_parts; h++) {
			for (t = 0; t < part_width; t++) {
				if (f_box_width) {
					for (i = 0; i < height * box_width; i++) {
						J = j0 * box_width;
						if (h == nb_col_parts) {
							fillBitmap(image, J - 1 - t, i, color);
						}
						else {
							fillBitmap(image, J + t, i, color);
						}
					}
				}
			}
			if (h < nb_col_parts) {
				j0 += Col_part[h];
			}
		}
	}

	cout << "before writing the image to file as " << fname_out << endl;

	  image.WriteToFile(fname_out.c_str());

	  std::cout << "Written file " << fname_out << std::endl;
	  {
		  file_io Fio;
		  cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;
	  }

}

std::vector<int> get_color(int bit_depth, int max_value, int loopCount, int f_invert_colors, int verbose_level)
{
	int f_v = (verbose_level>= 1);
	int r, g, b;
#if 0
		Black	#000000	(0,0,0)
		 	White	#FFFFFF	(255,255,255)
		 	Red	#FF0000	(255,0,0)
		 	Lime	#00FF00	(0,255,0)
		 	Blue	#0000FF	(0,0,255)
		 	Yellow	#FFFF00	(255,255,0)
		 	Cyan / Aqua	#00FFFF	(0,255,255)
		 	Magenta / Fuchsia	#FF00FF	(255,0,255)
		 	Silver	#C0C0C0	(192,192,192)
		 	Gray	#808080	(128,128,128)
		 	Maroon	#800000	(128,0,0)
		 	Olive	#808000	(128,128,0)
		 	Green	#008000	(0,128,0)
		 	Purple	#800080	(128,0,128)
		 	Teal	#008080	(0,128,128)
		 	Navy	#000080	(0,0,128)
#endif
	int table[] = {
			255,255,255, // white
			0,0,0, // black
			255,0,0,
			0,255,0,
			0,0,255,
			255,255,0,
			0,255,255,
			255,0,255,
			192,192,192,
			128,128,128,
			128,0,0,
			128,128,0,
			0,128,0,
			128,0,128,
			0,128,128,
			0,0,128
	};

	if (loopCount < 16 && bit_depth == 8) {
		r = table[loopCount * 3 + 0];
		g = table[loopCount * 3 + 1];
		b = table[loopCount * 3 + 2];
	}
	else {
		double a1, a2, x, y, z;
		int max_color;

		max_color = (1 << bit_depth) - 1;

		if (loopCount > max_value) {
			cout << "loopCount > max_value" << endl;
			cout << "loopCount=" << loopCount << endl;
			cout << "max_value=" << max_value << endl;
			exit(1);
		}

		if (loopCount < 16) {
			r = table[loopCount * 3 + 0];
			g = table[loopCount * 3 + 1];
			b = table[loopCount * 3 + 2];
			return { r, g, b};
		}
		loopCount -= 16;

		a1 = (double) loopCount / (double) max_value;


		if (f_invert_colors) {
			a2 = 1. - a1;
		}
		else {
			a2 = a1;
		}
		x = a2;
		y = a2 * a2;
		z = y * a2;
		r = x * max_color;
		g = y * max_color;
		b = z * max_color;
		if (f_v) {
			cout << loopCount << " : " << max_value << " : "
					<< a1 << " : " << a2 << " : " << x << "," << y << "," << z << " : " << r << "," << g << "," << b << endl;
		}
	}
	return { r, g, b};
}

void fillBitmap(BMP &image, int i, int j, std::vector<int> color)
{
	// The pixel is set using its image location and stacks 3 variables (RGB) into the vector word.
	image(i, j)->Red = color[0];
	image(i, j)->Green = color[1];
	image(i, j)->Blue = color[2];
};



}}

