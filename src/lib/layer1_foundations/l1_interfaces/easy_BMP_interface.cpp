/*
 * easy_BMP_interface.cpp
 *
 *  Created on: Apr 5, 2023
 *      Author: betten
 */



#include "EasyBMP.h"
#include "foundations.h"



using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace l1_interfaces {


std::vector<int> get_color_grayscale(int bit_depth, int max_value, int loopCount, int f_invert_colors, int verbose_level);
static std::vector<int> get_color(int bit_depth, int max_value, int loopCount, int f_invert_colors, int verbose_level);
static void fillBitmap(BMP &image, int i, int j, std::vector<int> color);



void easy_BMP_interface::draw_bitmap(
		graphics::draw_bitmap_control *C, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "easy_BMP_interface::draw_bitmap" << endl;
	}
	orbiter_kernel_system::file_io Fio;


	if (C->f_input_csv_file) {

		Fio.int_matrix_read_csv(C->input_csv_file_name,
				C->M, C->m, C->n,
				verbose_level);

		if (C->f_secondary_input_csv_file) {


			int m, n;


			Fio.int_matrix_read_csv(C->secondary_input_csv_file_name,
					C->M2, m, n,
					verbose_level);
			if (m != C->m) {
				cout << "secondary matrix must have the same size as the primary input matrix" << endl;
				exit(1);
			}
			if (n != C->n) {
				cout << "secondary matrix must have the same size as the primary input matrix" << endl;
				exit(1);
			}
		}

	}
	else if (C->f_input_object) {

		Get_matrix(C->input_object_label,
				C->M, C->m, C->n);
	}

	if (f_v) {
		cout << "easy_BMP_interface::draw_bitmap drawing matrix of size " << C->m << " x " << C->n << endl;
	}

	int *Row_parts = NULL;
	int nb_row_parts = 0;
	int *Col_parts = NULL;
	int nb_col_parts = 0;


	if (C->f_partition) {

		Get_int_vector_from_label(C->part_row, Row_parts, nb_row_parts, 0 /* verbose_level*/);
		Get_int_vector_from_label(C->part_col, Col_parts, nb_col_parts, 0 /* verbose_level*/);

		if (f_v) {
			cout << "row_part: ";
			Int_vec_print(cout, Row_parts, nb_row_parts);
			cout << endl;
			cout << "col_part: ";
			Int_vec_print(cout, Col_parts, nb_col_parts);
			cout << endl;
		}
	}
	int i;
	int max_value;
	data_structures::string_tools ST;

	max_value = Int_vec_maximum(C->M, C->m * C->n);
	if (f_v) {
		cout << "max_value=" << max_value << endl;
	}

	//max_value += 5;
	//cout << "max_value after adjustment=" << max_value << endl;


	string fname_out;

	if (C->f_input_csv_file) {
		fname_out.assign(C->input_csv_file_name);
	}
	else {
		fname_out.assign("bitmatrix.csv");

	}
	ST.replace_extension_with(fname_out, "_draw.bmp");

	//int bit_depth = 8;

	BMP image;

	//int bit_depth = 24;


	if (max_value > 10000) {
		cout << "easy_BMP_interface::draw_bitmap max_value > 10000" << endl;
		exit(1);
	}
	if (max_value == 0) {
		max_value = 1;
	}


	if (f_v) {
		cout << "easy_BMP_interface::draw_bitmap color palette:" << endl;
		for (i = 0; i <= max_value; i++) {
			std::vector<int> color;

			if (C->f_grayscale) {
				color = get_color_grayscale(C->bit_depth, max_value, i, C->f_invert_colors, 1);
			}
			else {
				color = get_color(C->bit_depth, max_value, i, C->f_invert_colors, 1);
			}
			cout << "color " << i << " : " << color[0] << "," << color[1] << "," << color[2] << endl;
		}
	}

	int width, height;
	//int *Table;
	geometry::geometry_global Gg;

	width = C->n;
	height = C->m;

	if (f_v) {
		cout << "width=" << width << endl;
	}

	if (C->f_box_width) {
		image.SetSize(width * C->box_width, height * C->box_width);
	}
	else {
		image.SetSize(width, height);
	}

	image.SetBitDepth(C->bit_depth);

	int j, d;
	int N, N100, cnt;
	int indent = 0;

	N = height * width;
	N100 = N / 100 + 1;

	if (f_v) {
		cout << "N100=" << N100 << endl;
	}

	cnt = 0;

	if (C->f_secondary_input_csv_file) {
		indent = C->box_width >> 2;
	}
	if (f_v) {
		cout << "indent=" << indent << endl;
	}

	std::vector<int> color_white;

	if (C->f_grayscale) {
		color_white = get_color_grayscale(C->bit_depth, max_value, 0, C->f_invert_colors, 0);
	}
	else {
		color_white = get_color(C->bit_depth, max_value, 0, C->f_invert_colors, 0);
	}

	if (f_v) {
		cout << "color_white=" << color_white[0] << "," << color_white[1] << "," << color_white[2] << endl;
	}



	for (i = 0; i < height; i++) {



		for (j = 0; j < width; j++, cnt++) {


			if ((cnt % N100) == 0) {
				cout << "we are at " << ((double) cnt / (double) N) * 100. << " %" << endl;
			}
			d = C->M[i * width + j];
			//std::vector<int> color = getColor(M[idx_x * width + idx_z]);
			std::vector<int> color;

			if (C->f_grayscale) {
				color = get_color_grayscale(C->bit_depth, max_value, d, C->f_invert_colors, 0);
			}
			else {
				color = get_color(C->bit_depth, max_value, d, C->f_invert_colors, 0);

			}



			// Here the pixel is set on the image.
			if (C->f_box_width) {
				int I, J, u, v;

				I = i * C->box_width;
				J = j * C->box_width;

				if (C->f_secondary_input_csv_file) {
					if (C->M2[i * width + j] == 0) {
						for (u = 0; u < C->box_width; u++) {
							for (v = 0; v < C->box_width; v++) {
								if (u < indent || u >= C->box_width - indent || v < indent || v >= C->box_width - indent) {
									fillBitmap(image, J + v, I + u, color_white);
								}
								else {
									fillBitmap(image, J + v, I + u, color);
								}
							}
						}
					}
					else {
						for (u = 0; u < C->box_width; u++) {
							for (v = 0; v < C->box_width; v++) {
								fillBitmap(image, J + v, I + u, color);
							}
						}
					}
				}
				else {
					for (u = 0; u < C->box_width; u++) {
						for (v = 0; v < C->box_width; v++) {
							fillBitmap(image, J + v, I + u, color);
						}
					}
				}

			}
			else {
				fillBitmap(image, j, i, color);
			}
		}
	}
	if (C->f_partition) {

		if (f_v) {
			cout << "drawing the partition" << endl;
		}
		int i0, j0;
		int h, t, I, J;
		std::vector<int> color;


		if (C->f_grayscale) {
			color = get_color_grayscale(C->bit_depth, max_value, max_value, C->f_invert_colors, 0);
		}
		else {
			color = get_color(C->bit_depth, max_value, 1, C->f_invert_colors, 0);

		}

		// row partition:
		i0 = 0;
		for (h = 0; h <= nb_row_parts; h++) {
			for (t = 0; t < C->part_width; t++) {
				if (C->f_box_width) {
					for (j = 0; j < width * C->box_width; j++) {
						I = i0 * C->box_width;
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
				i0 += Row_parts[h];
			}
		}

		// col partition:
		j0 = 0;
		for (h = 0; h <= nb_col_parts; h++) {
			for (t = 0; t < C->part_width; t++) {
				if (C->f_box_width) {
					for (i = 0; i < height * C->box_width; i++) {
						J = j0 * C->box_width;
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
				j0 += Col_parts[h];
			}
		}
	}

	if (f_v) {
		cout << "before writing the image to file as " << fname_out << endl;
	}

	image.WriteToFile(fname_out.c_str());

	if (f_v) {
		std::cout << "Written file " << fname_out << std::endl;
		{
			orbiter_kernel_system::file_io Fio;
			cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;
		}
	}



	if (f_v) {
		cout << "easy_BMP_interface::draw_bitmap done" << endl;
	}

}

void easy_BMP_interface::random_noise_in_bitmap_file(
		std::string fname_input,
		std::string fname_output,
		int probability_numerator,
		int probability_denominator,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graphical_output::random_noise_in_bitmap_file" << endl;
	}
	orbiter_kernel_system::file_io Fio;

	BMP image;
	int H, W, i, j, r, c;
	orbiter_kernel_system::os_interface Os;

	image.ReadFromFile(fname_input.c_str());

	H = image.TellHeight();
	W = image.TellWidth();

	cout << "image size H=" << H << " W=" << W << endl;


	for (i = 0; i < H; i++) {
		for (j = 0; j < W; j++) {
			RGBApixel pix, pix1;

			pix = image.GetPixel(j, i);
			cout << i << " : " << j << " : " << (int) pix.Blue << "," << (int) pix.Green << "," << (int) pix.Red << endl;


			r = Os.random_integer(probability_denominator);
			if (r < probability_numerator) {

				c = Os.random_integer(256);
				pix1.Blue = pix1.Green = pix1.Red = c;
				pix1.Alpha = pix.Alpha;
				image.SetPixel(j, i, pix1);
			}
		}
	}

	image.WriteToFile(fname_output.c_str());


	if (f_v) {
		cout << "easy_BMP_interface::random_noise_in_bitmap_file done" << endl;
	}
}

void easy_BMP_interface::random_noise_in_bitmap_file_burst(
		std::string fname_input,
		std::string fname_output,
		int probability_numerator,
		int probability_denominator,
		int burst_length_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "easy_BMP_interface::random_noise_in_bitmap_file_burst" << endl;
	}
	orbiter_kernel_system::file_io Fio;

	BMP image;
	int H, W, i, j, h, k, r, c, burst_length;
	orbiter_kernel_system::os_interface Os;

	image.ReadFromFile(fname_input.c_str());

	H = image.TellHeight();
	W = image.TellWidth();

	cout << "image size H=" << H << " W=" << W << endl;

	probability_denominator *= burst_length_max * 2;

	for (i = 0; i < H; i++) {
		for (j = 0; j < W; j++) {
			RGBApixel pix, pix1;

			pix = image.GetPixel(j, i);
			cout << i << " : " << j << " : " << (int) pix.Blue << "," << (int) pix.Green << "," << (int) pix.Red << endl;


			r = Os.random_integer(probability_denominator);
			if (r < probability_numerator) {

				burst_length = Os.random_integer(burst_length_max);
				c = Os.random_integer(256);

				for (h = 0; h < burst_length; h++) {
					for (k = 0; k < 5; k++) {
						pix1.Blue = pix1.Green = pix1.Red = c;
						pix1.Alpha = pix.Alpha;
						if (j + h < W && i + k < H) {
							image.SetPixel(j + h, i + k, pix1);
						}
					}
				}
			}
		}
	}

	image.WriteToFile(fname_output.c_str());


	if (f_v) {
		cout << "easy_BMP_interface::random_noise_in_bitmap_file_burst done" << endl;
	}
}



std::vector<int> get_color_grayscale(int bit_depth, int max_value, int loopCount, int f_invert_colors, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int r, g, b;

	int d;

	d = 255 / max_value;

	if (loopCount > max_value) {
		loopCount = max_value;
	}

	if (f_invert_colors) {
		loopCount = max_value - loopCount;
	}
	r = g = b = loopCount * d;

	return { r, g, b};
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

#if 0
			maroon,#800000,128,0,0
			dark red,#8B0000,139,0,0
			brown,#A52A2A,165,42,42
			firebrick,#B22222,178,34,34
			crimson,#DC143C,220,20,60
			red,#FF0000,255,0,0
			tomato,#FF6347,255,99,71
			coral,#FF7F50,255,127,80
			indian red,#CD5C5C,205,92,92
			light coral,#F08080,240,128,128
			dark salmon,#E9967A,233,150,122
			salmon,#FA8072,250,128,114
			light salmon,#FFA07A,255,160,122
			orange red,#FF4500,255,69,0
			dark orange,#FF8C00,255,140,0
			orange,#FFA500,255,165,0
			gold,#FFD700,255,215,0
			dark golden rod,#B8860B,184,134,11
			golden rod,#DAA520,218,165,32
			pale golden rod,#EEE8AA,238,232,170
			dark khaki,#BDB76B,189,183,107
			khaki,#F0E68C,240,230,140
			olive,#808000,128,128,0
			yellow,#FFFF00,255,255,0
			yellow green,#9ACD32,154,205,50
			dark olive green,#556B2F,85,107,47
			olive drab,#6B8E23,107,142,35
			lawn green,#7CFC00,124,252,0
			chartreuse,#7FFF00,127,255,0
			green yellow,#ADFF2F,173,255,47
			dark green,#006400,0,100,0
			green,#008000,0,128,0
			forest green,#228B22,34,139,34
			lime,#00FF00,0,255,0
			lime green,#32CD32,50,205,50
			light green,#90EE90,144,238,144
			pale green,#98FB98,152,251,152
			dark sea green,#8FBC8F,143,188,143
			medium spring green,#00FA9A,0,250,154
			spring green,#00FF7F,0,255,127
			sea green,#2E8B57,46,139,87
			medium aqua marine,#66CDAA,102,205,170
			medium sea green,#3CB371,60,179,113
			light sea green,#20B2AA,32,178,170
			dark slate gray,#2F4F4F,47,79,79
			teal,#008080,0,128,128
			dark cyan,#008B8B,0,139,139
			aqua,#00FFFF,0,255,255
			cyan,#00FFFF,0,255,255
			light cyan,#E0FFFF,224,255,255
			dark turquoise,#00CED1,0,206,209
			turquoise,#40E0D0,64,224,208
			medium turquoise,#48D1CC,72,209,204
			pale turquoise,#AFEEEE,175,238,238
			aqua marine,#7FFFD4,127,255,212
			powder blue,#B0E0E6,176,224,230
			cadet blue,#5F9EA0,95,158,160
			steel blue,#4682B4,70,130,180
			corn flower blue,#6495ED,100,149,237
			deep sky blue,#00BFFF,0,191,255
			dodger blue,#1E90FF,30,144,255
			light blue,#ADD8E6,173,216,230
			sky blue,#87CEEB,135,206,235
			light sky blue,#87CEFA,135,206,250
			midnight blue,#191970,25,25,112
			navy,#000080,0,0,128
			dark blue,#00008B,0,0,139
			medium blue,#0000CD,0,0,205
			blue,#0000FF,0,0,255
			royal blue,#4169E1,65,105,225
			blue violet,#8A2BE2,138,43,226
			indigo,#4B0082,75,0,130
			dark slate blue,#483D8B,72,61,139
			slate blue,#6A5ACD,106,90,205
			medium slate blue,#7B68EE,123,104,238
			medium purple,#9370DB,147,112,219
			dark magenta,#8B008B,139,0,139
			dark violet,#9400D3,148,0,211
			dark orchid,#9932CC,153,50,204
			medium orchid,#BA55D3,186,85,211
			purple,#800080,128,0,128
			thistle,#D8BFD8,216,191,216
			plum,#DDA0DD,221,160,221
			violet,#EE82EE,238,130,238
			magenta / fuchsia,#FF00FF,255,0,255
			orchid,#DA70D6,218,112,214
			medium violet red,#C71585,199,21,133
			pale violet red,#DB7093,219,112,147
			deep pink,#FF1493,255,20,147
			hot pink,#FF69B4,255,105,180
			light pink,#FFB6C1,255,182,193
			pink,#FFC0CB,255,192,203
			antique white,#FAEBD7,250,235,215
			beige,#F5F5DC,245,245,220
			bisque,#FFE4C4,255,228,196
			blanched almond,#FFEBCD,255,235,205
			wheat,#F5DEB3,245,222,179
			corn silk,#FFF8DC,255,248,220
			lemon chiffon,#FFFACD,255,250,205
			light golden rod yellow,#FAFAD2,250,250,210
			light yellow,#FFFFE0,255,255,224
			saddle brown,#8B4513,139,69,19
			sienna,#A0522D,160,82,45
			chocolate,#D2691E,210,105,30
			peru,#CD853F,205,133,63
			sandy brown,#F4A460,244,164,96
			burly wood,#DEB887,222,184,135
			tan,#D2B48C,210,180,140
			rosy brown,#BC8F8F,188,143,143
			moccasin,#FFE4B5,255,228,181
			navajo white,#FFDEAD,255,222,173
			peach puff,#FFDAB9,255,218,185
			misty rose,#FFE4E1,255,228,225
			lavender blush,#FFF0F5,255,240,245
			linen,#FAF0E6,250,240,230
			old lace,#FDF5E6,253,245,230
			papaya whip,#FFEFD5,255,239,213
			sea shell,#FFF5EE,255,245,238
			mint cream,#F5FFFA,245,255,250
			slate gray,#708090,112,128,144
			light slate gray,#778899,119,136,153
			light steel blue,#B0C4DE,176,196,222
			lavender,#E6E6FA,230,230,250
			floral white,#FFFAF0,255,250,240
			alice blue,#F0F8FF,240,248,255
			ghost white,#F8F8FF,248,248,255
			honeydew,#F0FFF0,240,255,240
			ivory,#FFFFF0,255,255,240
			azure,#F0FFFF,240,255,255
			snow,#FFFAFA,255,250,250
			black,#000000,0,0,0
			dim gray / dim grey,#696969,105,105,105
			gray / grey,#808080,128,128,128
			dark gray / dark grey,#A9A9A9,169,169,169
			silver,#C0C0C0,192,192,192
			light gray / light grey,#D3D3D3,211,211,211
			gainsboro,#DCDCDC,220,220,220
			white smoke,#F5F5F5,245,245,245
			white,#FFFFFF,255,255,255
#endif

	int table[] = {

			//https://www.rapidtables.com/web/color/RGB_Color.html

			// 16 predefined colors:
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
			0,0,128,

			// 137 more:
			128,0,0,
			139,0,0,
			165,42,42,
			178,34,34,
			220,20,60,
			255,0,0,
			255,99,71,
			255,127,80,
			205,92,92,
			240,128,128,
			233,150,122,
			250,128,114,
			255,160,122,
			255,69,0,
			255,140,0,
			255,165,0,
			255,215,0,
			184,134,11,
			218,165,32,
			238,232,170,
			189,183,107,
			240,230,140,
			128,128,0,
			255,255,0,
			154,205,50,
			85,107,47,
			107,142,35,
			124,252,0,
			127,255,0,
			173,255,47,
			0,100,0,
			0,128,0,
			34,139,34,
			0,255,0,
			50,205,50,
			144,238,144,
			152,251,152,
			143,188,143,
			0,250,154,
			0,255,127,
			46,139,87,
			102,205,170,
			60,179,113,
			32,178,170,
			47,79,79,
			0,128,128,
			0,139,139,
			0,255,255,
			0,255,255,
			224,255,255,
			0,206,209,
			64,224,208,
			72,209,204,
			175,238,238,
			127,255,212,
			176,224,230,
			95,158,160,
			70,130,180,
			100,149,237,
			0,191,255,
			30,144,255,
			173,216,230,
			135,206,235,
			135,206,250,
			25,25,112,
			0,0,128,
			0,0,139,
			0,0,205,
			0,0,255,
			65,105,225,
			138,43,226,
			75,0,130,
			72,61,139,
			106,90,205,
			123,104,238,
			147,112,219,
			139,0,139,
			148,0,211,
			153,50,204,
			186,85,211,
			128,0,128,
			216,191,216,
			221,160,221,
			238,130,238,
			255,0,255,
			218,112,214,
			199,21,133,
			219,112,147,
			255,20,147,
			255,105,180,
			255,182,193,
			255,192,203,
			250,235,215,
			245,245,220,
			255,228,196,
			255,235,205,
			245,222,179,
			255,248,220,
			255,250,205,
			250,250,210,
			255,255,224,
			139,69,19,
			160,82,45,
			210,105,30,
			205,133,63,
			244,164,96,
			222,184,135,
			210,180,140,
			188,143,143,
			255,228,181,
			255,222,173,
			255,218,185,
			255,228,225,
			255,240,245,
			250,240,230,
			253,245,230,
			255,239,213,
			255,245,238,
			245,255,250,
			112,128,144,
			119,136,153,
			176,196,222,
			230,230,250,
			255,250,240,
			240,248,255,
			248,248,255,
			240,255,240,
			255,255,240,
			//240,255,255,
			255,250,250,
			//0,0,0,
			105,105,105,
			//128,128,128,
			169,169,169,
			//192,192,192,
			211,211,211,
			220,220,220,
			245,245,245,

			//255,255,255,

			// 16 + 127 = 143
	};


	if (loopCount < 16 /*&& bit_depth == 8*/) {
		r = table[loopCount * 3 + 0];
		g = table[loopCount * 3 + 1];
		b = table[loopCount * 3 + 2];
	}
	else if (loopCount < 143 /*&& bit_depth == 8*/) {

		loopCount -= 16;

		number_theory::number_theory_domain NT;
		int idx;

		idx = NT.power_mod(57, loopCount, 127);

		idx += 16;

		r = table[idx * 3 + 0];
		g = table[idx * 3 + 1];
		b = table[idx * 3 + 2];
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

		if (loopCount < 153) {
			r = table[loopCount * 3 + 0];
			g = table[loopCount * 3 + 1];
			b = table[loopCount * 3 + 2];
			return { r, g, b};
		}
		loopCount -= 153;

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

	//cout << "color " << loopCount << " : " << r << "," << g << "," << b << endl;
	return { r, g, b};
}

void fillBitmap(BMP &image, int i, int j, std::vector<int> color)
{
	// The pixel is set using its image
	// location and stacks 3 variables (RGB) into the vector word.
	image(i, j)->Red = color[0];
	image(i, j)->Green = color[1];
	image(i, j)->Blue = color[2];
};


}}}



