/*
 * video_draw_options.cpp
 *
 *  Created on: Feb 10, 2019
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace graphics {



video_draw_options::video_draw_options()
{

	f_rotate = TRUE;
	rotation_axis_type = 1;
		// 1 = 1,1,1
		// 2 = 0,0,1
		// 3 = custom
	//double rotation_axis_custom[3]

	boundary_type = 1;
		// 1 = sphere
		// 2 = box
		// 3 = no clipping

	f_has_global_picture_scale = FALSE;
	global_picture_scale = 0.;

	f_has_font_size = TRUE;
	font_size = 36; // works well for 800 x 600

	f_has_stroke_width = TRUE;
	stroke_width = 2; // works well for 800 x 600

	f_W = FALSE;
	W = 0;
	f_H = FALSE;
	H = 0;

	f_default_angle = TRUE;
	default_angle = 22;

	f_clipping_radius = TRUE;
	clipping_radius = 2.1;

	nb_clipping = 0;
	//int clipping_round[1000];
	//double clipping_value[1000];

	nb_camera = 0;
	//int camera_round[1000];
	//const char *camera_sky[1000];
	//const char *camera_location[1000];
	//const char *camera_look_at[1000];

	nb_zoom = 0;
	//int zoom_round[1000];
	//int zoom_start[1000];
	//int zoom_end[1000];


	nb_zoom_sequence = 0;
	//int zoom_sequence_round[1000];
	//const char *zoom_sequence_text[1000];

	nb_pan = 0;
	//int pan_round[1000];
	//int pan_f_reverse[1000];
	//double pan_from[1000 * 3];
	//double pan_to[1000 * 3];
	//double pan_center[1000 * 3];

	nb_no_background = 0;
	//int no_background_round[1000];

	nb_no_bottom_plane = 0;
	//int no_bottom_plane_round[1000];

	cnt_nb_frames = 0;
	//int nb_frames_round[1000];
	//int nb_frames_value[1000];


	nb_round_text = 0;
	//int round_text_round[1000];
	//int round_text_sustain[1000];
	//const char *round_text_text[1000];

	nb_label = 0;
	//int label_round[1000];
	//int label_start[1000];
	//int label_sustain[1000];
	//const char *label_gravity[1000];
	//const char *label_text[1000];


	nb_latex_label = 0;
	//int latex_label_round[1000];
	//int latex_label_start[1000];
	//int latex_label_sustain[1000];
	//const char *latex_extras_for_praeamble[1000];
	//const char *latex_label_gravity[1000];
	//const char *latex_label_text[1000];
	//int latex_f_label_has_been_prepared[1000];
	//char *latex_fname_base[1000];


	nb_picture = 0;
	//int picture_round[1000];
	//double picture_scale[1000];
	//const char *picture_fname[1000];
	//const char *picture_options[1000];


	latex_file_count = 0;
	f_omit_bottom_plane = FALSE;

	//sky = "<1,1,1>";
	//location = "<-3,1,3>";
	//look_at = "<0,0,0>";
	sky[0] = 1;
	sky[1] = 1;
	sky[2] = 1;
	location[0] = -3;
	location[1] = 1;
	location[2] = 3;
	f_look_at = FALSE;
	look_at[0] = 0;
	look_at[1] = 0;
	look_at[2] = 0;

	f_scale_factor = FALSE;
	scale_factor = 1.;

	f_line_radius = FALSE;
	line_radius = 0.02;

}

video_draw_options::~video_draw_options()
{
}

int video_draw_options::read_arguments(
		int argc, std::string *argv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "video_draw_options::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-v") == 0) {
			verbose_level = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-v " << verbose_level << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-do_not_rotate") == 0) {
			f_rotate = FALSE;
			if (f_v) {
				cout << "video_draw_options::read_arguments -do_not_rotate " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-rotate_about_z_axis") == 0) {
			f_rotate = TRUE;
			rotation_axis_type = 2;
			if (f_v) {
				cout << "-rotate_about_z_axis " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-rotate_about_111") == 0) {
			f_rotate = TRUE;
			rotation_axis_type = 1;
			if (f_v) {
				cout << "-rotate_about_111 " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-rotate_about_custom_axis") == 0) {
			f_rotate = TRUE;
			rotation_axis_type = 3;

			ST.text_to_three_double(argv[++i], rotation_axis_custom);
			if (f_v) {
				cout << "-rotate_about_custom_axis " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-boundary_none") == 0) {
			boundary_type = 3;
			if (f_v) {
				cout << "-boundary_none " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-boundary_box") == 0) {
			boundary_type = 2;
			if (f_v) {
				cout << "-boundary_box " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-boundary_sphere") == 0) {
			boundary_type = 1;
			if (f_v) {
				cout << "-boundary_sphere " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-font_size") == 0) {
			f_has_font_size = TRUE;
			font_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-font_size " << font_size << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-stroke_width") == 0) {
			f_has_stroke_width = TRUE;
			stroke_width = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-stroke_width " << stroke_width << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-omit_bottom_plane") == 0) {
			f_omit_bottom_plane = TRUE;
			if (f_v) {
				cout << "-omit_bottom_plane " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-W") == 0) {
			f_W = TRUE;
			W = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-W " << W << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-H") == 0) {
			f_H = TRUE;
			H = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-H " << H << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-nb_frames") == 0) {
			nb_frames_round[cnt_nb_frames] = ST.strtoi(argv[++i]);
			nb_frames_value[cnt_nb_frames] = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-nb_frames "
					<< nb_frames_round[cnt_nb_frames] << " "
					<< nb_frames_value[cnt_nb_frames] << endl;
			}
			cnt_nb_frames++;
		}
		else if (ST.stringcmp(argv[i], "-zoom") == 0) {
			i++;
			zoom_round[nb_zoom] = ST.strtoi(argv[i]);
			i++;
			zoom_start[nb_zoom] = ST.strtoi(argv[i]);
			i++;
			zoom_end[nb_zoom] = ST.strtoi(argv[i]);
			double d;
			i++;
			d = ST.strtof(argv[++i]);
			zoom_clipping_start[nb_zoom] = d;
			i++;
			d = ST.strtof(argv[++i]);
			zoom_clipping_end[nb_zoom] = d;
			if (f_v) {
				cout << "-zoom "
					<< zoom_round[nb_zoom] << " "
					<< zoom_start[nb_zoom] << " "
					<< zoom_end[nb_zoom] << " "
					<< zoom_clipping_start[nb_zoom] << " "
					<< zoom_clipping_end[nb_zoom] << " "
					<< endl;
			}
			nb_zoom++;
		}
		else if (ST.stringcmp(argv[i], "-zoom_sequence") == 0) {
			zoom_sequence_round[nb_zoom_sequence] = ST.strtoi(argv[++i]);
			zoom_sequence_text[nb_zoom_sequence].assign(argv[++i]);
			if (f_v) {
				cout << "-zoom_sequence "
					<< zoom_sequence_round[nb_zoom_sequence] << " "
					<< zoom_sequence_text[nb_zoom_sequence] << endl;
			}
			nb_zoom_sequence++;
		}
		else if (ST.stringcmp(argv[i], "-pan") == 0) {
			pan_round[nb_pan] = ST.strtoi(argv[++i]);
			pan_f_reverse[nb_pan] = FALSE;
			orbiter_kernel_system::numerics Num;


			ST.text_to_three_double(argv[++i], pan_from + nb_pan * 3);
			ST.text_to_three_double(argv[++i], pan_to + nb_pan * 3);
			ST.text_to_three_double(argv[++i], pan_center + nb_pan * 3);

			if (f_v) {
				cout << "video_draw_options::read_arguments -pan "
						<< pan_round[nb_pan];
				cout << " ";
				Num.vec_print(pan_from + 3 * nb_pan, 3);
				cout << " ";
				Num.vec_print(pan_to + 3 * nb_pan, 3);
				cout << " ";
				Num.vec_print(pan_center + 3 * nb_pan, 3);
				cout << endl;
			}
			nb_pan++;
		}
		else if (ST.stringcmp(argv[i], "-pan_reverse") == 0) {
			pan_round[nb_pan] = ST.strtoi(argv[++i]);
			pan_f_reverse[nb_pan] = TRUE;
			orbiter_kernel_system::numerics Num;


			ST.text_to_three_double(argv[++i], pan_from + nb_pan * 3);
			ST.text_to_three_double(argv[++i], pan_to + nb_pan * 3);
			ST.text_to_three_double(argv[++i], pan_center + nb_pan * 3);

			if (f_v) {
				cout << "-pan_reverse "
					<< pan_round[nb_pan];
				cout << " ";
				Num.vec_print(pan_from + 3 * nb_pan, 3);
				cout << " ";
				Num.vec_print(pan_to + 3 * nb_pan, 3);
				cout << " ";
				Num.vec_print(pan_center + 3 * nb_pan, 3);
				cout << endl;
			}
			nb_pan++;
		}
		else if (ST.stringcmp(argv[i], "-no_background") == 0) {
			no_background_round[nb_no_background] = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-no_background "
					<< no_background_round[nb_no_background] << endl;
			}
			nb_no_background++;
		}
		else if (ST.stringcmp(argv[i], "-no_bottom_plane") == 0) {
			no_bottom_plane_round[nb_no_bottom_plane] = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-no_bottom_plane "
					<< no_bottom_plane_round[nb_no_bottom_plane] << endl;
			}
			nb_no_bottom_plane++;
		}
		else if (ST.stringcmp(argv[i], "-camera") == 0) {
			camera_round[nb_camera] = ST.strtoi(argv[++i]);



			ST.text_to_three_double(argv[++i], camera_sky + nb_camera * 3);
			ST.text_to_three_double(argv[++i], camera_location + nb_camera * 3);
			ST.text_to_three_double(argv[++i], camera_look_at + nb_camera * 3);

			if (f_v) {
				cout << "-camera "
						<< camera_round[nb_camera] << " "
						<< camera_sky[nb_camera * 3 + 0] << " "
						<< camera_sky[nb_camera * 3 + 1] << " "
						<< camera_sky[nb_camera * 3 + 2] << " "
						<< camera_location[nb_camera * 3 + 0] << " "
						<< camera_location[nb_camera * 3 + 1] << " "
						<< camera_location[nb_camera * 3 + 2] << " "
						<< camera_look_at[nb_camera * 3 + 0] << " "
						<< camera_look_at[nb_camera * 3 + 1] << " "
						<< camera_look_at[nb_camera * 3 + 2] << " "
						<< endl;
			}
			nb_camera++;

			   //sky <1,1,1>
			   	   //direction <1,0,0>
			   	   //right <1,1,0>
			   //location  <-3,1,3>
			   //look_at  <0,0,0>
		}
		else if (ST.stringcmp(argv[i], "-clipping") == 0) {
			clipping_round[nb_clipping] = ST.strtoi(argv[++i]);
			double d;
			d = ST.strtof(argv[++i]);
			clipping_value[nb_clipping] = d;
			if (f_v) {
				cout << "-clipping "
					<< clipping_round[nb_clipping] << " "
					<< clipping_value[nb_clipping] << endl;
			}
			nb_clipping++;
		}
		else if (ST.stringcmp(argv[i], "-text") == 0) {
			round_text_round[nb_round_text] = ST.strtoi(argv[++i]);
			round_text_sustain[nb_round_text] = ST.strtoi(argv[++i]);
			round_text_text[nb_round_text].assign(argv[++i]);
			if (f_v) {
				cout << "-text "
					<< round_text_round[nb_round_text] << " "
					<< round_text_sustain[nb_round_text] << " "
					<< round_text_text[nb_round_text] << endl;
			}
			nb_round_text++;
		}
		else if (ST.stringcmp(argv[i], "-label") == 0) {
			label_round[nb_label] = ST.strtoi(argv[++i]);
			label_start[nb_label] = ST.strtoi(argv[++i]);
			label_sustain[nb_label] = ST.strtoi(argv[++i]);
			label_gravity[nb_label].assign(argv[++i]);
			label_text[nb_label].assign(argv[++i]);
			if (f_v) {
				cout << "-label "
					<< label_round[nb_label] << " "
					<< label_start[nb_label] << " "
					<< label_sustain[nb_label] << " "
					<< label_gravity[nb_label] << " "
					<< label_text[nb_label] << " "
					<< endl;
			}
			nb_label++;
		}
		else if (ST.stringcmp(argv[i], "-latex") == 0) {
			latex_label_round[nb_latex_label] = ST.strtoi(argv[++i]);
			latex_label_start[nb_latex_label] = ST.strtoi(argv[++i]);
			latex_label_sustain[nb_latex_label] = ST.strtoi(argv[++i]);
			latex_extras_for_praeamble[nb_latex_label].assign(argv[++i]);
			latex_label_gravity[nb_latex_label].assign(argv[++i]);
			latex_label_text[nb_latex_label].assign(argv[++i]);
			latex_f_label_has_been_prepared[nb_latex_label] = FALSE;
			//latex_fname_base[nb_latex_label] = NEW_char(1000);
			if (f_v) {
				cout << "-latex "
					<< latex_label_round[nb_latex_label] << " "
					<< latex_label_start[nb_latex_label] << " "
					<< latex_label_sustain[nb_latex_label] << " "
					<< latex_extras_for_praeamble[nb_latex_label] << " "
					<< latex_label_gravity[nb_latex_label] << " "
					<< latex_label_text[nb_latex_label] << " "
					<< endl;
			}
			nb_latex_label++;
		}
		else if (ST.stringcmp(argv[i], "-global_picture_scale") == 0) {
			f_has_global_picture_scale = TRUE;
			double d;
			d = ST.strtof(argv[++i]);
			global_picture_scale = d;
			if (f_v) {
				cout << "-global_picture_scale " << d << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-picture") == 0) {
			picture_round[nb_picture] = ST.strtoi(argv[++i]);
			double d;
			d = ST.strtof(argv[++i]);
			picture_scale[nb_picture] = d;
			picture_fname[nb_picture].assign(argv[++i]);
			picture_options[nb_picture].assign(argv[++i]);
			if (f_v) {
				cout << "-picture "
					<< picture_round[nb_picture] << " "
					<< picture_scale[nb_picture] << " "
					<< picture_fname[nb_picture] << " "
					<< picture_options[nb_picture] << " "
					<< endl;
			}
			nb_picture++;
		}
		else if (ST.stringcmp(argv[i], "-look_at") == 0) {
			//look_at = argv[++i];
			f_look_at = TRUE;

			ST.text_to_three_double(argv[++i], look_at);
			if (f_v) {
				cout << "-look_at "
						<< look_at[0] << " " << look_at[1] << " " << look_at[2] << " " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-default_angle") == 0) {
			f_default_angle = TRUE;
			default_angle = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-default_angle " << default_angle << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-clipping_radius") == 0) {
			f_clipping_radius = TRUE;
			clipping_radius = ST.strtof(argv[++i]);
			if (f_v) {
				cout << "-clipping_radius " << clipping_radius << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-scale_factor") == 0) {
			f_scale_factor = TRUE;
			scale_factor = ST.strtof(argv[++i]);
			if (f_v) {
				cout << "-scale_factor " << scale_factor << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-line_radius") == 0) {
			f_line_radius = TRUE;
			line_radius = ST.strtof(argv[++i]);
			if (f_v) {
				cout << "-line_radius " << line_radius << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "video_draw_options::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "video_draw_options::read_arguments done" << endl;
	}
	return i + 1;
}


void video_draw_options::print()
{
	if (f_rotate == FALSE) {
		cout << "-do_not_rotate " << endl;
	}
	if (f_rotate && rotation_axis_type == 2) {
		cout << "-rotate_about_z_axis " << endl;
	}
	if (f_rotate && rotation_axis_type == 1) {
		cout << "-rotate_about_111 " << endl;
	}
	if (f_rotate && rotation_axis_type == 3) {
		cout << "-rotate_about_custom_axis " << endl;
	}
	if (boundary_type == 3) {
		cout << "-boundary_none " << endl;
	}
	if (boundary_type == 2) {
		cout << "-boundary_box " << endl;
	}
	if (boundary_type == 1) {
		cout << "-boundary_sphere " << endl;
	}
	if (f_has_font_size) {
		cout << "-font_size " << font_size << endl;
	}
	if (f_has_stroke_width) {
		cout << "-stroke_width " << stroke_width << endl;
	}
	if (f_omit_bottom_plane) {
		cout << "-omit_bottom_plane " << endl;
	}

	if (f_W) {
		cout << "-W " << W << endl;
	}
	if (f_H) {
		cout << "-H " << H << endl;
	}
	for (int i = 0; i < cnt_nb_frames; i++) {
		cout << "-nb_frames "
			<< nb_frames_round[i] << " "
			<< nb_frames_value[i] << endl;
	}
	for (int i = 0; i < nb_zoom; i++) {
		cout << "-zoom "
			<< zoom_round[i] << " "
			<< zoom_start[i] << " "
			<< zoom_end[i] << " "
			<< zoom_clipping_start[i] << " "
			<< zoom_clipping_end[i] << " "
			<< endl;
	}
	for (int i = 0; i < nb_zoom_sequence; i++) {
		cout << "-zoom_sequence "
			<< zoom_sequence_round[i] << " "
			<< zoom_sequence_text[i] << endl;
	}
	for (int i = 0; i < nb_pan; i++) {
		if (pan_f_reverse[nb_pan]) {
			orbiter_kernel_system::numerics Num;
			cout << "-pan_reverse "
					<< pan_round[i];
			cout << " ";
			Num.vec_print(pan_from + 3 * i, 3);
			cout << " ";
			Num.vec_print(pan_to + 3 * i, 3);
			cout << " ";
			Num.vec_print(pan_center + 3 * i, 3);
			cout << endl;

		}
		else {
			orbiter_kernel_system::numerics Num;
			cout << "-pan "
					<< pan_round[i];
			cout << " ";
			Num.vec_print(pan_from + 3 * i, 3);
			cout << " ";
			Num.vec_print(pan_to + 3 * i, 3);
			cout << " ";
			Num.vec_print(pan_center + 3 * i, 3);
			cout << endl;
		}
	}
	for (int i = 0; i < nb_no_background; i++) {
		cout << "-no_background "
			<< no_background_round[i] << endl;
	}
	for (int i = 0; i < nb_no_bottom_plane; i++) {
		cout << "-no_bottom_plane "
			<< no_bottom_plane_round[i] << endl;
	}
	for (int i = 0; i < nb_camera; i++) {
		cout << "-camera "
				<< camera_round[i] << " "
				<< camera_sky[i * 3 + 0] << " "
				<< camera_sky[i * 3 + 1] << " "
				<< camera_sky[i * 3 + 2] << " "
				<< camera_location[i * 3 + 0] << " "
				<< camera_location[i * 3 + 1] << " "
				<< camera_location[i * 3 + 2] << " "
				<< camera_look_at[i * 3 + 0] << " "
				<< camera_look_at[i * 3 + 1] << " "
				<< camera_look_at[i * 3 + 2] << " "
				<< endl;
	}
	for (int i = 0; i < nb_clipping; i++) {
		cout << "-clipping "
			<< clipping_round[i] << " "
			<< clipping_value[i] << endl;
	}
	for (int i = 0; i < nb_round_text; i++) {
		cout << "-text "
			<< round_text_round[i] << " "
			<< round_text_sustain[i] << " "
			<< round_text_text[i] << endl;
	}
	for (int i = 0; i < nb_label; i++) {
		cout << "-label "
			<< label_round[i] << " "
			<< label_start[i] << " "
			<< label_sustain[i] << " "
			<< label_gravity[i] << " "
			<< label_text[i] << " "
			<< endl;
	}
	for (int i = 0; i < nb_latex_label; i++) {
		cout << "-latex "
			<< latex_label_round[i] << " "
			<< latex_label_start[i] << " "
			<< latex_label_sustain[i] << " "
			<< latex_extras_for_praeamble[i] << " "
			<< latex_label_gravity[i] << " "
			<< latex_label_text[i] << " "
			<< endl;
	}
	if (f_has_global_picture_scale) {
		cout << "-global_picture_scale " << global_picture_scale << endl;
	}
	for (int i = 0; i < nb_picture; i++) {
		cout << "-picture "
			<< picture_round[i] << " "
			<< picture_scale[i] << " "
			<< picture_fname[i] << " "
			<< picture_options[i] << " "
			<< endl;
	}
	if (f_look_at) {
		cout << "-look_at "
					<< look_at[0] << " " << look_at[1] << " " << look_at[2] << " " << endl;
	}

	if (f_default_angle) {
		cout << "-default_angle " << default_angle << endl;
	}
	if (f_clipping_radius) {
		cout << "-clipping_radius " << clipping_radius << endl;
	}
	if (f_scale_factor) {
		cout << "-scale_factor " << scale_factor << endl;
	}
	if (f_line_radius) {
		cout << "-line_radius " << line_radius << endl;
	}
}




}}}


