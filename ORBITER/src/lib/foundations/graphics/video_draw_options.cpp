/*
 * video_draw_options.cpp
 *
 *  Created on: Feb 10, 2019
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {



video_draw_options::video_draw_options()
{
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
	//double pan_from_x[1000];
	//double pan_from_y[1000];
	//double pan_from_z[1000];
	//double pan_to_x[1000];
	//double pan_to_y[1000];
	//double pan_to_z[1000];
	//double pan_center_x[1000];
	//double pan_center_y[1000];
	//double pan_center_z[1000];

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

	sky = "<1,1,1>";
	location = "<-3,1,3>";
	look_at = "<0,0,0>";


}

video_draw_options::~video_draw_options()
{
}

int video_draw_options::read_arguments(
		int argc, const char **argv,
		int verbose_level)
{
	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-font_size") == 0) {
			f_has_font_size = TRUE;
			font_size = atoi(argv[++i]);
			cout << "-font_size " << font_size << endl;
			}
		else if (strcmp(argv[i], "-stroke_width") == 0) {
			f_has_stroke_width = TRUE;
			stroke_width = atoi(argv[++i]);
			cout << "-stroke_width " << stroke_width << endl;
			}
		else if (strcmp(argv[i], "-omit_bottom_plane") == 0) {
			f_omit_bottom_plane = TRUE;
			cout << "-omit_bottom_plane " << endl;
			}

		else if (strcmp(argv[i], "-W") == 0) {
			f_W = TRUE;
			W = atoi(argv[++i]);
			cout << "-W " << W << endl;
			}
		else if (strcmp(argv[i], "-H") == 0) {
			f_H = TRUE;
			H = atoi(argv[++i]);
			cout << "-H " << H << endl;
			}
		else if (strcmp(argv[i], "-nb_frames") == 0) {
			nb_frames_round[cnt_nb_frames] = atoi(argv[++i]);
			nb_frames_value[cnt_nb_frames] = atoi(argv[++i]);
			cout << "-nb_frames "
				<< nb_frames_round[cnt_nb_frames] << " "
				<< nb_frames_value[cnt_nb_frames] << endl;
			cnt_nb_frames++;
			}
		else if (strcmp(argv[i], "-zoom") == 0) {
			zoom_round[nb_zoom] = atoi(argv[++i]);
			zoom_start[nb_zoom] = atoi(argv[++i]);
			zoom_end[nb_zoom] = atoi(argv[++i]);
			cout << "-zoom "
				<< zoom_round[nb_zoom] << " "
				<< zoom_start[nb_zoom] << " "
				<< zoom_end[nb_zoom] << endl;
			nb_zoom++;
			}
		else if (strcmp(argv[i], "-zoom_sequence") == 0) {
			zoom_sequence_round[nb_zoom_sequence] = atoi(argv[++i]);
			zoom_sequence_text[nb_zoom_sequence] = argv[++i];
			cout << "-zoom_sequence "
				<< zoom_sequence_round[nb_zoom_sequence] << " "
				<< zoom_sequence_text[nb_zoom_sequence] << endl;
			nb_zoom_sequence++;
			}
		else if (strcmp(argv[i], "-pan") == 0) {
			pan_round[nb_pan] = atoi(argv[++i]);
			pan_f_reverse[nb_pan] = FALSE;
			pan_from_x[nb_pan] = atof(argv[++i]);
			pan_from_y[nb_pan] = atof(argv[++i]);
			pan_from_z[nb_pan] = atof(argv[++i]);
			pan_to_x[nb_pan] = atof(argv[++i]);
			pan_to_y[nb_pan] = atof(argv[++i]);
			pan_to_z[nb_pan] = atof(argv[++i]);
			pan_center_x[nb_pan] = atof(argv[++i]);
			pan_center_y[nb_pan] = atof(argv[++i]);
			pan_center_z[nb_pan] = atof(argv[++i]);
			cout << "-pan "
				<< pan_round[nb_pan] << " "
				<< pan_from_x[nb_pan] << " "
				<< pan_from_y[nb_pan] << " "
				<< pan_from_z[nb_pan] << " "
				<< pan_to_x[nb_pan] << " "
				<< pan_to_y[nb_pan] << " "
				<< pan_to_z[nb_pan] << " "
				<< pan_center_x[nb_pan] << " "
				<< pan_center_y[nb_pan] << " "
				<< pan_center_z[nb_pan] << " "
				<< endl;
			nb_pan++;
			}
		else if (strcmp(argv[i], "-pan_reverse") == 0) {
			pan_round[nb_pan] = atoi(argv[++i]);
			pan_f_reverse[nb_pan] = TRUE;
			pan_from_x[nb_pan] = atof(argv[++i]);
			pan_from_y[nb_pan] = atof(argv[++i]);
			pan_from_z[nb_pan] = atof(argv[++i]);
			pan_to_x[nb_pan] = atof(argv[++i]);
			pan_to_y[nb_pan] = atof(argv[++i]);
			pan_to_z[nb_pan] = atof(argv[++i]);
			pan_center_x[nb_pan] = atof(argv[++i]);
			pan_center_y[nb_pan] = atof(argv[++i]);
			pan_center_z[nb_pan] = atof(argv[++i]);
			cout << "-pan_reverse "
				<< pan_round[nb_pan] << " "
				<< pan_from_x[nb_pan] << " "
				<< pan_from_y[nb_pan] << " "
				<< pan_from_z[nb_pan] << " "
				<< pan_to_x[nb_pan] << " "
				<< pan_to_y[nb_pan] << " "
				<< pan_to_z[nb_pan] << " "
				<< pan_center_x[nb_pan] << " "
				<< pan_center_y[nb_pan] << " "
				<< pan_center_z[nb_pan] << " "
				<< endl;
			nb_pan++;
			}
		else if (strcmp(argv[i], "-no_background") == 0) {
			no_background_round[nb_no_background] = atoi(argv[++i]);
			cout << "-no_background "
				<< no_background_round[nb_no_background] << endl;
			nb_no_background++;
			}
		else if (strcmp(argv[i], "-no_bottom_plane") == 0) {
			no_bottom_plane_round[nb_no_bottom_plane] = atoi(argv[++i]);
			cout << "-no_bottom_plane "
				<< no_bottom_plane_round[nb_no_bottom_plane] << endl;
			nb_no_bottom_plane++;
			}
		else if (strcmp(argv[i], "-camera") == 0) {
			camera_round[nb_camera] = atoi(argv[++i]);
			camera_sky[nb_camera] = argv[++i];
			camera_location[nb_camera] = argv[++i];
			camera_look_at[nb_camera] = argv[++i];
			cout << "-camera "
					<< camera_round[nb_camera] << " "
					<< camera_sky[nb_camera] << " "
					<< camera_location[nb_camera] << " "
					<< camera_look_at[nb_camera] << endl;
			nb_camera++;

			   //sky <1,1,1>
			   	   //direction <1,0,0>
			   	   //right <1,1,0>
			   //location  <-3,1,3>
			   //look_at  <0,0,0>
}
		else if (strcmp(argv[i], "-clipping") == 0) {
			clipping_round[nb_clipping] = atoi(argv[++i]);
			double d;
			sscanf(argv[++i], "%lf", &d);
			clipping_value[nb_clipping] = d;
			cout << "-clipping "
				<< clipping_round[nb_clipping] << " "
				<< clipping_value[nb_clipping] << endl;
			nb_clipping++;
			}
		else if (strcmp(argv[i], "-text") == 0) {
			round_text_round[nb_round_text] = atoi(argv[++i]);
			round_text_sustain[nb_round_text] = atoi(argv[++i]);
			round_text_text[nb_round_text] = argv[++i];
			cout << "-text "
				<< round_text_round[nb_round_text] << " "
				<< round_text_sustain[nb_round_text] << " "
				<< round_text_text[nb_round_text] << endl;
			nb_round_text++;
			}
		else if (strcmp(argv[i], "-label") == 0) {
			label_round[nb_label] = atoi(argv[++i]);
			label_start[nb_label] = atoi(argv[++i]);
			label_sustain[nb_label] = atoi(argv[++i]);
			label_gravity[nb_label] = argv[++i];
			label_text[nb_label] = argv[++i];
			cout << "-label "
				<< label_round[nb_label] << " "
				<< label_start[nb_label] << " "
				<< label_sustain[nb_label] << " "
				<< label_gravity[nb_label] << " "
				<< label_text[nb_label] << " "
				<< endl;
			nb_label++;
			}
		else if (strcmp(argv[i], "-latex") == 0) {
			latex_label_round[nb_latex_label] = atoi(argv[++i]);
			latex_label_start[nb_latex_label] = atoi(argv[++i]);
			latex_label_sustain[nb_latex_label] = atoi(argv[++i]);
			latex_extras_for_praeamble[nb_latex_label] = argv[++i];
			latex_label_gravity[nb_latex_label] = argv[++i];
			latex_label_text[nb_latex_label] = argv[++i];
			latex_f_label_has_been_prepared[nb_latex_label] = FALSE;
			latex_fname_base[nb_latex_label] = NEW_char(1000);
			cout << "-latex "
				<< latex_label_round[nb_latex_label] << " "
				<< latex_label_start[nb_latex_label] << " "
				<< latex_label_sustain[nb_latex_label] << " "
				<< latex_extras_for_praeamble[nb_latex_label] << " "
				<< latex_label_gravity[nb_latex_label] << " "
				<< latex_label_text[nb_latex_label] << " "
				<< endl;
			nb_latex_label++;
			}
		else if (strcmp(argv[i], "-global_picture_scale") == 0) {
			f_has_global_picture_scale = TRUE;
			double d;
			sscanf(argv[++i], "%lf", &d);
			global_picture_scale = d;
			cout << "-global_picture_scale " << d << endl;
			}
		else if (strcmp(argv[i], "-picture") == 0) {
			picture_round[nb_picture] = atoi(argv[++i]);
			double d;
			sscanf(argv[++i], "%lf", &d);
			picture_scale[nb_picture] = d;
			picture_fname[nb_picture] = argv[++i];
			picture_options[nb_picture] = argv[++i];
			cout << "-picture "
				<< picture_round[nb_picture] << " "
				<< picture_scale[nb_picture] << " "
				<< picture_fname[nb_picture] << " "
				<< picture_options[nb_picture] << " "
				<< endl;
			nb_picture++;
			}
		else if (strcmp(argv[i], "-look_at") == 0) {
			look_at = argv[++i];
			cout << "-look_at " << look_at << endl;
			}

		else if (strcmp(argv[i], "-default_angle") == 0) {
			f_default_angle = TRUE;
			default_angle = atoi(argv[++i]);
			cout << "-default_angle " << default_angle << endl;
			}
		else if (strcmp(argv[i], "-clipping_radius") == 0) {
			f_clipping_radius = TRUE;
			sscanf(argv[++i], "%lf", &clipping_radius);
			cout << "-clipping_radius " << clipping_radius << endl;
			}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			return i;
			}
		else {
			cout << "video_draw_options::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	}
	return 0;
}



}}
