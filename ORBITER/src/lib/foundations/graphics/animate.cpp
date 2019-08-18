/*
 * animate.cpp
 *
 *  Created on: Mar 13, 2019
 *      Author: betten
 */



#include "foundations.h"
#include <math.h>


using namespace std;


namespace orbiter {
namespace foundations {


animate::animate()
{
	S = NULL;
	output_mask = NULL;
	fname_makefile[0] = 0;
	nb_frames = 30;
	Opt = NULL;
	fpm = NULL;
	draw_frame_callback = NULL;
	extra_data = NULL;
}

animate::~animate()
{


}

void animate::init(scene *S,
		const char *output_mask,
		int nb_frames,
		video_draw_options *Opt,
		void *extra_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "animate::init" << endl;
	}
	animate::S = S;
	animate::output_mask = output_mask;
	animate::nb_frames = nb_frames;
	animate::Opt = Opt;
	animate::extra_data = extra_data;
	sprintf(fname_makefile, "makefile_animation");


	if (f_v) {
		cout << "animate::init done" << endl;
	}
}


void animate::animate_one_round(
	int round,
	int verbose_level)
{
	numerics N;
	int h, i, j;
	int f_has_camera = FALSE;
	const char *camera_sky = NULL;
	const char *camera_location = NULL;
	const char *camera_look_at = NULL;
	int f_has_zoom = FALSE;
	int zoom_start = 0;
	int zoom_end = 0;
	int f_has_zoom_sequence = FALSE;
	double *zoom_sequence = NULL;
	int zoom_sequence_length = 0;
	double *zoom_sequence_value = NULL;
	int *zoom_sequence_fst = NULL;
	int *zoom_sequence_len = NULL;
	int zoom_sequence_l;
	double angle;
	double zoom_increment;
	int nb_frames_this_round;
	int f_has_pan = FALSE;
	int pan_f_reverse = FALSE;
	double pan_from[3];
	double pan_to[3];
	double pan_center[3];
	double pan_u[3];
	double pan_v[3];
	double pan_w[3];
	double pan_z[3];
	double pan_normal_uv[3];
	double uu, suu;
	double uv;
	double vv, svv;
	//double ww, sww;
	double zz, szz;
	double pan_alpha, pan_delta;
	povray_interface Pov;

	nb_frames_this_round = nb_frames;
	for (i = 0; i < Opt->nb_camera; i++) {
		if (Opt->camera_round[i] == round) {
			f_has_camera = TRUE;
			camera_sky = Opt->camera_sky[i];
			camera_location = Opt->camera_location[i];
			camera_look_at = Opt->camera_look_at[i];
			}
		}
	for (i = 0; i < Opt->cnt_nb_frames; i++) {
		if (Opt->nb_frames_round[i] == round) {
			nb_frames_this_round = Opt->nb_frames_value[i];
			}
		}
	for (i = 0; i < Opt->nb_zoom; i++) {
		if (Opt->zoom_round[i] == round) {
			f_has_zoom = TRUE;
			zoom_start = Opt->zoom_start[i];
			zoom_end = Opt->zoom_end[i];
			zoom_increment = (double)(zoom_end - zoom_start) /
					(double) nb_frames_this_round;
			}
		}
	for (i = 0; i < Opt->nb_zoom_sequence; i++) {
		if (Opt->zoom_sequence_round[i] == round) {
			f_has_zoom_sequence = TRUE;
			N.vec_scan(Opt->zoom_sequence_text[i],
					zoom_sequence, zoom_sequence_length);
			cout << "round " << round << " : zoom_sequence ";
			N.vec_print(cout, zoom_sequence, zoom_sequence_length);
			cout << endl;
			if (EVEN(zoom_sequence_length)) {
				cout << "zoom_sequence mast have odd length" << endl;
				exit(1);
			}
			zoom_sequence_value = new double[zoom_sequence_length];
			zoom_sequence_fst = NEW_int(zoom_sequence_length);
			zoom_sequence_len = NEW_int(zoom_sequence_length);
			zoom_sequence_l = zoom_sequence_length >> 1;
			zoom_sequence_fst[0] = 0;
			for (j = 0; j <= zoom_sequence_l; j++) {
				zoom_sequence_value[j] = zoom_sequence[2 * j];
				if (j < zoom_sequence_l) {
					zoom_sequence_fst[j + 1] =
							zoom_sequence_fst[j] +
							(int) zoom_sequence[2 * j + 1];
					zoom_sequence_len[j] =
							zoom_sequence_fst[j + 1] -
							zoom_sequence_fst[j];
				}
			}
			cout << "zoom sequence: " << endl;
			for (j = 0; j < zoom_sequence_l; j++) {
				cout << zoom_sequence_fst[j] << " - "
						<< zoom_sequence_len[j] << " : "
						<< zoom_sequence_value[j] << endl;
			}
			cout << zoom_sequence_fst[zoom_sequence_l] << endl;
		}
	}
	for (i = 0; i < Opt->nb_pan; i++) {
		if (Opt->pan_round[i] == round) {
			f_has_pan = TRUE;
			pan_f_reverse = Opt->pan_f_reverse[i];
			pan_from[0] = Opt->pan_from_x[i];
			pan_from[1] = Opt->pan_from_y[i];
			pan_from[2] = Opt->pan_from_z[i];
			pan_to[0] = Opt->pan_to_x[i];
			pan_to[1] = Opt->pan_to_y[i];
			pan_to[2] = Opt->pan_to_z[i];
			pan_center[0] = Opt->pan_center_x[i];
			pan_center[1] = Opt->pan_center_y[i];
			pan_center[2] = Opt->pan_center_z[i];

			cout << "pan_from: ";
			N.vec_print(pan_from, 3);
			cout << endl;
			cout << "pan_to: ";
			N.vec_print(pan_to, 3);
			cout << endl;
			cout << "pan_center: ";
			N.vec_print(pan_center, 3);
			cout << endl;
			//zoom_increment = (double)(zoom_end - zoom_start) /
			//		(double) nb_frames_this_round;
			pan_u[0] = pan_from[0] - pan_center[0];
			pan_u[1] = pan_from[1] - pan_center[1];
			pan_u[2] = pan_from[2] - pan_center[2];
			pan_v[0] = pan_to[0] - pan_center[0];
			pan_v[1] = pan_to[1] - pan_center[1];
			pan_v[2] = pan_to[2] - pan_center[2];
			cout << "pan_u: ";
			N.vec_print(pan_u, 3);
			cout << endl;
			cout << "pan_v: ";
			N.vec_print(pan_v, 3);
			cout << endl;
			uu = N.dot_product(pan_u, pan_u, 3);
			uv = N.dot_product(pan_u, pan_v, 3);
			vv = N.dot_product(pan_v, pan_v, 3);
			suu = sqrt(uu);
			svv = sqrt(vv);
			if (ABS(vv) < 0.01) {
				cout << "animate vector v is too short" << endl;
				exit(1);
			}
			N.vec_linear_combination(1., pan_u,
					-1 * uv / vv, pan_v, pan_w, 3);

			cout << "pan_w: ";
			N.vec_print(pan_w, 3);
			cout << endl;



			N.vec_scalar_multiple(pan_u, 1./suu, 3);
			N.vec_scalar_multiple(pan_v, 1./svv, 3);


#if 1
			cout << "pan_u (normalized): ";
			N.vec_print(pan_u, 3);
			cout << endl;
			cout << "pan_v (normalized): ";
			N.vec_print(pan_v, 3);
			cout << endl;
			uu = N.dot_product(pan_u, pan_u, 3);
			uv = N.dot_product(pan_u, pan_v, 3);
			vv = N.dot_product(pan_v, pan_v, 3);
#endif

			pan_alpha = acos(uv /* / (suu * svv)*/);
			cout << "pan_alpha=" << pan_alpha << " = "
					<< pan_alpha * 180. / M_PI << " deg" << endl;
			pan_delta = pan_alpha / (double) (nb_frames_this_round - 1);
			cout << "pan_delta=" << pan_delta << " = "
					<< pan_delta * 180. / M_PI << " deg" << endl;


			N.cross_product(pan_u, pan_v, pan_normal_uv);

			//ww = dot_product(pan_w, pan_w, 3);
			//sww = sqrt(ww);
			//double_vec_scalar_multiple(pan_w, 1./sww, 3);

			double wv;

			wv = N.dot_product(pan_w, pan_v, 3);
			cout << "wv=" << wv << endl;
			if (ABS(wv) > 0.01) {
				cout << "error w is not orthogonal to v" << endl;
			}


			N.vec_linear_combination1(1/sin(pan_alpha) /*suu / sww*/, pan_w,
					pan_z, 3);
			cout << "pan_z: ";
			N.vec_print(pan_z, 3);
			cout << endl;
			zz = N.dot_product(pan_z, pan_z, 3);
			szz = sqrt(zz);

			N.vec_scalar_multiple(pan_z, 1./szz, 3);

			}
		}
	int f_with_background = TRUE;
	for (i = 0; i < Opt->nb_no_background; i++) {
		if (Opt->no_background_round[i] == round) {
			f_with_background = FALSE;
			}
		}


	for (h = 0; h < nb_frames_this_round; h++) {

		char fname_pov[1000];
		char fname_png[1000];
		char povray_opts[1000];


		povray_opts[0] = 0;
		if (Opt->f_W) {
			sprintf(povray_opts + strlen(povray_opts), "-W%d ", Opt->W);
			}
		if (Opt->f_H) {
			sprintf(povray_opts + strlen(povray_opts), "-H%d ", Opt->H);
			}
		//sprintf(povray_opts, "");
		// for instance -W1920 -H1200  for larger pictures
		sprintf(fname_pov, output_mask, round, h);
		sprintf(fname_png, output_mask, round, h);
		replace_extension_with(fname_png, ".png");

		cout << "round " << round << ", frame " << h << " / "
				<< nb_frames_this_round << " in " << fname_pov << endl;
		*fpm << "\t/usr/local/povray/bin/povray " << povray_opts
				<< " " << fname_pov << endl;

		{
		ofstream fp(fname_pov);

		if (f_has_zoom) {
			angle = ((double)zoom_start + (double) h * zoom_increment);
		}
		else {
			if (f_has_zoom_sequence) {
				angle = 0;
				for (j = 0; j < zoom_sequence_l; j++) {
					if (h >= zoom_sequence_fst[j] &&
						h < zoom_sequence_fst[j] + zoom_sequence_len[j]) {
						angle = zoom_sequence_value[j] +
							(h - zoom_sequence_fst[j]) *
							(zoom_sequence_value[j + 1] - zoom_sequence_value[j])
							/ zoom_sequence_len[j];
						break;
					}
				}
				if (j == zoom_sequence_l) {
					cout << "cound not find frame " << h << " in zoom sequence" << endl;
					exit(1);
				}
				if (zoom_sequence_fst[zoom_sequence_l] != nb_frames_this_round) {
					cout << "zoom_sequence the frames dont add up" << endl;
					cout << "have=" << zoom_sequence_fst[zoom_sequence_l] << endl;
					cout << "should have " << nb_frames_this_round << endl;
					exit(1);
				}

			} else {
				angle = Opt->default_angle;
			}
		}
		cout << "frame " << h << " / " << nb_frames_this_round
				<< ", angle " << angle << endl;

		if (f_has_pan) {
			double pan_a[3];
			double sky[3];
			double location[3];
			double direction_of_view[3];
			double beta;
			char sky_string[1000];
			char location_string[1000];
			char look_at_string[1000];

			if (pan_f_reverse) {
				beta = pan_alpha - pan_delta *
						(double) (nb_frames_this_round - 1 - h);
			} else {
				beta = pan_alpha - pan_delta * (double) h;
			}
			cout << "h=" << h << " / " << nb_frames_this_round
					<< " beta=" << beta << endl;

			N.vec_linear_combination(
					cos(beta) * suu, pan_v,
					sin(beta) * szz, pan_z,
					pan_a, 3);
			cout << "pan_a: ";
			N.vec_print(pan_a, 3);
			cout << endl;
			cout << "pan_u: ";
			N.vec_print(pan_u, 3);
			cout << endl;


			N.vec_linear_combination3(
					cos(beta) * suu, pan_v,
					sin(beta) * szz, pan_z,
					1., pan_center,
					location, 3);

			sprintf(location_string, "<%lf,%lf,%lf>",
					location[0], location[1], location[2]);
			cout << "location_string=" << location_string << endl;



			N.vec_linear_combination(
					-1., location,
					1., pan_center,
					direction_of_view, 3);

			N.cross_product(direction_of_view, pan_normal_uv, sky);
			sprintf(sky_string, "<%lf,%lf,%lf>",
					sky[0], sky[1], sky[2]);
			cout << "sky_string=" << sky_string << endl;

			sprintf(look_at_string, "<%lf,%lf,%lf>",
					pan_center[0], pan_center[1], pan_center[2]);
			cout << "look_at_string=" << look_at_string << endl;


			Pov.beginning(fp,
					angle,
					sky_string,
					location_string,
					look_at_string,
					f_with_background);

		} else {
			if (f_has_camera) {
				Pov.beginning(fp,
						angle,
						camera_sky,
						camera_location,
						camera_look_at,
						f_with_background);
			} else {
				Pov.beginning(fp,
						angle,
						Opt->sky,
						Opt->location,
						Opt->look_at,
						f_with_background);
			}
		}


		if (draw_frame_callback == NULL) {
			cout << "draw_frame_callback == NULL" << endl;
			exit(1);
		}
		(*draw_frame_callback)(this, h /* frame */,
							nb_frames_this_round, round,
							fp,
							verbose_level);



		if (Opt->f_omit_bottom_plane) {
			}
		else {
			int f_has_bottom_plane = TRUE;
			for (i = 0; i < Opt->nb_no_bottom_plane; i++) {
				if (Opt->no_bottom_plane_round[i] == round) {
					f_has_bottom_plane = FALSE;
					}
				}

			if (f_has_bottom_plane) {
				Pov.bottom_plane(fp);
			}
			}
		}
		file_io Fio;

		cout << "Written file " << fname_pov << " of size "
				<< Fio.file_size(fname_pov) << endl;


		for (i = 0; i < Opt->nb_picture; i++) {
			if (Opt->picture_round[i] == round) {
				char cmd[1000];
				double scale;

				scale = Opt->picture_scale[i];
				if (Opt->f_has_global_picture_scale) {
					scale *= Opt->global_picture_scale;
					}
				scale *= 100.;
				sprintf(cmd, "composite \\( %s "
						"-resize %lf%% \\)  %s    %s   tmp.png",
					Opt->picture_fname[i],
					scale, //Opt->picture_scale[i] * 100.,
					Opt->picture_options[i],
					fname_png);
				//cout << "system: " << cmd << endl;
				//system(cmd);
				*fpm << "\t" << cmd << endl;

				sprintf(cmd, "mv tmp.png %s", fname_png);
				//cout << "system: " << cmd << endl;
				//system(cmd);
				*fpm << "\t" << cmd << endl;
				}
			}
		for (i = 0; i < Opt->nb_round_text; i++) {

			if (Opt->round_text_round[i] == round) {
				char str[1000];
				char cmd[1000];

				strcpy(str, Opt->round_text_text[i]);
				if (strlen(str) > h) {
					str[h] = 0;
					}

				if (strlen(str) + Opt->round_text_sustain[i] > h &&
						strlen(str) && !is_all_whitespace(str)) {
					int font_size = 36;
					int stroke_width = 1;

					if (Opt->f_has_font_size) {
						font_size = Opt->font_size;
						}
					if (Opt->f_has_stroke_width) {
						stroke_width = Opt->stroke_width;
						}
					sprintf(cmd, "convert -background none  -fill white "
							"-stroke black -strokewidth %d -font "
							"Courier-10-Pitch-Bold  -pointsize %d   "
							"label:'%s'   overlay.png",
							stroke_width, font_size, str);
					//cout << "system: " << cmd << endl;
					//system(cmd);
					*fpm << "\t" << cmd << endl;


					sprintf(cmd, "composite -gravity center overlay.png  "
							" %s   tmp.png", fname_png);
					//cout << "system: " << cmd << endl;
					//system(cmd);
					*fpm << "\t" << cmd << endl;

					sprintf(cmd, "mv tmp.png %s", fname_png);
					//cout << "system: " << cmd << endl;
					//system(cmd);
					*fpm << "\t" << cmd << endl;
					}
				}
			} // end round text

		for (i = 0; i < Opt->nb_label; i++) {

			if (Opt->label_round[i] == round) {
				char str[1000];
				char cmd[1000];

				strcpy(str, Opt->label_text[i]);

				if (h >= Opt->label_start[i]
					&& h < Opt->label_start[i] + Opt->label_sustain[i]) {
					int font_size = 36;
					int stroke_width = 1;

					if (Opt->f_has_font_size) {
						font_size = Opt->font_size;
						}
					if (Opt->f_has_stroke_width) {
						stroke_width = Opt->stroke_width;
						}
					sprintf(cmd, "convert -background none  -fill white "
							"-stroke black -strokewidth %d -font "
							"Courier-10-Pitch-Bold  -pointsize %d   "
							"label:'%s'   overlay.png",
							stroke_width, font_size, str);
					//cout << "system: " << cmd << endl;
					//system(cmd);
					*fpm << "\t" << cmd << endl;


					sprintf(cmd, "composite %s overlay.png   %s   tmp.png",
							Opt->label_gravity[i], fname_png);
					//cout << "system: " << cmd << endl;
					//system(cmd);
					*fpm << "\t" << cmd << endl;

					sprintf(cmd, "mv tmp.png %s", fname_png);
					//cout << "system: " << cmd << endl;
					//system(cmd);
					*fpm << "\t" << cmd << endl;
					}
				}
			} // end label


		for (i = 0; i < Opt->nb_latex_label; i++) {

			if (Opt->latex_label_round[i] == round) {

				if (h >= Opt->latex_label_start[i]
					&& h < Opt->latex_label_start[i] +
						Opt->latex_label_sustain[i]) {


					if (!Opt->latex_f_label_has_been_prepared[i]) {

						cout << "creating latex label " << i << endl;
						sprintf(Opt->latex_fname_base[i],
								output_mask, round, h);
						chop_off_extension(Opt->latex_fname_base[i]);
						sprintf(Opt->latex_fname_base[i] +
							strlen(Opt->latex_fname_base[i]),
							"_%04d", i);

						cout << "latex_fname_base=" <<
								Opt->latex_fname_base[i] << endl;
						char cmd[1000];
						char fname_tex[1000];
						char fname_pdf[1000];

						sprintf(fname_tex, "%s.tex", Opt->latex_fname_base[i]);
						sprintf(fname_pdf, "%s.pdf", Opt->latex_fname_base[i]);

						cout << "begin latex source:" << endl;
						cout << Opt->latex_label_text[i] << endl;
						cout << "end latex source" << endl;
						{
							ofstream fp(fname_tex);
							latex_interface L;
							//latex_head_easy(fp);
							L.head_easy_with_extras_in_the_praeamble(fp,
									Opt->latex_extras_for_praeamble[i]);
							fp << Opt->latex_label_text[i] << endl;
							L.foot(fp);


						}

						sprintf(cmd, "pdflatex %s", fname_tex);
						//cout << "system: " << cmd << endl;
						system(cmd);
						//fpm << "\t" << cmd << endl;

						Opt->latex_f_label_has_been_prepared[i] = TRUE;
						}
					else {

						char cmd[1000];
						char fname_pdf[1000];
						char fname_label_png[1000];

						sprintf(fname_pdf, "%s.pdf",
								Opt->latex_fname_base[i]);
						sprintf(fname_label_png, "label.png");

						sprintf(cmd, "convert -trim %s %s",
								fname_pdf, fname_label_png);
						//cout << "system: " << cmd << endl;
						//system(cmd);
						*fpm << "\t" << cmd << endl;


						sprintf(cmd, "composite %s %s   %s   tmp.png",
								Opt->latex_label_gravity[i],
								fname_label_png, fname_png);
						//cout << "system: " << cmd << endl;
						//system(cmd);
						*fpm << "\t" << cmd << endl;

						sprintf(cmd, "mv tmp.png %s", fname_png);
						//cout << "system: " << cmd << endl;
						//system(cmd);
						*fpm << "\t" << cmd << endl;
						}

					//Opt->latex_file_count++;
					}
				}
			} // end label



		}


}




}}



