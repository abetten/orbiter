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
	//output_mask;
	fname_makefile[0] = 0;
	nb_frames = 30;
	Opt = NULL;
	fpm = NULL;
	draw_frame_callback = NULL;
	extra_data = NULL;
	Pov = NULL;
}

animate::~animate()
{


}

void animate::init(scene *S,
		std::string &output_mask,
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
	animate::output_mask.assign(output_mask);
	animate::nb_frames = nb_frames;
	animate::Opt = Opt;
	animate::extra_data = extra_data;
	Pov = NEW_OBJECT(povray_interface);
	sprintf(fname_makefile, "makefile_animation");


	if (f_v) {
		cout << "animate::init done" << endl;
	}
}


void animate::animate_one_round(
	int round,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	numerics N;
	int h, i, j;
	int f_has_camera = FALSE;
	double camera_sky[3];
	double camera_location[3];
	double camera_look_at[3];
	int f_has_zoom = FALSE;
	int zoom_start = 0;
	int zoom_end = 0;
	double zoom_clipping_start = 0.;
	double zoom_clipping_end = 0.;
	int f_has_zoom_sequence = FALSE;
	double *zoom_sequence = NULL;
	int zoom_sequence_length = 0;
	double *zoom_sequence_value = NULL;
	int *zoom_sequence_fst = NULL;
	int *zoom_sequence_len = NULL;
	int zoom_sequence_l = 0;
	double angle = 0;
	double clipping_radius = 0;
	double zoom_increment = 0;
	double zoom_clipping_increment = 0;
	int nb_frames_this_round = 0;
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
	double uu, suu = 0;
	double uv;
	double vv, svv = 0;
	//double ww, sww;
	double zz, szz = 0;
	double pan_alpha = 0, pan_delta = 0;

	if (f_v) {
		cout << "animate::animate_one_round" << endl;
		cout << "nb_frames=" << nb_frames << endl;
	}
	nb_frames_this_round = nb_frames;
	for (i = 0; i < Opt->nb_camera; i++) {
		if (Opt->camera_round[i] == round) {
			f_has_camera = TRUE;
			for (j = 0; j < 3; j++) {
				camera_sky[j] = Opt->camera_sky[i * 3 + j];
				camera_location[j] = Opt->camera_location[i * 3 + j];
				camera_look_at[j] = Opt->camera_look_at[i * 3 + j];
			}
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
			zoom_clipping_start = Opt->zoom_clipping_start[i];
			zoom_clipping_end = Opt->zoom_clipping_end[i];
			zoom_increment = (double)(zoom_end - zoom_start) /
					(double) nb_frames_this_round;
			zoom_clipping_increment = (double)(zoom_clipping_end - zoom_clipping_start) /
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
			pan_from[0] = Opt->pan_from[i * 3 + 0];
			pan_from[1] = Opt->pan_from[i * 3 + 1];
			pan_from[2] = Opt->pan_from[i * 3 + 2];
			pan_to[0] = Opt->pan_to[i * 3 + 0];
			pan_to[1] = Opt->pan_to[i * 3 + 1];
			pan_to[2] = Opt->pan_to[i * 3 + 2];
			pan_center[0] = Opt->pan_center[i * 3 + 0];
			pan_center[1] = Opt->pan_center[i * 3 + 1];
			pan_center[2] = Opt->pan_center[i * 3 + 2];

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
		string_tools ST;


		povray_opts[0] = 0;
		if (Opt->f_W) {
			sprintf(povray_opts + strlen(povray_opts), "-W%d ", Opt->W);
			}
		if (Opt->f_H) {
			sprintf(povray_opts + strlen(povray_opts), "-H%d ", Opt->H);
			}
		//sprintf(povray_opts, "");
		// for instance -W1920 -H1200  for larger pictures
		sprintf(fname_pov, output_mask.c_str(), round, h);
		sprintf(fname_png, output_mask.c_str(), round, h);
		ST.replace_extension_with(fname_png, ".png");

		cout << "round " << round << ", frame " << h << " / "
				<< nb_frames_this_round << " in " << fname_pov << endl;
		*fpm << "\t/usr/local/povray/bin/povray " << povray_opts
				<< " " << fname_pov << endl;

		{
		ofstream fp(fname_pov);


		if (Opt->f_clipping_radius) {
			clipping_radius = Opt->clipping_radius;
		}
		else {
			clipping_radius = 2.7; // default
		}
		for (i = 0; i < Opt->nb_clipping; i++) {
			if (Opt->clipping_round[i] == round) {
				clipping_radius = Opt->clipping_value[i];
				}
			}

		if (f_has_zoom) {
			angle = ((double)zoom_start + (double) h * zoom_increment);
			clipping_radius = zoom_clipping_start + (double) h * zoom_clipping_increment;
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

			}
			else {
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
			//char sky_string[1000];
			//char location_string[1000];
			//char look_at_string[1000];

			if (pan_f_reverse) {
				beta = pan_alpha - pan_delta *
						(double) (nb_frames_this_round - 1 - h);
			}
			else {
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

			//sprintf(location_string, "<%lf,%lf,%lf>",
			//		location[0], location[1], location[2]);
			//cout << "location_string=" << location_string << endl;



			N.vec_linear_combination(
					-1., location,
					1., pan_center,
					direction_of_view, 3);

			N.cross_product(direction_of_view, pan_normal_uv, sky);
			//sprintf(sky_string, "<%lf,%lf,%lf>",
			//		sky[0], sky[1], sky[2]);
			//cout << "sky_string=" << sky_string << endl;

			//sprintf(look_at_string, "<%lf,%lf,%lf>",
			//		pan_center[0], pan_center[1], pan_center[2]);
			//cout << "look_at_string=" << look_at_string << endl;


			Pov->beginning(fp,
					angle,
					sky,
					location,
					pan_center /* look_at*/,
					//sky_string,
					//location_string,
					//look_at_string,
					f_with_background);

		}
		else {
			if (f_has_camera) {
				Pov->beginning(fp,
						angle,
						camera_sky,
						camera_location,
						camera_look_at,
						f_with_background);
			}
			else {
				Pov->beginning(fp,
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
							clipping_radius,
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
				Pov->bottom_plane(fp);
			}
			}
		}
		file_io Fio;

		cout << "Written file " << fname_pov << " of size "
				<< Fio.file_size(fname_pov) << endl;


		for (i = 0; i < Opt->nb_picture; i++) {
			if (Opt->picture_round[i] == round) {
				char cmd[5000];
				double scale;

				scale = Opt->picture_scale[i];
				if (Opt->f_has_global_picture_scale) {
					scale *= Opt->global_picture_scale;
					}
				scale *= 100.;
				snprintf(cmd, 5000, "composite \\( %s "
						"-resize %lf%% \\)  %s    %s   tmp.png",
					Opt->picture_fname[i].c_str(),
					scale, //Opt->picture_scale[i] * 100.,
					Opt->picture_options[i].c_str(),
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
				char str[2000];

				strcpy(str, Opt->round_text_text[i].c_str());
				if ((int) strlen(str) > h) {
					str[h] = 0;
					}

				if ((int) strlen(str) + Opt->round_text_sustain[i] > h &&
						(int) strlen(str) && !is_all_whitespace(str)) {
					int font_size = 36;
					int stroke_width = 1;

					if (Opt->f_has_font_size) {
						font_size = Opt->font_size;
						}
					if (Opt->f_has_stroke_width) {
						stroke_width = Opt->stroke_width;
						}

					char cmd[10000];

					snprintf(cmd, 10000, "convert -background none  -fill white "
							"-stroke black -strokewidth %d -font "
							"Courier-10-Pitch-Bold  -pointsize %d   "
							"label:'%s'   overlay.png",
							stroke_width, font_size, str);
					//cout << "system: " << cmd << endl;
					//system(cmd);
					*fpm << "\t" << cmd << endl;


					snprintf(cmd, 10000, "composite -gravity center overlay.png  "
							" %s   tmp.png", fname_png);
					//cout << "system: " << cmd << endl;
					//system(cmd);
					*fpm << "\t" << cmd << endl;

					snprintf(cmd, 10000, "mv tmp.png %s", fname_png);
					//cout << "system: " << cmd << endl;
					//system(cmd);
					*fpm << "\t" << cmd << endl;
					}
				}
			} // end round text

		for (i = 0; i < Opt->nb_label; i++) {

			if (Opt->label_round[i] == round) {
				string label;
				string cmd;
				char str[1000];

				label.assign(Opt->label_text[i]);

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
					snprintf(str, 1000, "convert -background none  -fill white "
							"-stroke black -strokewidth %d -font "
							"Courier-10-Pitch-Bold  -pointsize %d   "
							"label:'%s'   overlay.png",
							stroke_width, font_size, label.c_str());
					cmd.assign(str);
					//cout << "system: " << cmd << endl;
					//system(cmd);
					*fpm << "\t" << cmd << endl;


					snprintf(str, 1000, "composite %s overlay.png   %s   tmp.png",
							Opt->label_gravity[i].c_str(), fname_png);
					cmd.assign(str);
					//cout << "system: " << cmd << endl;
					//system(cmd);
					*fpm << "\t" << cmd << endl;

					snprintf(str, 10000, "mv tmp.png %s", fname_png);
					cmd.assign(str);
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

						char str[1000];
						string_tools ST;


						cout << "creating latex label " << i << endl;
						sprintf(str, output_mask.c_str(), round, h);

						Opt->latex_fname_base[i].assign(str);

						ST.chop_off_extension(Opt->latex_fname_base[i]);
						sprintf(str, "_%04d", i);
						Opt->latex_fname_base[i].append(str);

						//sprintf(Opt->latex_fname_base[i] +
						//	strlen(Opt->latex_fname_base[i]), "_%04d", i);

						cout << "latex_fname_base=" <<
								Opt->latex_fname_base[i] << endl;
						string cmd;
						string fname_tex;
						string fname_pdf;

						fname_tex.assign(Opt->latex_fname_base[i]);
						fname_tex.append(".tex");
						fname_pdf.assign(Opt->latex_fname_base[i]);
						fname_pdf.append(".pdf");
						//snprintf(fname_tex, 2000, "%s.tex", Opt->latex_fname_base[i]);
						//snprintf(fname_pdf, 2000, "%s.pdf", Opt->latex_fname_base[i]);

						cout << "begin latex source:" << endl;
						cout << Opt->latex_label_text[i] << endl;
						cout << "end latex source" << endl;
						{
							ofstream fp(fname_tex);
							latex_interface L;
							//latex_head_easy(fp);
							L.head_easy_with_extras_in_the_praeamble(fp,
									Opt->latex_extras_for_praeamble[i].c_str());
							fp << Opt->latex_label_text[i] << endl;
							L.foot(fp);


						}

						cmd.assign("pdflatex ");
						cmd.append(fname_tex);

						//snprintf(cmd, 10000, "pdflatex %s", fname_tex);
						//cout << "system: " << cmd << endl;
						system(cmd.c_str());
						//fpm << "\t" << cmd << endl;

						Opt->latex_f_label_has_been_prepared[i] = TRUE;
						}
					else {

						string cmd;
						string fname_pdf;
						string fname_label_png;

						fname_pdf.assign(Opt->latex_fname_base[i]);
						fname_pdf.append(".pdf");
						fname_label_png.assign("label.png");

						cmd.assign("convert -trim ");
						cmd.append(fname_pdf);
						cmd.append(" ");
						cmd.append(fname_label_png);
						//cout << "system: " << cmd << endl;
						//system(cmd.c_str());
						*fpm << "\t" << cmd << endl;


						cmd.assign("composite ");
						cmd.append(Opt->latex_label_gravity[i]);
						cmd.append(" ");
						cmd.append(fname_label_png);
						cmd.append(" ");
						cmd.append(fname_png);
						cmd.append(" tmp.png");
						//cout << "system: " << cmd << endl;
						//system(cmd);
						*fpm << "\t" << cmd << endl;

						cmd.assign("mv tmp.png ");
						cmd.append(fname_png);
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


void animate::draw_single_line(int line_idx, std::string &color, ostream &fp)
{
	int s[1];

	s[0] = line_idx;
	S->draw_lines_with_selection(s, 1, color, fp);
}

void animate::draw_single_quadric(int idx, std::string &color, ostream &fp)
{
	int s[1];

	s[0] = idx;
	S->draw_quadric_with_selection(s, 1, color, fp);
}

void animate::draw_single_surface(int surface_idx, ostream &fp)
{
	int s[1];

	s[0] = surface_idx;
	S->draw_cubic_with_selection(s, 1, Pov->color_white, fp);
}

void animate::draw_single_surface_with_color(int surface_idx, std::string &color, ostream &fp)
{
	int s[1];

	s[0] = surface_idx;
	S->draw_cubic_with_selection(s, 1, color, fp);
}

void animate::draw_Hilbert_point(int point_idx, double rad,
		std::string &options, ostream &fp)
{
	int s[1];

	s[0] = point_idx;
	S->draw_points_with_selection(s, 1, rad, options, fp);
}

void animate::draw_Hilbert_line(int line_idx, std::string &color, ostream &fp)
{
	int s[1];

	s[0] = line_idx;
	S->draw_lines_with_selection(s, 1,
		color,
		fp);
}

void animate::draw_Hilbert_plane(int plane_idx, std::string &color, ostream &fp)
{
	int s[1];

	s[0] = plane_idx;
	S->draw_planes_with_selection(s, sizeof(s) / sizeof(int), color, fp);
}

void animate::draw_Hilbert_red_line(int idx_one_based, ostream &fp)
{
	int s[] = {12, 13, 14, 15, 16, 17};
	S->draw_edges_with_selection(s + idx_one_based - 1, 1, Pov->color_red, fp);
}

void animate::draw_Hilbert_blue_line(int idx_one_based, ostream &fp)
{
	int s[] = {18, 19, 20, 21, 22, 23};
	S->draw_edges_with_selection(s + idx_one_based - 1, 1, Pov->color_blue, fp);
}

void animate::draw_Hilbert_red_lines(ostream &fp)
{
	int s[] = {12, 13, 14, 15, 16, 17};
	S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_red, fp);
}

void animate::draw_Hilbert_blue_lines(ostream &fp)
{
	int s[] = {18, 19, 20, 21, 22, 23};
	S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_blue, fp);
}

void animate::draw_Hilbert_cube_extended_edges(ostream &fp)
{
	int s[] = {30,31,32,33,34,35,36,37,38,39,40,41};

	override_double line_radius(&S->line_radius, S->line_radius * 0.5);

	S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_black, fp);
}

void animate::draw_Hilbert_cube_faces(ostream &fp)
{
	int s[] = {0,1,2,3,4,5};
	S->draw_faces_with_selection(s, sizeof(s) / sizeof(int), 0.01, Pov->color_pink, fp);
}

void animate::draw_Hilbert_cube_boxed(ostream &fp)
{
	int s[] = {0,1,2,3,4,5,6,7,8,9,10,11};

	override_double line_radius(&S->line_radius, S->line_radius * 0.5);

	S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_black, fp);
}

void animate::draw_Hilbert_tetrahedron_boxed(ostream &fp)
{
	int s[] = {24,25,26,27,28,29};

	override_double line_radius(&S->line_radius, S->line_radius * 0.5);

	S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_black, fp);
}

void animate::draw_Hilbert_tetrahedron_faces(ostream &fp)
{
	int s[] = {6,7,8,9};
	S->draw_faces_with_selection(s, sizeof(s) / sizeof(int), 0.01, Pov->color_orange, fp);
}

void animate::draw_frame_Hilbert(
	int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level)
// round 0: cube + faces
// round 1: cube + faces + extended edges + red lines + blue lines
// round 2: red lines + blue lines + surface
// round 3: three blue lines
// round 4: three blue lines + quadric
// round 5: four blue lines + 1 red line + quadric
// round 6: four blue lines + 2 red lines + quadric
// round 7: four blue lines + 2 red lines
// round 8: double six (6 red + 6 blue)
// round 9, 11, 12: surface
// round 10: nothing
// round 13: cube + tetrahedron
// round 14: cube + tetrahedron + extended edges
// round 15: cube + tetrahedron + extended edges + double six
// round 16: 5 blue lines + 1 red (5 + 1)
// round 17: red + blue + tetrahedron
// round 18: red + blue + surface + tritangent plane + 3 yellow lines in it
// round 19: surface + plane
// round 20: cube + some red lines appearing
// round 21: cube + some blue lines appearing
// round 23: red + blue + surface + tritangent plane + 3 yellow lines in it + point appearing
// round 24: all yellow lines (not at infinity)
// round 25: red lines + blue lines + all yellow lines (not at infinity)
// round 26: red lines + blue lines + all yellow lines (not at infinity) + surface
// round 27, 28, 30, 31: empty
// round 29: cube + 1 red line + 1 blue line
// round 32: tritangent plane + 6 arc points
// round 33-38, 40: nothing
// round 39: surface 1 (transformed HCV)
// round 41,42: Cayley's nodal surface + 6 lines
// round 43: Clebsch surface
// round 44: Clebsch surface with lines
// round 45: Fermat surface
// round 46: Fermat surface with lines
// round 48-55: Cayley's ruled surface
// round 72: all tritangent planes one-by-one
// round 73: all red lines, 2 blue lines, surface, tritangent plane + 3 yellow lines, 6 arc points
// round 74: surface, tritangent plane + 3 yellow lines
// round 75: all red, 2 blue lines, tritangent plane + 3 yellow lines, 6 arc points (no surface)
// round 76: tritangent plane, 6 point, 2 blue lines, 6 red lines
// round 77: all red, 2 blue lines, tritangent plane + 3 yellow lines, 6 arc points, with surface
// round 78: all red, 2 blue lines, tritangent plane + 3 yellow lines, 6 arc points, with surface, trying to plot points under the Clebsch map
// round 79: like round 76
// round 80: tarun
{
	int i;
	//povray_interface Pov;


	cout << "draw_frame_Hilbert round=" << round << endl;

	double scale_factor = Opt->scale_factor;

	Pov->union_start(fp);


	if (round == 0) {
		draw_Hilbert_cube_boxed(fp);
		draw_Hilbert_cube_faces(fp);
		}

	if (round == 1) {
		draw_Hilbert_cube_boxed(fp);
		draw_Hilbert_cube_faces(fp);
		draw_Hilbert_cube_extended_edges(fp);
		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		}

	if (round == 2) {
		//draw_Hilbert_cube_boxed(S, fp);
		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		draw_single_surface(0, fp);
		}
	if (round == 3) {
		//{
		//int s[] = {12 /*, 13, 14, 15, 16, 17*/};
		//S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), color_red, fp);
		//}
		{
		int s[] = {/*18,*/ 19, 20, 21 /*, 22, 23*/};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_blue, fp);
		}
		}
	if (round == 4) {
		//{
		//int s[] = {12 /*, 13, 14, 15, 16, 17*/};
		//S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), color_red, fp);
		//}
		{
		int s[] = {/*18,*/ 19, 20, 21 /*, 22, 23*/};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_blue, fp);
		}
		draw_single_quadric(0, Pov->color_yellow_transparent, fp);
		}
	if (round == 5) {
		{
		int s[] = {12 /*, 13, 14, 15, 16, 17*/};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_red, fp);
		}
		{
		int s[] = {/*18,*/ 19, 20, 21, 22 /*, 23*/};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_blue, fp);
		}
		draw_single_quadric(0, Pov->color_yellow_transparent, fp);
		}
	if (round == 6) {
		{
		int s[] = {12 /*, 13, 14, 15, 16*/, 17};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_red, fp);
		}
		{
		int s[] = {/*18,*/ 19, 20, 21, 22 /*, 23*/};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_blue, fp);
		}
		draw_single_quadric(0, Pov->color_yellow_transparent, fp);
		}
	if (round == 7) {
		{
		int s[] = {12 /*, 13, 14, 15, 16*/, 17};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_red, fp);
		}
		{
		int s[] = {/*18,*/ 19, 20, 21, 22 /*, 23*/};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_blue, fp);
		}
		}
	if (round == 8) {
		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		}
	if (round == 9 || round == 11 || round == 12) {
		draw_single_surface(0, fp);
		}
	if (round == 10) {
		}
	if (round == 13) {
		draw_Hilbert_cube_boxed(fp);
		draw_Hilbert_tetrahedron_boxed(fp);
		draw_Hilbert_tetrahedron_faces(fp);
		}
	if (round == 14) {
		if (h < (nb_frames >> 1)) {
			draw_Hilbert_cube_boxed(fp);
			draw_Hilbert_tetrahedron_boxed(fp);
			draw_Hilbert_tetrahedron_faces(fp);
			draw_Hilbert_cube_extended_edges(fp);
			}
		else {
			draw_Hilbert_cube_boxed(fp);
			draw_Hilbert_cube_faces(fp);
			draw_Hilbert_cube_extended_edges(fp);
			}
		}
	if (round == 15) {
		draw_Hilbert_cube_boxed(fp);
		draw_Hilbert_cube_faces(fp);
		draw_Hilbert_cube_extended_edges(fp);
		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		}
	if (round == 16) {
		{
		int s[] = {12 /*, 13, 14, 15, 16, 17*/};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_red, fp);
		}
		{
		int s[] = {/*18,*/ 19, 20, 21, 22, 23};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_blue, fp);
		}
		}
	if (round == 17) {
		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		draw_Hilbert_tetrahedron_boxed(fp);
		draw_Hilbert_tetrahedron_faces(fp);
		}
	if (round == 18) {

#if 0
		draw_Hilbert_red_line(S, 1, fp); // a1
		draw_Hilbert_blue_line(S, 2, fp); // b2
		draw_Hilbert_red_line(S, 2, fp); // a2
		draw_Hilbert_blue_line(S, 1, fp); // b1
		draw_Hilbert_plane(S, 3 + 0, Pov->color_orange, fp); // a1b2
		draw_Hilbert_plane(S, 3 + 1, Pov->color_orange, fp); // a2b1
		draw_Hilbert_line(S, 0, Pov->color_yellow, fp); // c12
#endif

		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		draw_single_surface(0, fp);
		//draw_Hilbert_plane(S, 3 + 0, Pov->color_orange, fp); // a1b2
		//draw_Hilbert_plane(S, 3 + 1, Pov->color_orange, fp); // a2b1
		draw_Hilbert_plane(3 + 2, Pov->color_orange, fp); // pi_{12,34,56}
		draw_Hilbert_line(12 + 0, Pov->color_yellow, fp); // c12
		draw_Hilbert_line(12 + 9, Pov->color_yellow, fp);
		draw_Hilbert_line(12 + 14, Pov->color_yellow, fp);
		}
	if (round == 19) {
		draw_single_surface(0, fp);
		draw_Hilbert_plane(2, Pov->color_orange, fp); // Z=0
		}
	if (round == 20) {
		int nb_frames_half, nb1, j;

		nb_frames_half = nb_frames >> 1;
		nb1 = nb_frames_half / 6;

		if (h < nb_frames_half) {
			draw_Hilbert_cube_boxed(fp);
			draw_Hilbert_cube_extended_edges(fp);

			j = h / nb1;

			if (j < 6) {
				draw_Hilbert_red_line(1 + j, fp); // a{j+1}
				}

			}
		else {
			draw_Hilbert_cube_boxed(fp);
			//draw_Hilbert_cube_faces(Anim, fp);
			draw_Hilbert_cube_extended_edges(fp);
			draw_Hilbert_red_lines(fp);
			}
		}
	if (round == 21) {
		int nb_frames_half, nb1, j;

		nb_frames_half = nb_frames >> 1;
		nb1 = nb_frames_half / 6;

		if (h < nb_frames_half) {
			draw_Hilbert_cube_boxed(fp);
			draw_Hilbert_cube_extended_edges(fp);

			j = h / nb1;

			if (j < 6) {
				draw_Hilbert_blue_line(1 + j, fp); // b{j+1}
				}

			}
		else {
			draw_Hilbert_cube_boxed(fp);
			//draw_Hilbert_cube_faces(Anim, fp);
			draw_Hilbert_cube_extended_edges(fp);
			draw_Hilbert_blue_lines(fp);
			}
		}
	if (round == 22) {


		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		draw_single_surface(0, fp);
		draw_Hilbert_plane(3 + 2, Pov->color_orange, fp); // pi_{12,34,56}
		draw_Hilbert_line(12 + 0, Pov->color_yellow, fp); // c12
		draw_Hilbert_line(12 + 9, Pov->color_yellow, fp);
		draw_Hilbert_line(12 + 14, Pov->color_yellow, fp);


		int nb_frames_half, nb1, j;

		nb_frames_half = nb_frames; // >> 1;
		nb1 = nb_frames_half / 6;

		if (h < nb_frames_half) {
			j = h / nb1;

			if (j < 6) {
				draw_Hilbert_point(24 + j, 0.15, Pov->color_chrome, fp);
				}

			}
		else {
			draw_Hilbert_point(24, 0.15, Pov->color_chrome, fp);
			draw_Hilbert_point(25, 0.15, Pov->color_chrome, fp);
			draw_Hilbert_point(26, 0.15, Pov->color_chrome, fp);
			draw_Hilbert_point(27, 0.15, Pov->color_chrome, fp);
			draw_Hilbert_point(28, 0.15, Pov->color_chrome, fp);
			draw_Hilbert_point(29, 0.15, Pov->color_chrome, fp);
			}
		}
	if (round == 23) {


		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		draw_single_surface(0, fp);
		draw_Hilbert_plane(3 + 2, Pov->color_orange, fp); // pi_{12,34,56}
		draw_Hilbert_line(12 + 0, Pov->color_yellow, fp); // c12
		draw_Hilbert_line(12 + 9, Pov->color_yellow, fp);
		draw_Hilbert_line(12 + 14, Pov->color_yellow, fp);
		draw_Hilbert_point(24, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(25, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(26, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(27, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(28, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(29, 0.15, Pov->color_chrome, fp);
		}
	if (round == 24) {
		for (i = 0; i < 15; i++) {
			if (i == 2 || i == 7 || i == 11) {
				continue;
				}
			draw_Hilbert_line(12 + i, Pov->color_yellow, fp);
			}
		}
	if (round == 25) {
		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		for (i = 0; i < 15; i++) {
			if (i == 2 || i == 7 || i == 11) {
				continue;
				}
			draw_Hilbert_line(12 + i, Pov->color_yellow, fp);
			}
		}
	if (round == 26) {
		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		for (i = 0; i < 15; i++) {
			if (i == 2 || i == 7 || i == 11) {
				continue;
				}
			draw_Hilbert_line(12 + i, Pov->color_yellow, fp);
			}
		draw_single_surface(0, fp);
		}

	if (round == 27) {
		}
	if (round == 28) {
		}
	if (round == 29) {
		draw_Hilbert_cube_boxed(fp);
		draw_Hilbert_red_line(1, fp); // a{j+1}
		draw_Hilbert_blue_line(1, fp); // b{j+1}
		}
	if (round == 30) {
		}
	if (round == 31) {
		}
	if (round == 32) {
		//draw_Hilbert_red_lines(S, fp);
		//draw_Hilbert_blue_lines(S, fp);
		//draw_surface(S, 0, fp);
		draw_Hilbert_plane(3 + 2, Pov->color_orange, fp); // pi_{12,34,56}
		//draw_Hilbert_line(S, 0, Pov->color_yellow, fp); // c12
		//draw_Hilbert_line(S, 9, Pov->color_yellow, fp);
		//draw_Hilbert_line(S, 14, Pov->color_yellow, fp);
		draw_Hilbert_point(24, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(25, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(26, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(27, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(28, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(29, 0.15, Pov->color_chrome, fp);
		}
	if (round == 33) {
		}
	if (round == 34) {
		}
	if (round == 35) {
		}
	if (round == 36) {
		}
	if (round == 37) {
		}
	if (round == 38) {
		}
	if (round == 39) {
		draw_single_surface(1, fp);
		}
	if (round == 40) {
		}
	if (round == 41) {
		// Cayley's nodal surface:
		int idx0 = 27;
		draw_single_surface(1, fp);
		draw_Hilbert_line(idx0 + 0, Pov->color_red, fp);
		draw_Hilbert_line(idx0 + 1, Pov->color_red, fp);
		draw_Hilbert_line(idx0 + 2, Pov->color_red, fp);
		draw_Hilbert_line(idx0 + 3, Pov->color_red, fp);
		draw_Hilbert_line(idx0 + 4, Pov->color_red, fp);
		draw_Hilbert_line(idx0 + 5, Pov->color_red, fp);
		}
	if (round == 42) {
		// Cayley's nodal surface:
		int idx0 = 27;
		draw_single_surface(1, fp);
		draw_Hilbert_line(idx0 + 0, Pov->color_red, fp);
		draw_Hilbert_line(idx0 + 1, Pov->color_red, fp);
		draw_Hilbert_line(idx0 + 2, Pov->color_red, fp);
		draw_Hilbert_line(idx0 + 3, Pov->color_red, fp);
		draw_Hilbert_line(idx0 + 4, Pov->color_red, fp);
		draw_Hilbert_line(idx0 + 5, Pov->color_red, fp);
		}

	if (round == 43) {
		// Clebsch surface:
		draw_single_surface(2, fp);
		}
	if (round == 44) {
		// Clebsch surface with lines
		//S->line_radius = 0.04;
		draw_single_surface(2, fp);
		int red[6] = {12 + 21,12 + 22,12 + 23,12 + 24,12 + 25,12 + 26};
		S->draw_lines_with_selection(red, 6, Pov->color_red, fp);
		int blue[6] = {12 + 27,12 + 28,12 + 29,12 + 30,12 + 31,12 + 32};
		S->draw_lines_with_selection(blue, 6, Pov->color_blue, fp);
		int yellow[15] = {12 + 33,12 + 34,12 + 35,12 + 36,12 + 37,
			12 + 38,12 + 39,12 + 40,12 + 41,12 + 42,12 + 43,
			12 + 44,12 + 45,12 + 46,12 + 47};
		S->draw_lines_with_selection(yellow, 15, Pov->color_yellow, fp);
		}
	if (round == 45) {
		draw_single_surface(3, fp); // Fermat
		}
	if (round == 46) {
		draw_single_surface(3, fp); // Fermat's surface
		int red[3] = {60,61,62};
		S->draw_lines_with_selection(red, 3, Pov->color_red, fp);
		}
	if (round == 47) {
		}
	if (round == 48) {
		// Cayleys ruled surface, also due to Chasles
		draw_single_surface(6, fp);
		}
	if (round == 49) {
		// Cayleys ruled surface, also due to Chasles
		S->line_radius = 0.04;
		draw_single_surface(6, fp);
		int nb_lines0 = 63;
		int nb_lines_actual = 15;
		int *idx;

		idx = NEW_int(nb_lines_actual);
		for (i = 0; i < nb_lines_actual; i++) {
			idx[i] = nb_lines0 + i;
			}


		int nb_frames_half, nb1, nb2;

		nb_frames_half = nb_frames >> 1;
		nb1 = nb_frames_half / nb_lines_actual + 1;

		if (h < nb1 * nb_lines_actual) {
			nb2 = h / nb1;
			S->draw_lines_with_selection(idx, nb2, Pov->color_brown, fp);
			}
		else {
			S->draw_lines_with_selection(idx,
					nb_lines_actual, Pov->color_brown, fp);
			}
		FREE_int(idx);
		}
	if (round == 50) {
		// Cayleys ruled surface, also due to Chasles
		S->line_radius = 0.04;
		draw_single_surface(6, fp);
		int nb_lines0 = 63;
		int nb_lines_actual = 15;
		int *idx;

		idx = NEW_int(nb_lines_actual);
		for (i = 0; i < nb_lines_actual; i++) {
			idx[i] = nb_lines0 + i;
			}
		S->draw_lines_with_selection(idx, nb_lines_actual, Pov->color_brown, fp);
		FREE_int(idx);
		}
	if (round == 51) {
		// Cayleys ruled surface, also due to Chasles
		S->line_radius = 0.04;
		draw_single_surface(6, fp);
		int nb_lines0 = 63;
		int nb_lines_actual = 15;
		int *idx;

		idx = NEW_int(nb_lines_actual);
		for (i = 0; i < nb_lines_actual; i++) {
			idx[i] = nb_lines0 + i;
			}
		S->draw_lines_with_selection(idx, nb_lines_actual, Pov->color_brown, fp);
		draw_Hilbert_plane(2, Pov->color_orange, fp); // Z=0
		draw_Hilbert_point(38, 0.25, Pov->color_chrome, fp);
		FREE_int(idx);
		}
	if (round == 52) {
		// Cayleys ruled surface, also due to Chasles
		S->line_radius = 0.04;
		draw_single_surface(6, fp);
		int nb_lines0 = 63;
		int nb_lines_actual = 15;
		int *idx;

		idx = NEW_int(nb_lines_actual);
		for (i = 0; i < nb_lines_actual; i++) {
			idx[i] = nb_lines0 + i;
			}
		S->draw_lines_with_selection(idx, nb_lines_actual, Pov->color_brown, fp);
		draw_Hilbert_plane(2, Pov->color_orange, fp); // Z=0
		draw_Hilbert_point(38, 0.25, Pov->color_chrome, fp);
		FREE_int(idx);
		}
	if (round == 53) {
		draw_single_surface(0, fp);
		}
	if (round == 54) {
		// Cayleys ruled surface, also due to Chasles
		S->line_radius = 0.04;
		draw_single_surface(6, fp);
		int nb_lines0 = 63;
		int nb_lines_actual = 15;
		int nb_lines1 = nb_lines0 + nb_lines_actual;
		int *idx;

		idx = NEW_int(nb_lines_actual);
		for (i = 0; i < nb_lines_actual; i++) {
			idx[i] = nb_lines0 + i;
			}
		S->draw_lines_with_selection(idx, nb_lines_actual, Pov->color_brown, fp);
		draw_Hilbert_plane(2, Pov->color_orange, fp); // Z=0
		draw_Hilbert_point(38, 0.25, Pov->color_chrome, fp);
		S->draw_line_with_selection(nb_lines1, Pov->color_yellow, fp);
		FREE_int(idx);
		}
	if (round == 55) {
		// Cayleys ruled surface, also due to Chasles
		S->line_radius = 0.04;
		draw_single_surface(6, fp);
		int nb_lines0 = 63;
		int nb_lines_actual = 15;
		int nb_lines1 = nb_lines0 + nb_lines_actual;
		int *idx;

		idx = NEW_int(nb_lines_actual);
		for (i = 0; i < nb_lines_actual; i++) {
			idx[i] = nb_lines0 + i;
			}
		S->draw_lines_with_selection(idx, nb_lines_actual, Pov->color_brown, fp);
		draw_Hilbert_plane(2, Pov->color_orange, fp); // Z=0
		draw_Hilbert_point(38, 0.25, Pov->color_chrome, fp);
		S->draw_line_with_selection(nb_lines1, Pov->color_yellow, fp);
		FREE_int(idx);
		}
	if (round == 56) {
		S->line_radius = 0.04;
		//draw_Hilbert_cube_boxed(S, fp);
		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		//draw_Hilbert_plane(S, 3 + 2, Pov->color_orange, fp); // pi_{12,34,56}
		draw_Hilbert_plane(3 + 2, Pov->color_orange_no_phong, fp); // pi_{12,34,56}
		draw_Hilbert_line(12 + 0, Pov->color_yellow, fp); // c12
		draw_Hilbert_line(12 + 9, Pov->color_yellow, fp);
		draw_Hilbert_line(12 + 14, Pov->color_yellow, fp);
		//draw_surface(S, 7, fp);
		//draw_quadric(S, 1, Pov->color_gold_transparent, fp);
		//draw_quadric(S, 1, Pov->color_pink, fp);
		draw_single_quadric(1, Pov->color_pink_transparent, fp);
		draw_Hilbert_point(24, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(25, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(26, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(27, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(28, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(29, 0.15, Pov->color_chrome, fp);
		}
	if (round == 57) {
		S->line_radius = 0.04;
		//draw_Hilbert_cube_boxed(S, fp);
		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		//draw_Hilbert_plane(Anim, 3 + 2, Pov->color_orange, fp); // pi_{12,34,56}
		draw_Hilbert_plane(3 + 2, Pov->color_orange_no_phong, fp); // pi_{12,34,56}
		draw_Hilbert_line(12 + 0, Pov->color_yellow, fp); // c12
		draw_Hilbert_line(12 + 9, Pov->color_yellow, fp);
		draw_Hilbert_line(12 + 14, Pov->color_yellow, fp);
		//draw_surface(S, 7, fp);
		//draw_quadric(S, 1, Pov->color_gold_transparent, fp);
		//draw_quadric(S, 1, Pov->color_pink, fp);
		draw_single_quadric(2, Pov->color_pink_transparent, fp);
		draw_Hilbert_point(24, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(25, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(26, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(27, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(28, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(29, 0.15, Pov->color_chrome, fp);
		}
	if (round == 58) {
		S->line_radius = 0.04;
		//draw_Hilbert_cube_boxed(Anim, fp);
		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		//draw_Hilbert_plane(S, 3 + 2, Pov->color_orange, fp); // pi_{12,34,56}
		draw_Hilbert_plane(3 + 2, Pov->color_orange_no_phong, fp); // pi_{12,34,56}
		draw_Hilbert_line(12 + 0, Pov->color_yellow, fp); // c12
		draw_Hilbert_line(12 + 9, Pov->color_yellow, fp);
		draw_Hilbert_line(12 + 14, Pov->color_yellow, fp);
		//draw_surface(S, 7, fp);
		//draw_quadric(S, 1, Pov->color_gold_transparent, fp);
		//draw_quadric(S, 1, Pov->color_pink, fp);
		draw_single_quadric(3, Pov->color_pink_transparent, fp);
		draw_Hilbert_point(24, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(25, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(26, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(27, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(28, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(29, 0.15, Pov->color_chrome, fp);
		}
	if (round == 59) {
		S->line_radius = 0.04;
		//draw_Hilbert_cube_boxed(S, fp);
		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		//draw_Hilbert_plane(Anim, 3 + 2, Pov->color_orange, fp); // pi_{12,34,56}
		draw_Hilbert_plane(3 + 2, Pov->color_orange_no_phong, fp); // pi_{12,34,56}
		draw_Hilbert_line(12 + 0, Pov->color_yellow, fp); // c12
		draw_Hilbert_line(12 + 9, Pov->color_yellow, fp);
		draw_Hilbert_line(12 + 14, Pov->color_yellow, fp);
		//draw_surface(S, 7, fp);
		//draw_quadric(S, 1, Pov->color_gold_transparent, fp);
		//draw_quadric(S, 1, Pov->color_pink, fp);
		draw_single_quadric(4, Pov->color_pink_transparent, fp);
		draw_Hilbert_point(24, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(25, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(26, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(27, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(28, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(29, 0.15, Pov->color_chrome, fp);
		}
	if (round == 60) {
		draw_single_surface(7, fp); // from arc_lifting
		}
	if (round == 61) {
		draw_single_surface(7, fp); // from arc_lifting
		}
	if (round == 62) {
		draw_single_surface(7, fp); // from arc_lifting
		}
	if (round == 63) {
		draw_single_surface(8, fp); // from arc_lifting
		}
	if (round == 64) {
		draw_single_surface(8, fp); // from arc_lifting
		}
	if (round == 65) {
		draw_single_surface(8, fp); // from arc_lifting
		}
	if (round == 66) {
		draw_single_surface(0, fp); // Hilbert surface
		draw_Hilbert_plane(6, Pov->color_orange, fp); // F1
		draw_Hilbert_plane(7, Pov->color_orange, fp); // F2
		}
	if (round == 67) {
		draw_single_surface(0, fp); // Hilbert surface
		draw_Hilbert_plane(6, Pov->color_orange, fp); // F1
		draw_Hilbert_plane(7, Pov->color_orange, fp); // F2


		S->line_radius = 0.04;

		{
		int s[] = {12, 13 /*, 14, 15, 16, 17*/};
		S->draw_edges_with_selection(s, 2, Pov->color_red, fp); // a1, a2
		}
		{
		int s[] = {/*18, 19, 20,*/ 21, 22 /*, 23*/};
		S->draw_edges_with_selection(s, 2, Pov->color_blue, fp); // b4, b5
		}
		draw_Hilbert_line(12 + 3, Pov->color_yellow, fp); // c15
		draw_Hilbert_line(12 + 6, Pov->color_yellow, fp); // c24

		}
	if (round == 68) {
		draw_single_surface(0, fp); // Hilbert surface
		draw_Hilbert_plane(9, Pov->color_yellow_lemon_transparent, fp); // G1
		draw_Hilbert_plane(10, Pov->color_yellow_lemon_transparent, fp); // G2
		draw_Hilbert_plane(11, Pov->color_yellow_lemon_transparent, fp); // G2
		}
	if (round == 69) {
		draw_single_surface(0, fp); // Hilbert surface
		draw_Hilbert_plane(9, Pov->color_yellow_lemon_transparent, fp); // G1
		draw_Hilbert_plane(10, Pov->color_yellow_lemon_transparent, fp); // G2
		draw_Hilbert_plane(11, Pov->color_yellow_lemon_transparent, fp); // G2

		S->line_radius = 0.04;
		{
		int s[] = {12, 13 /*, 14, 15, 16, 17*/};
		S->draw_edges_with_selection(s, 2, Pov->color_red, fp); // a1, a2
		}
		{
		int s[] = {/*18, 19, 20,*/ 21, 22 /*, 23*/};
		S->draw_edges_with_selection(s, 2, Pov->color_blue, fp); // b4, b5
		}
		draw_Hilbert_line(12 + 3, Pov->color_yellow, fp); // c15
		draw_Hilbert_line(12 + 6, Pov->color_yellow, fp); // c24

		}
	if (round == 70) {
		draw_single_surface(0, fp); // Hilbert surface
		draw_Hilbert_plane(6, Pov->color_orange, fp); // F1
		draw_Hilbert_plane(7, Pov->color_orange, fp); // F2
		draw_Hilbert_plane(9, Pov->color_yellow_lemon_transparent, fp); // G1
		draw_Hilbert_plane(10, Pov->color_yellow_lemon_transparent, fp); // G2
		draw_Hilbert_plane(11, Pov->color_yellow_lemon_transparent, fp); // G2

		S->line_radius = 0.04;
		{
		int s[] = {12, 13 /*, 14, 15, 16, 17*/};
		S->draw_edges_with_selection(s, 2, Pov->color_red, fp); // a1, a2
		}
		{
		int s[] = {/*18, 19, 20,*/ 21, 22 /*, 23*/};
		S->draw_edges_with_selection(s, 2, Pov->color_blue, fp); // b4, b5
		}
		draw_Hilbert_line(12 + 3, Pov->color_yellow, fp); // c15
		draw_Hilbert_line(12 + 6, Pov->color_yellow, fp); // c24

		}
	if (round == 71) {
		//draw_surface(S, 0, fp); // Hilbert surface
		draw_Hilbert_plane(6, Pov->color_orange, fp); // F1
		draw_Hilbert_plane(7, Pov->color_orange, fp); // F2
		draw_Hilbert_plane(9, Pov->color_yellow_lemon_transparent, fp); // G1
		draw_Hilbert_plane(10, Pov->color_yellow_lemon_transparent, fp); // G2
		draw_Hilbert_plane(11, Pov->color_yellow_lemon_transparent, fp); // G2

		S->line_radius = 0.04;
		{
		int s[] = {12, 13 /*, 14, 15, 16, 17*/};
		S->draw_edges_with_selection(s, 2, Pov->color_red, fp); // a1, a2
		}
		{
		int s[] = {/*18, 19, 20,*/ 21, 22 /*, 23*/};
		S->draw_edges_with_selection(s, 2, Pov->color_blue, fp); // b4, b5
		}
		draw_Hilbert_line(12 + 3, Pov->color_yellow, fp); // c15
		draw_Hilbert_line(12 + 6, Pov->color_yellow, fp); // c24

		}

	if (round == 72) {
		draw_single_surface(0, fp); // Hilbert surface
		//S->line_radius = 0.04;

		int nb1, quo;

		nb1 = nb_frames / 45;

		quo = h / nb1;

		// avoid drawing the plane at infinity:
		if (quo != 37) {
			draw_Hilbert_plane(12 + quo, Pov->color_orange, fp); // tritangent plane quo
			}

		draw_Hilbert_red_lines(fp);
		draw_Hilbert_blue_lines(fp);
		for (i = 0; i < 15; i++) {
			if (i == 2 || i == 7 || i == 11) {
				continue;
				}
			draw_Hilbert_line(12 + i, Pov->color_yellow, fp);
			}

		}
	if (round == 73) {
		//S->line_radius = 0.04;
		//draw_Hilbert_cube_boxed(S, fp);
		draw_Hilbert_red_lines(fp);
		//draw_Hilbert_blue_lines(S, fp);
		{
		int s[] = {18, /* 19, 20, 21, 22,*/ 23};
		S->draw_edges_with_selection(s, 2, Pov->color_blue, fp); // b1, b6
		}
		draw_Hilbert_plane(12 + 43, Pov->color_orange, fp); // pi_{16,24,35}
		draw_Hilbert_line(12 + 4, Pov->color_yellow, fp); // c16
		draw_Hilbert_line(12 + 6, Pov->color_yellow, fp); // c24
		draw_Hilbert_line(12 + 10, Pov->color_yellow, fp); // c35
		draw_single_surface(0, fp);
		draw_Hilbert_point(31, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(32, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(33, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(34, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(36, 0.15, Pov->color_chrome, fp);
		draw_Hilbert_point(37, 0.15, Pov->color_chrome, fp);
		}
	if (round == 74) {
		//S->line_radius = 0.04;
		draw_single_surface(0, fp); // Hilbert surface

		draw_Hilbert_plane(12 + 43, Pov->color_orange, fp); // tritangent plane quo
		draw_Hilbert_line(12 + 4, Pov->color_yellow, fp); // c16
		draw_Hilbert_line(12 + 6, Pov->color_yellow, fp); // c24
		draw_Hilbert_line(12 + 10, Pov->color_yellow, fp); // c35

		}
	if (round == 75) {
		//S->line_radius = 0.04;
		//draw_Hilbert_cube_boxed(S, fp);
		draw_Hilbert_red_lines(fp);
		//draw_Hilbert_blue_lines(S, fp);
		{
		int s[] = {18, /* 19, 20, 21, 22,*/ 23};
		S->draw_edges_with_selection(s, 2, Pov->color_blue, fp); // b1, b6
		}
		draw_Hilbert_plane(12 + 43, Pov->color_orange, fp); // pi_{16,24,35}
		draw_Hilbert_line(12 + 4, Pov->color_yellow, fp); // c16
		draw_Hilbert_line(12 + 6, Pov->color_yellow, fp); // c24
		draw_Hilbert_line(12 + 10, Pov->color_yellow, fp); // c35
		//draw_surface(S, 0, fp);
		draw_Hilbert_point(31, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(32, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(33, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(34, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(36, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(37, 0.075, Pov->color_chrome, fp);
		}
	if (round == 76) {
		//S->line_radius = 0.04;
		draw_frame_Hilbert_round_76(Opt, h, nb_frames, round,
			fp, verbose_level);
		}
	if (round == 77) {

		//S->line_radius = 0.04;


		draw_Hilbert_red_lines(fp);
		//draw_Hilbert_blue_lines(S, fp);
		{
		int s[] = {18, /* 19, 20, 21, 22,*/ 23};
		S->draw_edges_with_selection(s, 2, Pov->color_blue, fp); // b1, b6
		}
		draw_Hilbert_plane(12 + 43, Pov->color_orange, fp); // pi_{16,24,35}
		draw_Hilbert_line(12 + 4, Pov->color_yellow, fp); // c16
		draw_Hilbert_line(12 + 6, Pov->color_yellow, fp); // c24
		draw_Hilbert_line(12 + 10, Pov->color_yellow, fp); // c35
		draw_single_surface_with_color(0, Pov->color_white_very_transparent, fp);
		draw_Hilbert_point(31, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(32, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(33, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(34, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(36, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(37, 0.075, Pov->color_chrome, fp);

		//draw_Hilbert_cube_boxed(S, fp);
		//draw_Hilbert_cube_faces(S,fp);

		Pov->rotate_111(h, nb_frames, fp);
		Pov->union_end(fp, scale_factor, clipping_radius);
		Pov->union_start(fp);

		//my_clipping_radius = 5 * my_clipping_radius;


#if 0
		// ToDo
		clebsch_scene *CS = (clebsch_scene *) S->extra_data;
		int n, nb;

		n = h / 3;

		if (n) {
			nb = CS->original_element_idx[n - 1];
			CS->draw_points_down_original(0, n, 0.05, Pov->color_scarlet, fp);
			CS->draw_points_up(0, nb, 0.05, Pov->color_black, fp);
			CS->draw_lines_up_original(n - 1, 1, Pov->color_gold, fp);
			}
#endif
		}

	if (round == 78) {


		//S->line_radius = 0.04;

		draw_Hilbert_red_lines(fp);
		//draw_Hilbert_blue_lines(S, fp);
		{
		int s[] = {18, /* 19, 20, 21, 22,*/ 23};
		S->draw_edges_with_selection(s, 2, Pov->color_blue, fp); // b1, b6
		}
		draw_Hilbert_plane(12 + 43, Pov->color_orange, fp); // pi_{16,24,35}
		draw_Hilbert_line(12 + 4, Pov->color_yellow, fp); // c16
		draw_Hilbert_line(12 + 6, Pov->color_yellow, fp); // c24
		draw_Hilbert_line(12 + 10, Pov->color_yellow, fp); // c35
		draw_single_surface_with_color(0, Pov->color_white_very_transparent, fp);
		draw_Hilbert_point(31, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(32, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(33, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(34, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(36, 0.075, Pov->color_chrome, fp);
		draw_Hilbert_point(37, 0.075, Pov->color_chrome, fp);

		//draw_Hilbert_cube_boxed(S, fp);
		//draw_Hilbert_cube_faces(S,fp);

		Pov->rotate_111(h, nb_frames, fp);
		Pov->union_end(fp, scale_factor, clipping_radius);
		Pov->union_start(fp);

		//my_clipping_radius = 5 * my_clipping_radius;

#if 0
		// ToDo
		clebsch_scene *CS = (clebsch_scene *) S->extra_data;
		int nb;

		nb = CS->nb_elements;

		CS->draw_points_down_original(0, CS->nb_steps, 0.05, Pov->color_scarlet, fp);
		CS->draw_points_up(0, nb, 0.05, Pov->color_black, fp);
#endif
		}
	if (round == 79) {
		//S->line_radius = 0.04;
		draw_frame_Hilbert_round_76(Opt, h, nb_frames, round,
			fp, verbose_level);
		}
	if (round == 80) {
		// for Tarun
		//S->line_radius = 0.04;
		draw_Hilbert_cube_boxed(fp);
		int orbit_reps[] = {
				0,1,2,
				0,1,3,
				0,1,4,
				0,1,5,
				0,1,6,
				0,2,4,
				0,2,5,
				0,2,6,
				0,4,5,
				0,4,6,
				0,5,6,
				2,4,5,
				2,4,6
		};
		for (i = 0; i < 13 * 3; i++) {
			orbit_reps[i]++;
		}
		int rk;
		int faces[4];
		int set[3];
		combinatorics_domain Combi;
		sorting Sorting;

		int *cur_orbit_rep = orbit_reps + 2 * 3;

		// draw 4 faces:
		rk = Combi.rank_k_subset(cur_orbit_rep, 8 /*n*/, 3 /*k*/);
		faces[0] = 10 + rk;
		for (i = 0; i < 3; i++) {
			set[0] = 0;
			set[1] = cur_orbit_rep[i];
			set[2] = cur_orbit_rep[(i + 1) % 3];
			Sorting.int_vec_heapsort(set, 3);
			rk = Combi.rank_k_subset(set, 8 /*n*/, 3 /*k*/);
			faces[1 + i] = 10 + rk;
		}
		cout << "faces=";
		Orbiter->Int_vec.print(cout, faces, 4);
		cout << endl;
		S->draw_faces_with_selection(faces, 4,
				0.01, Pov->color_pink, fp);
	}
	if (round == 81) {
			// same as round 76

		//S->line_radius = 0.04;
		draw_frame_Hilbert_round_76(Opt, h, nb_frames, round,
			fp, verbose_level);
		}
	if (round == 82) {
		//S->line_radius = 0.04;
		draw_Hilbert_red_lines(fp);
		//draw_Hilbert_blue_lines(S, fp);
		//{
		//int s[] = {18, /* 19, 20, 21, 22,*/ 23};
		//S->draw_edges_with_selection(s, 2, color_blue, fp); // b1, b6
		//}
		draw_Hilbert_plane(12 + 43, Pov->color_orange, fp); // pi_{16,24,35}
		//draw_Hilbert_line(S, 12 + 4, color_yellow, fp); // c16
		//draw_Hilbert_line(S, 12 + 6, color_yellow, fp); // c24
		//draw_Hilbert_line(S, 12 + 10, color_yellow, fp); // c35
		draw_single_surface(0, fp);
		draw_Hilbert_point(31, 0.12, Pov->color_chrome, fp);
		draw_Hilbert_point(32, 0.12, Pov->color_chrome, fp);
		draw_Hilbert_point(33, 0.12, Pov->color_chrome, fp);
		draw_Hilbert_point(34, 0.12, Pov->color_chrome, fp);
		draw_Hilbert_point(36, 0.12, Pov->color_chrome, fp);
		draw_Hilbert_point(37, 0.12, Pov->color_chrome, fp);

	}
	if (round == 83) {

		//S->line_radius = 0.04;
		draw_Hilbert_plane(12 + 43, Pov->color_orange, fp); // pi_{16,24,35}

		int start_of_long_lines = 79;
		draw_Hilbert_line(start_of_long_lines + 12 + 4, Pov->color_yellow, fp); // c16
		draw_Hilbert_line(start_of_long_lines + 12 + 6, Pov->color_yellow, fp); // c24
		draw_Hilbert_line(start_of_long_lines + 12 + 10, Pov->color_yellow, fp); // c35

		draw_Hilbert_red_lines(fp);
		//draw_Hilbert_blue_lines(S, fp);
		{
		int s[] = {18, /* 19, 20, 21, 22,*/ 23};
		S->draw_edges_with_selection(s, 2, Pov->color_blue, fp); // b1, b6
		}

		//draw_surface(S, 0, fp);
		draw_Hilbert_point(31, 0.12, Pov->color_chrome, fp);
		draw_Hilbert_point(32, 0.12, Pov->color_chrome, fp);
		draw_Hilbert_point(33, 0.12, Pov->color_chrome, fp);
		draw_Hilbert_point(34, 0.12, Pov->color_chrome, fp);
		draw_Hilbert_point(36, 0.12, Pov->color_chrome, fp);
		draw_Hilbert_point(37, 0.12, Pov->color_chrome, fp);
		}
	if (round == 84) {

		//S->line_radius = 0.04;

		draw_Hilbert_red_lines(fp);

		draw_single_surface(0, fp);

		}

	if (round == 85) {

		//S->line_radius = 0.04;

		draw_Hilbert_red_lines(fp);

		draw_single_surface(0, fp);

		}

	if (round == 86) {

		draw_surface_13_1(fp);

		}
	if (round == 87) {

		//S->line_radius = 0.04;
		int *selection;
		int q = 13;
		int nb_select = q * q * q;

		if (!S->f_has_affine_space) {
			cout << "scene does not have affine space" << endl;
			exit(1);
		}
		selection = NEW_int(nb_select);
		for (i = 0; i < nb_select; i++) {
			selection[i] = S->affine_space_starting_point + i;
		}

		S->draw_points_with_selection(selection, nb_select,
				0.12, Pov->color_chrome, fp);

		FREE_int(selection);


		}
	if (round == 88) {
		{
			// red lines:
		int s[] = {12/*, 13, 14, 15, 16, 17*/};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_red, fp);
		}

		{
			// axes:
		int s[] = {42,43,44};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_black, fp);
		}


		{
			// blue lines
		//int s[] = {18, 19, 20, 21, 22, 23};
		//S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_blue, fp);
		}
		//draw_Hilbert_red_lines(S, fp);
		//draw_Hilbert_blue_lines(S, fp);
		}
	if (round == 89) {
		{
			// red lines:
		int s[] = {12/*, 13, 14, 15, 16, 17*/};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_red, fp);
		}

		{
			// axes:
		int s[] = {42,43,44};
		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_black, fp);
		}

		draw_Hilbert_cube_boxed(fp);

		{
			// blue lines
		//int s[] = {18, 19, 20, 21, 22, 23};
		//S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), color_blue, fp);
		}
		//draw_Hilbert_red_lines(S, fp);
		//draw_Hilbert_blue_lines(S, fp);
		}
	if (round == 90) {
		//S->line_radius = 0.04;
		//draw_Hilbert_red_lines(S, fp);
		//draw_Hilbert_blue_lines(S, fp);
		//{
		//int s[] = {18, /* 19, 20, 21, 22,*/ 23};
		//S->draw_edges_with_selection(s, 2, color_blue, fp); // b1, b6
		//}
		//draw_Hilbert_plane(S, 12 + 43, color_orange, fp); // pi_{16,24,35}
		//draw_Hilbert_line(S, 12 + 4, color_yellow, fp); // c16
		//draw_Hilbert_line(S, 12 + 6, color_yellow, fp); // c24
		//draw_Hilbert_line(S, 12 + 10, color_yellow, fp); // c35
		draw_single_surface(0, fp);

		{
			int selection[] = {0,1};
			int nb_select = 2;
			S->draw_lines_ai_with_selection(selection, nb_select, fp);
		}
		{
			int selection[] = {0,1};
			int nb_select = 2;
			S->draw_lines_bj_with_selection(selection, nb_select, fp);
		}
		{
			int selection[] = {0};
			int nb_select = 1;
			S->draw_lines_cij_with_selection(selection, nb_select, fp);
		}
		draw_Hilbert_plane(12 + 0, Pov->color_orange_transparent, fp); // pi_{12}
		draw_Hilbert_plane(12 + 1, Pov->color_orange_transparent, fp); // pi_{21}

		//draw_Hilbert_point(S, 31, 0.12, color_chrome, fp);
		//draw_Hilbert_point(S, 32, 0.12, color_chrome, fp);
		//draw_Hilbert_point(S, 33, 0.12, color_chrome, fp);
		//draw_Hilbert_point(S, 34, 0.12, color_chrome, fp);
		//draw_Hilbert_point(S, 36, 0.12, color_chrome, fp);
		//draw_Hilbert_point(S, 37, 0.12, color_chrome, fp);

	}

	rotation(h, nb_frames, round, fp);

	union_end(
			h, nb_frames, round,
			clipping_radius,
			fp);

	cout << "animate::draw_frame_Hilbert done" << endl;


}


void animate::draw_surface_13_1(ostream &fp)
{
	int Pts[] = {
		0, 0, 0, 1,
		1, 1, 1, 1,
		8, 0, 0, 1,
		0, 1, 0, 1,
		12, 1, 0, 1,
		0, 2, 0, 1,
		3, 2, 0, 1,
		0, 3, 0, 1,
		7, 3, 0, 1,
		0, 4, 0, 1,
		11, 4, 0, 1,
		0, 5, 0, 1,
		2, 5, 0, 1,
		0, 6, 0, 1,
		6, 6, 0, 1,
		0, 7, 0, 1,
		10, 7, 0, 1,
		0, 8, 0, 1,
		1, 8, 0, 1,
		0, 9, 0, 1,
		5, 9, 0, 1,
		0, 10, 0, 1,
		9, 10, 0, 1,
		0, 11, 0, 1,
		0, 12, 0, 1,
		4, 12, 0, 1,
		0, 0, 1, 1,
		5, 0, 1, 1,
		12, 1, 1, 1,
		2, 2, 1, 1,
		6, 2, 1, 1,
		0, 3, 1, 1,
		3, 3, 1, 1,
		4, 4, 1, 1,
		7, 4, 1, 1,
		1, 5, 1, 1,
		5, 5, 1, 1,
		6, 6, 1, 1,
		8, 6, 1, 1,
		2, 7, 1, 1,
		7, 7, 1, 1,
		8, 8, 1, 1,
		9, 8, 1, 1,
		3, 9, 1, 1,
		9, 9, 1, 1,
		10, 10, 1, 1,
		4, 11, 1, 1,
		11, 11, 1, 1,
		11, 12, 1, 1,
		12, 12, 1, 1,
		0, 0, 2, 1,
		2, 0, 2, 1,
		3, 1, 2, 1,
		11, 1, 2, 1,
		6, 2, 2, 1,
		7, 2, 2, 1,
		3, 3, 2, 1,
		9, 3, 2, 1,
		12, 4, 2, 1,
		2, 5, 2, 1,
		8, 5, 2, 1,
		4, 6, 2, 1,
		5, 6, 2, 1,
		0, 7, 2, 1,
		8, 7, 2, 1,
		9, 8, 2, 1,
		11, 8, 2, 1,
		1, 9, 2, 1,
		5, 9, 2, 1,
		1, 10, 2, 1,
		4, 10, 2, 1,
		7, 11, 2, 1,
		10, 11, 2, 1,
		6, 12, 2, 1,
		10, 12, 2, 1,
		0, 0, 3, 1,
		12, 0, 3, 1,
		5, 1, 3, 1,
		10, 1, 3, 1,
		7, 2, 3, 1,
		11, 2, 3, 1,
		4, 3, 3, 1,
		1, 4, 3, 1,
		10, 4, 3, 1,
		3, 5, 3, 1,
		11, 5, 3, 1,
		8, 6, 3, 1,
		9, 6, 3, 1,
		2, 7, 3, 1,
		5, 7, 3, 1,
		2, 8, 3, 1,
		8, 8, 3, 1,
		1, 9, 3, 1,
		12, 9, 3, 1,
		7, 10, 3, 1,
		9, 10, 3, 1,
		0, 11, 3, 1,
		6, 11, 3, 1,
		3, 12, 3, 1,
		6, 12, 3, 1,
		0, 0, 4, 1,
		9, 0, 4, 1,
		7, 1, 4, 1,
		9, 1, 4, 1,
		0, 2, 4, 1,
		10, 2, 4, 1,
		1, 5, 4, 1,
		4, 5, 4, 1,
		5, 6, 4, 1,
		7, 6, 4, 1,
		1, 7, 4, 1,
		5, 7, 4, 1,
		10, 9, 4, 1,
		4, 11, 4, 1,
		0, 0, 5, 1,
		6, 0, 5, 1,
		8, 1, 5, 1,
		9, 1, 5, 1,
		4, 3, 5, 1,
		9, 3, 5, 1,
		4, 5, 5, 1,
		5, 5, 5, 1,
		0, 6, 5, 1,
		7, 6, 5, 1,
		6, 9, 5, 1,
		8, 9, 5, 1,
		5, 10, 5, 1,
		7, 10, 5, 1,
		0, 0, 6, 1,
		3, 0, 6, 1,
		7, 1, 6, 1,
		11, 1, 6, 1,
		6, 5, 6, 1,
		7, 5, 6, 1,
		2, 9, 6, 1,
		6, 9, 6, 1,
		0, 10, 6, 1,
		10, 10, 6, 1,
		2, 11, 6, 1,
		10, 11, 6, 1,
		3, 12, 6, 1,
		11, 12, 6, 1,
		0, 0, 7, 1,
		0, 1, 7, 1,
		6, 1, 7, 1,
		2, 2, 7, 1,
		10, 2, 7, 1,
		7, 5, 7, 1,
		10, 5, 7, 1,
		2, 8, 7, 1,
		7, 8, 7, 1,
		4, 9, 7, 1,
		11, 9, 7, 1,
		4, 10, 7, 1,
		0, 0, 8, 1,
		10, 0, 8, 1,
		2, 1, 8, 1,
		5, 1, 8, 1,
		1, 4, 8, 1,
		10, 4, 8, 1,
		0, 5, 8, 1,
		8, 5, 8, 1,
		7, 7, 8, 1,
		8, 7, 8, 1,
		2, 9, 8, 1,
		7, 9, 8, 1,
		1, 10, 8, 1,
		5, 10, 8, 1,
		0, 0, 9, 1,
		7, 0, 9, 1,
		4, 1, 9, 1,
		12, 4, 9, 1,
		3, 5, 9, 1,
		9, 5, 9, 1,
		4, 6, 9, 1,
		9, 6, 9, 1,
		0, 9, 9, 1,
		3, 9, 9, 1,
		7, 12, 9, 1,
		12, 12, 9, 1,
		0, 0, 10, 1,
		4, 0, 10, 1,
		3, 1, 10, 1,
		6, 1, 10, 1,
		3, 2, 10, 1,
		11, 2, 10, 1,
		7, 3, 10, 1,
		12, 3, 10, 1,
		4, 4, 10, 1,
		7, 4, 10, 1,
		6, 5, 10, 1,
		10, 5, 10, 1,
		11, 9, 10, 1,
		12, 9, 10, 1,
		0, 0, 11, 1,
		1, 0, 11, 1,
		2, 1, 11, 1,
		8, 1, 11, 1,
		0, 4, 11, 1,
		11, 4, 11, 1,
		9, 5, 11, 1,
		11, 5, 11, 1,
		1, 8, 11, 1,
		7, 8, 11, 1,
		8, 9, 11, 1,
		9, 9, 11, 1,
		2, 11, 11, 1,
		7, 11, 11, 1,
		0, 0, 12, 1,
		11, 0, 12, 1,
		1, 1, 12, 1,
		10, 1, 12, 1,
		12, 3, 12, 1,
		12, 5, 12, 1,
		1, 7, 12, 1,
		10, 7, 12, 1,
		0, 8, 12, 1,
		11, 8, 12, 1,
		4, 9, 12, 1,
		7, 9, 12, 1,
		4, 12, 12, 1,
		7, 12, 12, 1,
	};
	int nb_affine_pts = 222;
	int q = 13;
	int *selection;
	int nb_select = nb_affine_pts;
	long int i, rk;
	int v[3];
	geometry_global Gg;

	if (!S->f_has_affine_space) {
		cout << "draw_surface_13_1 "
				"scene does not have affine space" << endl;
		exit(1);
	}

	selection = NEW_int(nb_select);
	for (i = 0; i < nb_select; i++) {
		v[0] = Pts[i * 4 + 2]; // z
		v[1] = Pts[i * 4 + 1]; // y
		v[2] = Pts[i * 4 + 0]; // x
		rk = Gg.AG_element_rank(q, v, 1, 3);
		selection[i] = S->affine_space_starting_point + rk;
	}

	S->draw_points_with_selection(selection, nb_select,
			0.12, Pov->color_chrome, fp);
	FREE_int(selection);


}

void animate::draw_frame_Hilbert_round_76(video_draw_options *Opt,
		int h, int nb_frames, int round,
		ostream &fp,
		int verbose_level)
// tritangent plane, 6 point, 2 blue lines, 6 red lines, text
{
	//int i;

	//draw_Hilbert_cube_boxed(S, fp);
	draw_Hilbert_red_lines(fp);
	//draw_Hilbert_blue_lines(S, fp);
	{
	int s[] = {18, /* 19, 20, 21, 22,*/ 23};
	S->draw_edges_with_selection(s, 2, Pov->color_blue, fp); // b1, b6
	}
	draw_Hilbert_plane(12 + 43, Pov->color_orange, fp); // pi_{16,24,35}
	draw_Hilbert_line(12 + 4, Pov->color_yellow, fp); // c16
	draw_Hilbert_line(12 + 6, Pov->color_yellow, fp); // c24
	draw_Hilbert_line(12 + 10, Pov->color_yellow, fp); // c35
	//draw_surface(S, 0, fp);
	draw_Hilbert_point(31, 0.12, Pov->color_chrome, fp);
	draw_Hilbert_point(32, 0.12, Pov->color_chrome, fp);
	draw_Hilbert_point(33, 0.12, Pov->color_chrome, fp);
	draw_Hilbert_point(34, 0.12, Pov->color_chrome, fp);
	draw_Hilbert_point(36, 0.12, Pov->color_chrome, fp);
	draw_Hilbert_point(37, 0.12, Pov->color_chrome, fp);

	//draw_Hilbert_cube_boxed(S, fp);
	//draw_Hilbert_cube_faces(S,fp);

	int idx;
	double thickness_half = 0.15;
	double extra_spacing = 0;
	string color_options("pigment { Black } ");
	//const char *color_options = "pigment { BrightGold } finish { reflection .25 specular 1 }";
	//double up_x = 1.,up_y = 1., up_z = 1.;
	//double view[3];

	//double location[3] = {-3,1,3};
	//double look_at[3];


#if 0
	double a;
	a = -1 / sqrt(3.);
	for (i = 0; i < 3; i++) {
		look_at[i] = a;
		}
#endif
	//ost << "   location  <-3,1,3>" << endl;
	//ost << "   look_at   <1,1,1>*-1/sqrt(3)" << endl;

#if 0
	for (i = 0; i < 3; i++) {
		view[i] = look_at[i] - location[i];
		}
#endif

	double scale = 0.25;
	double off_x = -0.1;
	double off_y = 0.25;
	double off_z = -0.1;

	string one("1");
	string two("2");
	string three("3");
	string four("4");
	string five("5");
	string six("6");

	idx = 36;
	draw_text(one, thickness_half, extra_spacing,
		scale,
		off_x, off_y, off_z,
		color_options,
		idx,
		//up_x, up_y, up_z,
		//view[0], view[1], view[2],
		fp, verbose_level - 1);
	idx = 31;
	draw_text(two, thickness_half, extra_spacing,
		scale,
		off_x, off_y, off_z,
		color_options,
		idx,
		//up_x, up_y, up_z,
		//view[0], view[1], view[2],
		fp, verbose_level - 1);
	idx = 32;
	draw_text(three, thickness_half, extra_spacing,
		scale,
		off_x, off_y, off_z,
		color_options,
		idx,
		//up_x, up_y, up_z,
		//view[0], view[1], view[2],
		fp, verbose_level - 1);
	idx = 33;
	draw_text(four, thickness_half, extra_spacing,
		scale,
		off_x, off_y, off_z,
		color_options,
		idx,
		//up_x, up_y, up_z,
		//view[0], view[1], view[2],
		fp, verbose_level - 1);
	idx = 34;
	draw_text(five, thickness_half, extra_spacing,
		scale,
		off_x, off_y, off_z,
		color_options,
		idx,
		//up_x, up_y, up_z,
		//view[0], view[1], view[2],
		fp, verbose_level - 1);
	idx = 37;
	draw_text(six, thickness_half, extra_spacing,
		scale,
		off_x, off_y, off_z,
		color_options,
		idx,
		//up_x, up_y, up_z,
		//view[0], view[1], view[2],
		fp, verbose_level - 1);
}


void animate::draw_frame_HCV_surface(
	int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level)
{
	int i;
	//povray_interface Pov;


	cout << "animate::draw_frame_HCV_surface" << endl;

	double scale_factor;

	scale_factor = Opt->scale_factor;

	Pov->union_start(fp);



	if (round == 0) {

		int s[1];

		s[0] = 0;
		S->draw_cubic_with_selection(s, 1, Pov->color_white, fp);
		//S->draw_line_with_selection(0, Pov->color_red, fp);

	}
	else if (round == 1) {

		int s[1];

		s[0] = 1;
		S->draw_cubic_with_selection(s, 1, Pov->color_white, fp);
#if 0
		S->draw_line_with_selection(0,
				color_red,
				fp);
#endif

	}
	else if (round == 2) {

		int s[1];

		s[0] = 2;
		S->draw_cubic_with_selection(s, 1, Pov->color_white, fp);
#if 0
		S->draw_line_with_selection(0,
				color_red,
				fp);
#endif

	}
	else if (round == 3) {

		int selection[3];
		int nb_select;

		selection[0] = 2;
		nb_select = 1;
		S->draw_cubic_with_selection(selection, 1, Pov->color_white, fp);

		selection[0] = 0;
		nb_select = 1;
		S->draw_planes_with_selection(selection, nb_select,
				Pov->color_orange, fp);

		selection[0] = 14;
		selection[1] = 19;
		selection[2] = 23;
		nb_select = 3;
		S->draw_lines_with_selection(selection, nb_select,
				Pov->color_yellow,
				fp);

	}
	else if (round == 4 || round == 5 || round == 6) {

		int selection[27];
		//int nb_select;

		selection[0] = 2;
		//nb_select = 1;
		S->draw_cubic_with_selection(selection, 1, Pov->color_white, fp);

#if 0
		selection[0] = 0;
		nb_select = 1;
		S->draw_planes_with_selection(selection, nb_select,
				color_orange, fp);
#endif

		for (i = 0; i < 27; i++) {
			selection[i] = i;
		}
		//nb_select = 27;
		S->draw_lines_with_selection(selection, 6,
				Pov->color_red,
				fp);
		S->draw_lines_with_selection(selection + 6, 6,
				Pov->color_blue,
				fp);
		S->draw_lines_with_selection(selection + 12, 15,
				Pov->color_yellow,
				fp);


	}
	else if (round == 7) {

		int selection[27];
		int nb_select;

		selection[0] = 2;
		nb_select = 1;
		S->draw_cubic_with_selection(selection, 1, Pov->color_white, fp);

		selection[0] = 0;
		nb_select = 1;
		S->draw_planes_with_selection(selection, nb_select,
				Pov->color_orange, fp);

		for (i = 0; i < 27; i++) {
			selection[i] = i;
		}
		nb_select = 27;
		S->draw_lines_with_selection(selection, 6,
				Pov->color_red,
				fp);
		S->draw_lines_with_selection(selection + 6, 6,
				Pov->color_blue,
				fp);
		S->draw_lines_with_selection(selection + 12, 15,
				Pov->color_yellow,
				fp);


		selection[0] = 0;
		selection[1] = 1;
		selection[2] = 2;
		nb_select = 3;

		S->draw_points_with_selection(selection, 3, 0.12 * 3, Pov->color_chrome, fp);

	}
	else if (round == 8) {

		// yellow lines only:

		int selection[27];
		//int nb_select;

		selection[0] = 2;
		//nb_select = 1;
		S->draw_cubic_with_selection(selection, 1, Pov->color_white, fp);

#if 0
		selection[0] = 0;
		nb_select = 1;
		S->draw_planes_with_selection(selection, nb_select,
				color_orange, fp);
#endif

		for (i = 0; i < 27; i++) {
			selection[i] = i;
		}
		//nb_select = 27 - 12;
#if 0
		S->draw_lines_with_selection(selection, 6,
				color_red,
				fp);
		S->draw_lines_with_selection(selection + 6, 6,
				color_blue,
				fp);
#endif
		S->draw_lines_with_selection(selection + 12, 15,
				Pov->color_yellow,
				fp);


	}

	rotation(h, nb_frames, round, fp);
	Pov->union_end(fp, scale_factor, clipping_radius);

}

void animate::draw_frame_E4_surface(
	int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level)
{
	cout << "animate::draw_frame_E4_surface" << endl;

	//double scale_factor;

	//scale_factor = Opt->scale_factor;

	Pov->union_start(fp);


	if (round == 0) {

		int s[1];

		s[0] = 0;
		S->draw_cubic_with_selection(s, 1, Pov->color_white, fp);
		//S->draw_line_with_selection(0, Pov->color_red, fp);

	}
	else if (round == 1) {

		int s[1];

		s[0] = 1;
		S->draw_cubic_with_selection(s, 1, Pov->color_white, fp);
		S->draw_line_with_selection(0,
				Pov->color_red,
				fp);

	}

	rotation(h, nb_frames, round, fp);
	union_end(
			h, nb_frames, round,
			clipping_radius,
			fp);

}

void animate::draw_frame_triangulation_of_cube(
	int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level)
{
	//double scale_factor;

	//scale_factor = Opt->scale_factor;


	Pov->union_start(fp);


	if (round == 0) {


		draw_Hilbert_cube_boxed(fp);

		draw_Hilbert_cube_faces(fp);


	}
	else if (round == 1) {


		draw_Hilbert_cube_boxed(fp);

		//draw_Hilbert_cube_faces(S, fp);

		{
		int s[] = {6,7,8,9};
		S->draw_faces_with_selection(s, sizeof(s) / sizeof(int), 0.01, Pov->color_pink, fp);
		}

		{
		int s[] = {12,13,14,15,16,17};

		override_double line_radius(&S->line_radius, S->line_radius * 0.5);

		S->draw_edges_with_selection(s, sizeof(s) / sizeof(int), Pov->color_black, fp);
		}

	}

	rotation(h, nb_frames, round, fp);
	union_end(
			h, nb_frames, round,
			clipping_radius,
			fp);

}

void animate::draw_frame_twisted_cubic(
	int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level)
{
	int i;

	cout << "draw_frame_twisted_cubic" << endl;

	//double my_clipping_radius;
	//double scale_factor;

	//my_clipping_radius = Opt->clipping_radius;
	//scale_factor = Opt->scale_factor;

	Pov->union_start(fp);

	for (i = 0; i < Opt->nb_clipping; i++) {
		if (Opt->clipping_round[i] == round) {
			//my_clipping_radius = Opt->clipping_value[i];
			}
		}


	if (round == 0) {


		//draw_Hilbert_cube_boxed(S, fp);

		//draw_Hilbert_cube_faces(S, fp);

		{
			int s[] = {0,1,2,3,4,5,6,7,8,9,10,11};
			override_double line_radius(&S->line_radius, S->line_radius * 0.4);

			S->draw_edges_with_selection(s, 12, Pov->color_black, fp);
		}
		{
		int *s;

		s = NEW_int(h);
		for (i = 1; i < h; i++) {
			s[i - 1] = 12 + i - 1;
		}
		override_double line_radius(&S->line_radius, S->line_radius * 0.75);

		S->draw_edges_with_selection(s, h - 1, Pov->color_red, fp);
		FREE_int(s);
		}

	}
	else if (round == 1) {


		//draw_Hilbert_cube_boxed(S, fp);

		//draw_Hilbert_cube_faces(S, fp);

		{
			int s[] = {0,1,2,3,4,5,6,7,8,9,10,11};
			override_double line_radius(&S->line_radius, S->line_radius * 0.4);

			S->draw_edges_with_selection(s, 12, Pov->color_black, fp);
		}
		{
		int *s;

		s = NEW_int(nb_frames);
		for (i = 1; i < nb_frames; i++) {
			s[i - 1] = 12 + i - 1;
		}
		override_double line_radius(&S->line_radius, S->line_radius * 0.75);

		S->draw_edges_with_selection(s, nb_frames - 1, Pov->color_red, fp);
		FREE_int(s);
		}

	}

	rotation(h, nb_frames, round, fp);
	union_end(
			h, nb_frames, round,
			clipping_radius,
			fp);


}

void animate::draw_frame_five_plus_one(
	int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level)
{
	//double d_theta, theta;

	// prepare for a curve in polar coordinates
	// with theta from 0 to 2 * pi:

	//d_theta = 2 * M_PI / nb_frames;

	//theta = (double) h * d_theta;

	int plane0 = 3;
	int line0 = 0;

	{
	int s[2];
	s[0] = plane0 + 0;
	s[1] = plane0 + 1;
	S->draw_planes_with_selection(s, 2, Pov->color_orange, fp);
	}

	{
	int s[1];
	s[0] = line0 + 0;
	S->draw_lines_with_selection(s, 1,
			Pov->color_yellow, fp);
	}
	{
	int s[1];
	s[0] = line0 + 1;
	S->draw_lines_with_selection(s, 1,
			Pov->color_red, fp);
	}
	{
	int s[1];
	s[0] = line0 + 2;
	S->draw_lines_with_selection(s, 1,
			Pov->color_blue, fp);
	}
	{
	int s[1];
	s[0] = line0 + 3;
	S->draw_lines_with_selection(s, sizeof(s) / sizeof(int),
			Pov->color_black, fp);
	}
	{
	int s[] = {0};
	S->draw_cubic_with_selection(s, sizeof(s) / sizeof(int),
			Pov->color_white, fp);
	}


	rotation(h, nb_frames, round, fp);
	union_end(
			h, nb_frames, round,
			clipping_radius,
			fp);


}


void animate::draw_frame_windy(
	int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level)
{
	numerics N;
	int i;
	double d_theta, theta, r, x, y;
	double b1[3] = {2/sqrt(6),-1/sqrt(6),-1/sqrt(6)};
	double b2[3] = {0,1/sqrt(2),-1/sqrt(2)};
	double u[4];

	// prepare for a curve in polar coordinates with theta from 0 to 2 * pi:

	d_theta = 2 * M_PI / nb_frames;

	theta = (double) h * d_theta;

	r = 1. + 2. * cos(theta);
		// lemacon curve in polar coordinates

	// turn into cartesian coordinates:
	x = r * cos(theta);
	y = r * sin(theta);

	// create the vector u as a linear combination
	// of the basis vectors b1 and b2:

	for (i = 0; i < 3; i++) {
		u[i] = x * b1[i] + y * b2[i];
		}
	u[3] = 1.;

	double A[16];
	double Av[16];
	double varphi[4];

	// let varphi be the dual coordinates of the bottom plane
	// (which is plane 2)
	varphi[0] = S->plane_coords(2, 0);
	varphi[1] = S->plane_coords(2, 1);
	varphi[2] = S->plane_coords(2, 2);
	varphi[3] = S->plane_coords(2, 3);
	//N.vec_copy(S->Plane_coords + 2 * 4, varphi, 4);

	// change from povray coordinates to homogeneous coordinates in PG(3,q):
	varphi[3] *= -1.;

	cout << "varphi=" << endl;
	N.print_system(varphi, 4, 1);
	cout << "u=" << endl;
	N.print_system(u, 4, 1);

	double lambda = 0.2;
	N.vec_scalar_multiple(u, lambda, 4);

	// create the transformation matrix A and its inverse Av:

	N.make_transform_t_varphi_u_double(4, varphi,
			u, A, Av, 0 /* verbose_level */);

	cout << "A=" << endl;
	N.print_system(A, 4, 4);
	cout << "Av=" << endl;
	N.print_system(Av, 4, 4);

	scene *S1;
	double rad = 10.;

	S1 = S->transformed_copy(A, Av, rad, verbose_level);
	cout << "Original scene:" << endl;
	S->print();
	cout << "Transformed scene:" << endl;
	S1->print();
	{
	int s[] = {0};
	S1->draw_cubic_with_selection(s, sizeof(s) / sizeof(int), Pov->color_white, fp);
	}
	{
	int s[] = {1,7,11}; // bottom plane lines: c14, c25, c36
	S1->draw_lines_cij_with_selection(s, sizeof(s) / sizeof(int), fp);
	}
	{
	int s[] = {0,3}; // a1 and a4
	S1->draw_lines_ai_with_selection(s, sizeof(s) / sizeof(int), fp);
	}
	{
	int s[] = {2}; // bottom plane
	S1->draw_planes_with_selection(s, sizeof(s) / sizeof(int), Pov->color_orange, fp);
	}

	{
	int s[] = {1}; // a2
	S1->draw_lines_ai_with_selection(s, sizeof(s) / sizeof(int), fp);
	}
	{
	int s[] = {2,4,5}; // b3, b5, b6
	S1->draw_lines_bj_with_selection(s, sizeof(s) / sizeof(int), fp);
	}

	{
	int s[6];

	S1->intersect_line_and_plane(0 /* line_idx */, 2 /* plane_idx */, s[0], verbose_level);
	S1->intersect_line_and_plane(3 /* line_idx */, 2 /* plane_idx */, s[1], verbose_level);
	S1->intersect_line_and_plane(6 + 1 /* line_idx */, 2 /* plane_idx */, s[2], verbose_level);
	S1->intersect_line_and_plane(6 + 2 /* line_idx */, 2 /* plane_idx */, s[3], verbose_level);
	S1->intersect_line_and_plane(6 + 4 /* line_idx */, 2 /* plane_idx */, s[4], verbose_level);
	S1->intersect_line_and_plane(6 + 5 /* line_idx */, 2 /* plane_idx */, s[5], verbose_level);
	S1->draw_points_with_selection(s, sizeof(s) / sizeof(int), 0.1, Pov->color_chrome, fp);
	}
	delete S1;

	rotation(h, nb_frames, round, fp);
	union_end(
			h, nb_frames, round,
			clipping_radius,
			fp);

}

void animate::rotation(
		int h, int nb_frames, int round,
		ostream &fp)
{
	if (Opt->f_rotate) {
		if (Opt->rotation_axis_type == 1) {
			Pov->rotate_111(h, nb_frames, fp);
		}
		else if (Opt->rotation_axis_type == 2) {
			Pov->rotate_around_z_axis(h, nb_frames, fp);
		}
		else if (Opt->rotation_axis_type == 3) {

			double angle_zero_one = 1. - (h * 1. / (double) nb_frames);
				// rotate in the opposite direction

			double v[3];

			v[0]= Opt->rotation_axis_custom[0];
			v[1]= Opt->rotation_axis_custom[1];
			v[2]= Opt->rotation_axis_custom[2];

			Pov->animation_rotate_around_origin_and_given_vector_by_a_given_angle(
				v, angle_zero_one, fp);
		}

	}


}


void animate::union_end(
		int h, int nb_frames, int round,
		double clipping_radius,
		ostream &fp)
{
	double scale;

	if (Opt->f_has_global_picture_scale) {

		scale = Opt->global_picture_scale;
	}
	else {
		scale = 1.0;
	}

	if (Opt->boundary_type == 1) {
		Pov->union_end(fp, scale, clipping_radius);
	}
	else if (Opt->boundary_type == 2) {
		Pov->union_end_box_clipping(fp, scale,
				clipping_radius, clipping_radius, clipping_radius);
	}
	else if (Opt->boundary_type == 3) {
		Pov->union_end_no_clipping(fp, scale);
	}
	else {
		cout << "animate::union_end boundary_type unrecognized" << endl;
	}
}


void animate::draw_text(std::string &text,
		double thickness_half, double extra_spacing,
		double scale,
		double off_x, double off_y, double off_z,
		std::string &color_options,
		int idx_point,
		//double x, double y, double z,
		//double up_x, double up_y, double up_z,
		//double view_x, double view_y, double view_z,
		ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double P1[3];
	double P2[3];
	double P3[3];
	double abc3[3];
	double angles3[3];
	double T3[3];
	double u[3];
	double view[3];
	double up[3];
	double x, y, z;


	numerics N;
	int i;

	if (f_v) {
		cout << "animate::draw_text" << endl;
		}

	x = S->point_coords(idx_point, 0);
	y = S->point_coords(idx_point, 1);
	z = S->point_coords(idx_point, 2);

	if (f_v) {
		cout << "x,y,z=" << x << ", " << y << " , " << z << endl;
		}

	for (i = 0; i < 3; i++) {
		view[i] = Pov->look_at[i] - Pov->location[i];
		}
	for (i = 0; i < 3; i++) {
		up[i] = Pov->sky[i];
		}


	if (f_v) {
		cout << "view_x,view_y,view_z=" << view[0] << ", "
				<< view[1] << " , " << view[2] << endl;
		}
	if (f_v) {
		cout << "up_x,up_y,up_z=" << up[0] << ", " << up[1]
				<< " , " << up[2] << endl;
		}
	u[0] = view[1] * up[2] - view[2] * up[1];
	u[1] = -1 *(view[0] * up[2] - up[0] * view[2]);
	u[2] = view[0] * up[1] - up[0] * view[1];
	if (f_v) {
		cout << "u=" << u[0] << ", " << u[1] << " , " << u[2] << endl;
		}
	P1[0] = x;
	P1[1] = y;
	P1[2] = z;
	P2[0] = x + u[0];
	P2[1] = y + u[1];
	P2[2] = z + u[2];
	P3[0] = x + up[0];
	P3[1] = y + up[1];
	P3[2] = z + up[2];

	N.triangular_prism(P1, P2, P3,
		abc3, angles3, T3,
		verbose_level);
	double offset[3];
	//double up[3];
	//double view[3];
#if 0
	up[0] = up_x;
	up[1] = up_y;
	up[2] = up_z;
	view[0] = view_x;
	view[1] = view_y;
	view[2] = view_z;
#endif
	N.make_unit_vector(u, 3);
	N.make_unit_vector(up, 3);
	N.make_unit_vector(view, 3);
	if (f_v) {
		cout << "up normalized: ";
		N.vec_print(up, 3);
		cout << endl;
		cout << "u normalized: ";
		N.vec_print(u, 3);
		cout << endl;
		cout << "view normalized: ";
		N.vec_print(view, 3);
		cout << endl;
		}

	offset[0] = off_x * u[0] + off_y * up[0] + off_z * view[0];
	offset[1] = off_x * u[1] + off_y * up[1] + off_z * view[1];
	offset[2] = off_x * u[2] + off_y * up[2] + off_z * view[2];

	if (f_v) {
		cout << "offset: ";
		N.vec_print(offset, 3);
		cout << endl;
		}

	ost << "\ttext {" << endl;
		ost << "\t\tttf \"timrom.ttf\", \"" << text << "\", "
				<< thickness_half << ", " << extra_spacing << " ";
		ost << color_options << endl;
		ost << "\t\tscale " << scale << endl;
		ost << "\t\trotate<0,180,0>" << endl;
		ost << "\t\trotate<90,0,0>" << endl;
		ost << "\t\trotate<";
		N.output_double(N.rad2deg(angles3[0]), ost);
		ost << ",0,0>" << endl;
		ost << "\t\trotate<0, ";
		N.output_double(N.rad2deg(angles3[1]), ost);
		ost << ",0>" << endl;
		ost << "\t\trotate<0,0, ";
		N.output_double(N.rad2deg(angles3[2]), ost);
		ost << ">" << endl;
		ost << "\t\ttranslate<";
		N.output_double(T3[0] + offset[0], ost);
		ost << ", ";
		N.output_double(T3[1] + offset[1], ost);
		ost << ", ";
		N.output_double(T3[2] + offset[2], ost);
		ost << ">" << endl;
	ost << "\t}" << endl;
		//pigment { BrightGold }
		//finish { reflection .25 specular 1 }
		//translate <0,0,0>
	if (f_v) {
		cout << "animate::draw_text done" << endl;
		}
}

void animate::draw_text_with_selection(int *selection, int nb_select,
	double thickness_half, double extra_spacing,
	double scale,
	double off_x, double off_y, double off_z,
	std::string &options, std::string &group_options,
	ostream &ost, int verbose_level)
{
	int i, s;
	numerics N;

	ost << endl;
	//ost << "	union{ // labels" << endl;
	//ost << endl;
	//ost << "	        #declare r=" << line_radius << "; " << endl;
	ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];

		int idx_point;
		string text;

		idx_point = S->Labels[s].first;
		text = S->Labels[s].second;


		draw_text(text,
				thickness_half, extra_spacing,
				scale,
				off_x, off_y, off_z,
				options,
				idx_point,
				ost, verbose_level);
		}
	ost << endl;
	//ost << "		" << group_options << "" << endl;
	//ost << "	}" << endl;
}



}}



