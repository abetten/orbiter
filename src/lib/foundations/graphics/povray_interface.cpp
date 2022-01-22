/*
 * povray_interface.cpp
 *
 *  Created on: Jan 6, 2019
 *      Author: betten
 */


#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {

povray_interface::povray_interface()
{
	//color_white_simple = "pigment{White*0.5 }";
	color_white_simple.assign("texture{ pigment{ White*0.5 transmit 0.2 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }");
	color_white.assign("texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }");
	color_white_very_transparent.assign("texture{ pigment{ White*0.5 transmit 0.75 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }");
	color_black.assign("texture{ pigment{ color Black } finish { diffuse 0.9 phong 1}}");
	color_pink.assign("texture{ pigment{ color Pink } finish { diffuse 0.9 phong 1}}");
	color_pink_transparent.assign("texture{ pigment{ color Pink transmit 0.5 } finish {diffuse 0.9 phong 0.6} }");
	color_green.assign("texture{ pigment{ color Green } finish { diffuse 0.9 phong 1}}");
	color_gold.assign("texture{ pigment{ color Gold } finish { diffuse 0.9 phong 1}}");
	color_red.assign("texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}");
	color_blue.assign("texture{ pigment{ color Blue } finish { diffuse 0.9 phong 1}}");
	color_yellow.assign("texture{ pigment{ color Yellow } finish { diffuse 0.9 phong 1}}");
	color_yellow_transparent.assign("texture{ pigment{ color Yellow transmit 0.7 } finish {diffuse 0.9 phong 0.6} }");
	color_scarlet.assign("texture{ pigment{ color Scarlet } finish { diffuse 0.9 phong 1}}");
	color_brown.assign("texture{ pigment{ color Brown } finish { diffuse 0.9 phong 1}}");
	color_orange.assign("texture{ pigment{ color Orange transmit 0.5 } finish { diffuse 0.9 phong 1}}");
	color_orange_transparent.assign("texture{ pigment{ color Orange transmit 0.7 } finish { diffuse 0.9 phong 1}}");
	color_orange_no_phong.assign("texture{ pigment{ color Orange transmit 0.6 } finish { diffuse 0.6 brilliance 5 phong 0.3 metallic }}");
	color_chrome.assign("texture{ Polished_Chrome pigment{quick_color White} }");
	color_gold_dode.assign("texture{  pigment{color Gold transmit 0.7 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }");
	color_gold_transparent.assign("texture{  pigment{color Gold transmit 0.7 } finish { diffuse 0.6 brilliance 5 phong 0.3 metallic } }");
	color_red_wine_transparent.assign("texture{ pigment{ color rgb<1, 0,0.25> transmit 0.5 } finish {diffuse 0.9 phong 0.6} }");
	color_yellow_lemon_transparent.assign("texture{ pigment{ color rgb< 0.35, 0.6, 0.0> transmit 0.5 } finish {diffuse 0.9 phong 0.6} }");

	//double sky[3];
	//double location[3];
	//double look_at[3];

}

povray_interface::~povray_interface()
{

}


void povray_interface::beginning(ostream &ost,
		double angle,
		double *sky,
		double *location,
		double *look_at,
		int f_with_background)
// angle = 22
// sky <1,1,1>
// location  <-3,1,3>
// look_at = <0,0,0>
// or <1,1,1>*-1/sqrt(3)
{
	ost << "//Files with predefined colors and textures" << endl;
	ost << "#version 3.7;" << endl;
	ost << "#include \"colors.inc\"" << endl;
	ost << "#include \"glass.inc\"" << endl;
	ost << "#include \"golds.inc\"" << endl;
	ost << "#include \"metals.inc\"" << endl;
	ost << "#include \"stones.inc\"" << endl;
	ost << "#include \"woods.inc\"" << endl;
	ost << "#include \"textures.inc\"" << endl;
	ost << endl;

	int i;

	for (i = 0; i < 3; i++) {
		povray_interface::sky[i] = sky[i];
		povray_interface::location[i] = location[i];
		povray_interface::look_at[i] = look_at[i];
	}

#if 0
	ost << "//Place the camera" << endl;
	ost << "camera {" << endl;
	ost << "   sky <0,0,1> " << endl;
	ost << "   direction <-1,0,0>" << endl;
	ost << "   right <-4/3,0,0> " << endl;
	ost << "	//location <-2.5,0.6,-3>*3" << endl;
	ost << "	//look_at<0,0.2,0>" << endl;
	ost << "   location  <0,5,0>  //Camera location" << endl;
	ost << "   look_at   <0,0,0>    //Where camera is pointing" << endl;
	ost << "   angle " << angle << "      //Angle of the view" << endl;
	ost << "	// 22 is default, 18 is closer,  28 is further away" << endl;
	ost << "}" << endl;
	ost << endl;
	ost << "//Ambient light to brighten up darker pictures" << endl;
	ost << "//global_settings { ambient_light White }" << endl;
	ost << "global_settings { max_trace_level 10 }" << endl;
	ost << endl;
	ost << endl;
	ost << "//Place a light" << endl;
	ost << "//light_source { <15,30,1> color White*2 }   " << endl;
	ost << "//light_source { <10,10,0> color White*2 }  " << endl;
	ost << "light_source { <0,2,0> color White*2 }    " << endl;
	ost << "light_source { <0,0,2> color White }" << endl;
	ost << "//light_source { <0,10,0> color White*2}" << endl;
	ost << endl;
	ost << endl;
	ost << endl;
	ost << "//plane{z,7 pigment {SkyBlue} }" << endl;
	ost << "plane{y,7 pigment {SkyBlue} }" << endl;
	ost << endl;
	ost << "//texture {T_Silver_3A}" << endl;
	ost << endl;
	ost << "//Set a background color" << endl;
	ost << "background { color SkyBlue }" << endl;
	ost << endl;
	ost << "union{ " << endl;
	ost << "/* 	        #declare r=0.09 ; " << endl;
	ost << endl;
	ost << "object{ // x-axis" << endl;
	ost << "cylinder{< 0,0,0 >,<1.5,0,0 > ,r }" << endl;
	ost << " 	pigment{Red} " << endl;
	ost << endl;
	ost << "} " << endl;
	ost << "object{ // y-axis" << endl;
	ost << "cylinder{< 0,0,0 >,<0,1.5,0 > ,r }" << endl;
	ost << " 	pigment{Green} " << endl;
	ost << endl;
	ost << "} " << endl;
	ost << "object{ // z-axis" << endl;
	ost << "cylinder{< 0,0,0 >,<0,0,1.5 > ,r }" << endl;
	ost << " 	pigment{Blue} " << endl;
 	ost << endl;
	ost << "} */" << endl;
#else
	ost << "//Place the camera" << endl;
	ost << "camera {" << endl;
	ost << "   sky  <" << sky[0] << "," << sky[1] << "," << sky[2] << ">" << endl;
	//ost << "   sky " << sky << endl;
	ost << "   //direction <1,0,0>" << endl;
	ost << "   //right <1,1,0> " << endl;
	ost << "   location  <" << location[0] << "," << location[1] << "," << location[2] << ">" << endl;
	ost << "   look_at  <" << look_at[0] << "," << look_at[1] << "," << look_at[2] << ">" << endl;
	//ost << "   look_at  " << look_at << endl;
	ost << "   angle " << angle << "      //Angle of the view" << endl;
	ost << "	// smaller numbers are closer. Must be less than 180" << endl;
	ost << "}" << endl;
	ost << endl;
	ost << "//Ambient light to brighten up darker pictures" << endl;
	ost << "//global_settings { ambient_light White }" << endl;
	ost << "global_settings { max_trace_level 10 }" << endl;
	ost << "global_settings { assumed_gamma 1.0 }" << endl;
	ost << endl;
	ost << "//Place a light" << endl;
	//ost << "light_source { <4,4,4> color White }  " << endl;
	//ost << "light_source { <-5,0,5> color White }" << endl;
	ost << "light_source { <" << sky[0] * 6 << "," << sky[1] * 6 << "," << sky[2] * 6 << "> color White }  " << endl; // parallel
	ost << "light_source { <" << location[0] << "," << location[1] << "," << location[2] << "> color White * 2 }" << endl;
	ost << endl;

	if (f_with_background) {
		ost << "//Set a background color" << endl;
		ost << "background { color SkyBlue }" << endl;
		ost << endl;
	}
	else {
		ost << "//Set a background color" << endl;
		ost << "background { color White }" << endl;
		ost << endl;
	}
	ost << "// main part:" << endl;
#endif
	ost << endl;
	ost << endl;
}


void povray_interface::animation_rotate_around_origin_and_1_1_1(ostream &ost)
{
	ost << "	// the next three steps will perform a rotation" << endl;
	ost << "	// around the axis of symmetry 1,1,1:" << endl;
	ost << endl;
	ost << "	// move 1,1,1 to sqrt(3),0,0:" << endl;
	ost << "	matrix<" << endl;
	ost << "	1/sqrt(3),2/sqrt(6),0," << endl;
	ost << "	1/sqrt(3),-1/sqrt(6),1/sqrt(2)," << endl;
	ost << "	1/sqrt(3),-1/sqrt(6),-1/sqrt(2)," << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
	ost << "        rotate <360*clock,0,0> " << endl;
	ost << endl;
	ost << "	// move sqrt(3),0,0 back to 1,1,1:" << endl;
	ost << endl;
	ost << "	matrix<" << endl;
	ost << "	1/sqrt(3),1/sqrt(3),1/sqrt(3)," << endl;
	ost << "	2/sqrt(6),-1/sqrt(6),-1/sqrt(6)," << endl;
	ost << "	0,1/sqrt(2),-1/sqrt(2)," << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
}

void povray_interface::animation_rotate_around_origin_and_given_vector(
	double *v, ostream &ost)
{
	double A[9], Av[9];
	numerics N;

	N.orthogonal_transformation_from_point_to_basis_vector(v,
		A, Av, 0 /* verbose_level */);

	ost << "	// the next three steps will perform a rotation" << endl;
	ost << "	// around the axis of symmetry (" << v[0] << ", " << v[1] << "," << v[2] << "):" << endl;
	ost << endl;
	ost << "	// move axis of symmetry to 1,0,0:" << endl;
	ost << "	matrix<" << endl;
	ost << "	";
	N.output_double(A[0], ost);
	ost << ",";
	N.output_double(A[1], ost);
	ost << ",";
	N.output_double(A[2], ost);
	ost << ",";
	ost << "	";
	N.output_double(A[3], ost);
	ost << ",";
	N.output_double(A[4], ost);
	ost << ",";
	N.output_double(A[5], ost);
	ost << ",";
	ost << "	";
	N.output_double(A[6], ost);
	ost << ",";
	N.output_double(A[7], ost);
	ost << ",";
	N.output_double(A[8], ost);
	ost << ",";
	ost << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
	ost << "        rotate <360*clock,0,0> " << endl;
	ost << endl;
	ost << "	// move 1,0,0 back to axis of symmetry:" << endl;
	ost << endl;
	ost << "	matrix<" << endl;
	ost << "	";
	N.output_double(Av[0], ost);
	ost << ",";
	N.output_double(Av[1], ost);
	ost << ",";
	N.output_double(Av[2], ost);
	ost << ",";
	ost << "	";
	N.output_double(Av[3], ost);
	ost << ",";
	N.output_double(Av[4], ost);
	ost << ",";
	N.output_double(Av[5], ost);
	ost << ",";
	ost << "	";
	N.output_double(Av[6], ost);
	ost << ",";
	N.output_double(Av[7], ost);
	ost << ",";
	N.output_double(Av[8], ost);
	ost << ",";
	ost << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
}

void povray_interface::animation_rotate_xyz(
	double angle_x_deg, double angle_y_deg, double angle_z_deg, ostream &ost)
{
	numerics N;

	ost << "	rotate <";
	N.output_double(angle_x_deg, ost);
	ost << ",";
	N.output_double(angle_y_deg, ost);
	ost << ",";
	N.output_double(angle_z_deg, ost);
	ost << ">" << endl;
}

void povray_interface::animation_rotate_around_origin_and_given_vector_by_a_given_angle(
	double *v, double angle_zero_one, ostream &ost)
{
	double A[9], Av[9];
	numerics N;

	N.orthogonal_transformation_from_point_to_basis_vector(v,
		A, Av, 0 /* verbose_level */);

	ost << "	// the next three steps will perform a rotation" << endl;
	ost << "	// around the axis of symmetry (" << v[0] << ", " << v[1] << "," << v[2] << "):" << endl;
	ost << endl;
	ost << "	// move axis of symmetry to (" << v[0] << ", " << v[1] << ", " << v[2] << ") " << endl;
	ost << "	matrix<" << endl;
	ost << "	";
	N.output_double(A[0], ost);
	ost << ",";
	N.output_double(A[1], ost);
	ost << ",";
	N.output_double(A[2], ost);
	ost << ",";
	ost << "	";
	N.output_double(A[3], ost);
	ost << ",";
	N.output_double(A[4], ost);
	ost << ",";
	N.output_double(A[5], ost);
	ost << ",";
	ost << "	";
	N.output_double(A[6], ost);
	ost << ",";
	N.output_double(A[7], ost);
	ost << ",";
	N.output_double(A[8], ost);
	ost << ",";
	ost << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
	ost << "        rotate <" << angle_zero_one * 360. << ",0,0> " << endl;
	ost << endl;
	ost << "	// move (" << v[0] << ", " << v[1] << ", " << v[2] << ") back to axis:" << endl;
	ost << endl;
	ost << "	matrix<" << endl;
	ost << "	";
	N.output_double(Av[0], ost);
	ost << ",";
	N.output_double(Av[1], ost);
	ost << ",";
	N.output_double(Av[2], ost);
	ost << ",";
	ost << "	";
	N.output_double(Av[3], ost);
	ost << ",";
	N.output_double(Av[4], ost);
	ost << ",";
	N.output_double(Av[5], ost);
	ost << ",";
	ost << "	";
	N.output_double(Av[6], ost);
	ost << ",";
	N.output_double(Av[7], ost);
	ost << ",";
	N.output_double(Av[8], ost);
	ost << ",";
	ost << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
}


void povray_interface::union_start(ostream &ost)
{
	ost << "union{ " << endl;
	ost << endl;
	ost << endl;
	ost << "// uncomment this if you need axes:" << endl;
	ost << "/* 	        #declare r=0.09 ; " << endl;
	ost << endl;
	ost << "object{ // x-axis" << endl;
	ost << "cylinder{< 0,0,0 >,<1.5,0,0 > ,r }" << endl;
	ost << " 	pigment{Red} " << endl;
	ost << endl;
	ost << "} " << endl;
	ost << "object{ // y-axis" << endl;
	ost << "cylinder{< 0,0,0 >,<0,1.5,0 > ,r }" << endl;
	ost << " 	pigment{Green} " << endl;
	ost << endl;
	ost << "} " << endl;
	ost << "object{ // z-axis" << endl;
	ost << "cylinder{< 0,0,0 >,<0,0,1.5 > ,r }" << endl;
	ost << " 	pigment{Blue} " << endl;
 	ost << endl;
	ost << "} */" << endl;
}

void povray_interface::union_end(ostream &ost, double scale_factor, double clipping_radius)
{
	ost << endl;
	ost << " 	scale  " << scale_factor << endl;
	//ost << " 	scale  1.0" << endl;
	ost << endl;
	ost << "	clipped_by { sphere{ < 0.,0.,0. > , " << clipping_radius << "  } }" << endl;
	ost << "	bounded_by { clipped_by }" << endl;
	ost << endl;
	ost << "} // union" << endl;
}

void povray_interface::union_end_box_clipping(ostream &ost, double scale_factor,
		double box_x, double box_y, double box_z)
{
	ost << endl;
	ost << " 	scale  " << scale_factor << endl;
	//ost << " 	scale  1.0" << endl;
	ost << endl;
	ost << "	clipped_by { box{ < " << -1 * box_x << ", " << -1 * box_y << ", " << -1 * box_z << " > * " << scale_factor << ", "
			" < " << box_x << ", " << box_y << ", " << box_z << " > * " << scale_factor << " } }" << endl;
	ost << "	bounded_by { clipped_by }" << endl;
	ost << endl;
	ost << "} // union" << endl;
}

void povray_interface::union_end_no_clipping(ostream &ost, double scale_factor)
{
	ost << endl;
	ost << " 	scale  " << scale_factor << endl;
	//ost << " 	scale  1.0" << endl;
	ost << endl;
	ost << endl;
	ost << "} // union" << endl;
}

void povray_interface::bottom_plane(ostream &ost)
{

	ost << endl;
	ost << "//bottom plane:" << endl;
	ost << "plane {" << endl;
	ost << "    <1,1,1>*1/sqrt(3), -2" << endl;
	ost << "    texture {" << endl;
	ost << "      pigment {SteelBlue}" << endl;
	ost << "      finish {" << endl;
	ost << "        diffuse 0.6" << endl;
	ost << "        ambient 0.2" << endl;
	ost << "        phong 1" << endl;
	ost << "        phong_size 100" << endl;
	ost << "        reflection 0.25" << endl;
	ost << "      }" << endl;
	ost << "    }" << endl;
	ost << "  } // end plane" << endl;
#if 0
	ost << endl;
	ost << endl;
	ost << endl;
	ost << "#declare d = .8; " << endl;
	ost << endl;
	ost << "plane {" << endl;
	ost << "    //y, -d" << endl;
	ost << "    z, -d" << endl;
	ost << "    texture {" << endl;
	ost << "      pigment {SkyBlue}   // Yellow" << endl;
	ost << "      //pigment {" << endl;
	ost << "      //  checker" << endl;
	ost << "      //  color rgb<0.5, 0, 0>" << endl;
	ost << "      //  color rgb<0, 0.5, 0.5>" << endl;
	ost << "      //}" << endl;
	ost << "      finish {" << endl;
	ost << "        diffuse 0.6" << endl;
	ost << "        ambient 0.2" << endl;
	ost << "        phong 1" << endl;
	ost << "        phong_size 100" << endl;
	ost << "        reflection 0.25" << endl;
	ost << "      }" << endl;
	ost << "    }" << endl;
	ost << "  }" << endl;
#endif
	ost << endl;
	ost << endl;

}

void povray_interface::rotate_111(int h, int nb_frames, ostream &fp)
{
	//int nb_frames_per_rotation;
	//nb_frames_per_rotation = nb_frames;
	double angle_zero_one = 1. - (h * 1. / (double) nb_frames);
		// rotate in the opposite direction

	double v[3] = {1.,1.,1.};

	animation_rotate_around_origin_and_given_vector_by_a_given_angle(
		v, angle_zero_one, fp);
}

void povray_interface::rotate_around_z_axis(int h, int nb_frames, ostream &fp)
{
	//int nb_frames_per_rotation;
	//nb_frames_per_rotation = nb_frames;
	double angle_zero_one = 1. - (h * 1. / (double) nb_frames);
		// rotate in the opposite direction

	double v[3] = {0,0,1.};

	animation_rotate_around_origin_and_given_vector_by_a_given_angle(
		v, angle_zero_one, fp);
}


void povray_interface::ini(ostream &ost, const char *fname_pov,
	int first_frame, int last_frame)
{
	ost << "; Persistence Of Vision raytracer version 3.7 example file." << endl;
	ost << "Antialias=On" << endl;
	ost << endl;
	ost << "Antialias_Threshold=0.1" << endl;
	ost << "Antialias_Depth=2" << endl;
	ost << "Input_File_Name=" << fname_pov << endl;
	ost << endl;
	ost << "Initial_Frame=" << first_frame << endl;
	ost << "Final_Frame=" << last_frame << endl;
	ost << "Initial_Clock=0" << endl;
	ost << "Final_Clock=1" << endl;
	ost << endl;
	ost << "Cyclic_Animation=on" << endl;
	ost << "Pause_when_Done=off" << endl;
}

}}


