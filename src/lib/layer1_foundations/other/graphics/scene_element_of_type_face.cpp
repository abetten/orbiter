/*
 * scene_element_of_type_face.cpp
 *
 *  Created on: Apr 4, 2022
 *      Author: betten
 */






#include "foundations.h"

using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace graphics {

scene_element_of_type_face::scene_element_of_type_face()
{
	Record_birth();
}

scene_element_of_type_face::~scene_element_of_type_face()
{
	Record_death();
}

void scene_element_of_type_face::init(
		std::vector<std::string> &pts)
{
	int i;

	for (i = 0; i < pts.size(); i++) {
		Pts.push_back(pts[i]);
	}
}

void scene_element_of_type_face::print()
{
	int i;

	cout << "face containing ";
	for (i = 0; i < Pts.size(); i++) {
		cout << Pts[i];
		if (i < Pts.size() - 1) {
			cout << ", ";
		}
	}
	cout << endl;

}


}}}}



