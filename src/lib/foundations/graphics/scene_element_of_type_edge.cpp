/*
 * scene_element_of_type_edge.cpp
 *
 *  Created on: Apr 4, 2022
 *      Author: betten
 */




#include "foundations.h"

using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace graphics {

scene_element_of_type_edge::scene_element_of_type_edge()
{
}

scene_element_of_type_edge::~scene_element_of_type_edge()
{
}

void scene_element_of_type_edge::init(std::string &pt1, std::string &pt2)
{
	Idx.push_back(pt1);
	Idx.push_back(pt2);
}

void scene_element_of_type_edge::print()
{
	cout << "edge from " << Idx[0] << " to " << Idx[1] << endl;
}


}}}


