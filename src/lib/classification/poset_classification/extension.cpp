// extension.cpp
//
// Anton Betten
// Dec 19, 2011

#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {

extension::extension()
{
	pt = -1;
	orbit_len = 0;
	type = EXTENSION_TYPE_UNPROCESSED;
	data = 0;
	data1 = 0;
	data2 = 0;
}

extension::~extension()
{
}

int extension::get_pt()
{
	return pt;
}

void extension::set_pt(int pt)
{
	extension::pt = pt;
}

int extension::get_type()
{
	return type;
}

void extension::set_type(int type)
{
	extension::type = type;
}

int extension::get_orbit_len()
{
	return orbit_len;
}

void extension::set_orbit_len(int orbit_len)
{
	extension::orbit_len = orbit_len;
}

int extension::get_data()
{
	return data;
}

void extension::set_data(int data)
{
	extension::data = data;
}


int extension::get_data1()
{
	return data1;
}

void extension::set_data1(int data1)
{
	extension::data1 = data1;
}

int extension::get_data2()
{
	return data2;
}

void extension::set_data2(int data2)
{
	extension::data2 = data2;
}


void print_extension_type(ostream &ost, int t)
{
	if (t == EXTENSION_TYPE_UNPROCESSED) {
		ost << "   unprocessed";
		}
	else if (t == EXTENSION_TYPE_EXTENSION) {
		ost << "     extension";
		}
	else if (t == EXTENSION_TYPE_FUSION) {
		ost << "        fusion";
		}
	else if (t == EXTENSION_TYPE_PROCESSING) {
		ost << "    processing";
		}
	else if (t == EXTENSION_TYPE_NOT_CANONICAL) {
		ost << " not canonical";
		}
	else {
		ost << "type=" << t;
		}
}


}}

