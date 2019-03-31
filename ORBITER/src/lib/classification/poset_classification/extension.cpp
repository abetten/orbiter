// extension.C
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

