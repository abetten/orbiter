// snakes_and_ladders_global.C
//
// Anton Betten
//
// October 12, 2013

#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {





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

const char *trace_result_as_text(trace_result r)
{
	if (r == found_automorphism) {
		return "found_automorphism";
		}
	else if (r == not_canonical) {
		return "not_canonical";
		}
	else if (r == no_result_extension_not_found) {
		return "no_result_extension_not_found";
		}
	else if (r == no_result_fusion_node_installed) {
		return "no_result_fusion_node_installed";
		}
	else if (r == no_result_fusion_node_already_installed) {
		return "no_result_fusion_node_already_installed";
		}
	else {
		return "unkown trace result";
		}
}

int trace_result_is_no_result(trace_result r)
{
	if (r == no_result_extension_not_found || 
		r == no_result_fusion_node_installed || 
		r == no_result_fusion_node_already_installed) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}





}}





