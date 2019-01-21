// extension.C
//
// Anton Betten
// Dec 19, 2011

#include "foundations/foundations.h"
#include "groups_and_group_actions/groups_and_group_actions.h"
#include "poset_classification/poset_classification.h"

namespace orbiter {

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

}




