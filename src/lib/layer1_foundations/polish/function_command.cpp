/*
 * function_command.cpp
 *
 *  Created on: Apr 18, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace polish {


function_command::function_command()
{
	type = 0;
	f_has_argument = FALSE;
	arg = 0;
	val = 0;
}

function_command::~function_command()
{

}

void function_command::init_with_argument(int type, int arg)
{
	function_command::type = type;
	function_command::f_has_argument = TRUE;
	function_command::arg = arg;
}

void function_command::init_push_immediate_constant(double val)
{
	function_command::type = 2;
	function_command::val = val;

}

void function_command::init_simple(int type)
{
	function_command::type = type;
	function_command::f_has_argument = TRUE;
}


}}}



