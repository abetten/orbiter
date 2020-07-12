/*
 * logger_debugger.h
 *
 *  Created on: Jun 5, 2020
 *      Author: sajeeb
 */

#ifndef LOGGER_DEBUGGER_H_
#define LOGGER_DEBUGGER_H_

#ifdef __LOGGER_DEBUGGER__
#define __LOGGER_DEBUGGER__(fmt, ...) {\
	std::stringstream ss;\
	ss << std::this_thread::get_id();\
	const unsigned long long int id = std::stoull(ss.str());\
	const auto duration = std::chrono::system_clock::now().time_since_epoch();\
	const unsigned long int millis =\
			std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();\
	const char*  ___file___ = __FILE__;\
	const int ___line___ = __LINE__;\
	printf("[%llu][%lu]: %s:%d:: " fmt "\n", id, millis, ___file___, ___line___, ##__VA_ARGS__);\
	fflush((FILE*)stdout);\
}
#else
#define __LOGGER_DEBUGGER__(fmt, ...)
#endif


#endif /* LOGGER_DEBUGGER_H_ */
