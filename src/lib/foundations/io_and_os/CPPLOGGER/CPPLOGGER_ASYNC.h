/*
 * CPPLOGGER_ASYNC.h
 *
 *  Created on: Jun 12, 2020
 *      Author: sajeeb
 */
#ifndef CPPLOGGER_ASYNC
#define CPPLOGGER_ASYNC

/**
 * if the logger initializer is already defined, undefine it since
 * asynchronous logging takes precedence
 */
#ifdef logger_init
#undef logger_init
#endif

/**
 * if the logger output stream is already defined, undefine it since
 * asynchronous logging takes precedence
 */
#ifdef logger_output_stream
#undef logger_output_stream
#endif

/**
 * macro to initialize the extern variables in logger
 */
#ifndef logger_init
#define logger_init() \
namespace logger { \
	bool _print_timestamps_ = false; \
	bool _print_thread_id_ = false; \
	bool _print_log_type_ = true; \
	bool _print_file_ = true; \
	bool _print_line_ = true; \
	bool _enable_global_ = true; \
	bool _flush_immediately_ = false; \
	mutex critical_section; \
	va_list __args__; \
	FILE* _output_stream_ = stdout; \
	lock_free_queue queue;\
	timestamp_resolution _resolution = logger::timestamp_resolution::millisecond;\
	log_level _loglevel_global_ = logger::log_level::all;\
}
#endif

#ifndef logger_output_stream
#define logger_output_stream(v) \
		logger::_output_stream_ = v;\
		logger::queue.set_output_stream((FILE*)v)
#endif

#include "CPPLOGGER_SYNC.h"

namespace logger {

	#ifndef logger_async
	#define logger_async_info(...) logger::_info_async_(__FILE__, __LINE__, __VA_ARGS__)
	#define logger_async_error(...) logger::_error_async_(__FILE__, __LINE__, __VA_ARGS__)
	#define logger_async_warning(...) logger::_warning_async_(__FILE__, __LINE__, __VA_ARGS__)
	#define logger_async_debug(...) logger::_debug_async_(__FILE__, __LINE__, __VA_ARGS__)
	#endif

	/**
	 *
	 */
	extern lock_free_queue queue;

	/**
	 * wrapper function containing the boilerplate code for multi-threaded
	 * logging
	 */
	#ifndef __CPPLOGGER_ASYNC__
	#define __CPPLOGGER_ASYNC__(fmt, color, type) {\
		va_list __args__; \
		va_start(__args__, fmt); \
		lock_free_queue::node* _node = queue.get();\
		char* b = _node->data;\
		if (_print_log_type_) { \
			b += snprintf(b, 32, color "[" type "]" ANSI_RESET); \
		} \
		if (_print_timestamps_) { \
			b += __print_timestamp__(b);\
		} \
		if (_print_thread_id_) { \
			std::stringstream ss; ss << std::this_thread::get_id(); \
			const unsigned long long int id = std::stoull(ss.str()); \
			b += snprintf(b, 32, "[%04llu]", id); \
		} \
		if (_print_log_type_ || _print_timestamps_ || _print_thread_id_) { \
			b += snprintf(b, 4, ": "); \
		} \
		if (_print_file_) { \
			b += snprintf(b, 32, "%s:", _file_); \
		} \
		if (_print_line_) { \
			b += snprintf(b, 32, "%d:", line); \
		} \
		if (_print_file_ || _print_line_) { \
			b += snprintf(b, 4, ": "); \
		} \
		b += vsprintf(b, fmt, __args__); \
		b += snprintf(b, 32, ANSI_RESET "\n"); \
		va_end(__args__);\
		_node->done = true;\
	}
	#endif

	/**
	 * wrapper macro for the asynchronous functions provided by the logger
	 */
	#ifndef __CPPLOGGER_ASYNC_FUNC__
	#define __CPPLOGGER_ASYNC_FUNC__(func_name, log_color, log_level, log_type)\
	inline void func_name(const char* _file_, const int line, const int cvl, const int avl, const char* fmt, ...) {\
		if (_enable_global_ && _enable_translation_uint_ && cvl >= avl) {\
			if (_loglevel_global_ == all || _loglevel_global_ == log_level) {\
				__CPPLOGGER_ASYNC__(fmt, log_color, log_type);\
			}\
		}\
	}\
	\
	inline void func_name(const char* _file_, const int line, const char* fmt, ...) {\
		if (_enable_global_ && _enable_translation_uint_) {\
			if (_loglevel_global_ == all || _loglevel_global_ == log_level) {\
				__CPPLOGGER_ASYNC__(fmt, log_color, log_type);\
			}\
		}\
	}
	#endif

	/**
	 * Auto generate the asynchronous info logging function
	 */
	__CPPLOGGER_ASYNC_FUNC__(_info_async_, ANSI_GREEN, info, "INFO")

	/**
	 * Auto generate the asynchronous error logging function
	 */
	__CPPLOGGER_ASYNC_FUNC__(_error_async_, ANSI_RED, error, "ERROR")

	/**
	 * Auto generate the asynchronous warning logging function
	 */
	__CPPLOGGER_ASYNC_FUNC__(_warning_async_, ANSI_PURPLE, warning, "WARNING")

	/**
	 * Auto generate the asynchronous debug logging function
	 */
	__CPPLOGGER_ASYNC_FUNC__(_debug_async_, ANSI_CYAN, debug, "DEBUG")

}
#endif
