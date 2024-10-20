//计时器使用了 chrono::high_resolution_clock，
//具有微秒级的精度，非常适合需要精细测量时间的场景，如性能调优和代码优化。
//该 Clock 类用于测量代码的执行时间，并提供了丰富的操作，如暂停、恢复和多次计时。
//对计时器的名称和计时过程的输出进行格式化，计时信息更加直观
#pragma once

#include <string>
#include <chrono>
#include <cstdio>
#include <cstdarg>

using namespace std;
using namespace std::chrono;

class Clock {
public:
    explicit Clock(const string &str) {
        s = str;
    }

    const char *start() {
		passed_time = 0.0;
		paused = false;
        start_time = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
        sprintf(str, "[\033[1m\033[44;30m%-12s\033[0m] Start...", s.c_str());
        return str;
    }

    const char *count(const char *fmt = "", ...) {
        va_list args;
        char str2[1000];
        va_start(args, fmt);
        vsprintf(str2, fmt, args);
        va_end(args);
        uint64_t end_time = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
        double t = double(end_time - start_time) / 1e6;
		if (paused) {
			t = passed_time;
		} else {
			t += passed_time;
		}
        sprintf(str, "[\033[1m\033[44;31m%-12s\033[0m] %.6lfs   %s", s.c_str(), t, str2);
        return str;
    }

	const char *pause() {
		if (!paused) {
			uint64_t end_time = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
        	double t = double(end_time - start_time) / 1e6;
			passed_time += t;
			paused = true;
		}
		return str;
	}
	const char *goon() {
		if (paused) {
			start_time = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
			paused = false;
		}
		return str;
	}
private:
    char str[1000]{};
    string s{};
	double passed_time;
    uint64_t start_time{};
	bool paused;
};
