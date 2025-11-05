#pragma once
#include <chrono>
#include <iostream>
#include <string>

using namespace std::string_literals;

class Timer {
public:
    Timer() = default;

    ~Timer() = default;

    void startWatch() {
        _startWatchTime = std::chrono::steady_clock::now();
    }

	Timer& endWatch() {
        _endWatchTime = std::chrono::steady_clock::now();
        _elapsedTime = getElapsedTime();
        return *this;
    }

    double elapsedTime() {
        return _elapsedTime;
    }

    void printElapsedTime(const std::string& str = "") {
        if (str.empty())
            std::cout << "elapsed time: " << _elapsedTime << "us\n";
        else
            std::cout << str << _elapsedTime << "us\n";
    }

private:
    double getElapsedTime() {
        const auto& _start =
            std::chrono::time_point_cast<std::chrono::microseconds>(_startWatchTime)
            .time_since_epoch()
            .count();
        const auto& _end = std::chrono::time_point_cast<std::chrono::microseconds>(_endWatchTime)
            .time_since_epoch()
            .count();
        auto fps = _end - _start;
        return fps;
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> _startWatchTime;
    std::chrono::time_point<std::chrono::steady_clock> _endWatchTime;
    double _elapsedTime{};
};
