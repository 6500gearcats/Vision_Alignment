#include "layer.hpp"
#include <vector>
#include <string>

#ifndef optimize_trained_net_hpp
#define optimize_trained_net_hpp

class Network
{
public:
    Network(std::vector<uint32_t> lSize_arr, std::vector<uint32_t> act_func_arr);
    void get_data(std::string file);
    void run(void);

private:
    std::vector<uint32_t> lSize_arr;
    std::vector<uint32_t> act_func_arr;

    Matrix **weight_matrix;
};

#endif