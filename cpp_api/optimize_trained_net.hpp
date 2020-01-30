#include <vector>
#include <string>
#include "layer.hpp"

#ifndef optimize_trained_net_hpp
#define optimize_trained_net_hpp

class Network
{
public:
    Network(std::vector<uint32_t> lSize_arr, std::vector<std::string> act_func_arr, std::vector<bool> bias_arr);
    void get_data(std::string file);
    void init_net(void);
    void init_input_data(float **, uint32_t num_sets, uint32_t set_size);
    void toString(void);
    uint32_t get_output(uint32_t data_idx);

private:
    std::vector<uint32_t> lSize_arr;
    std::vector<std::string> act_func_arr;
    std::vector<bool> bias_arr;
    float **input_data;
    float ***weight_matrix;
    float **bias_matrix;
    float **pseudo_net;
};

#endif