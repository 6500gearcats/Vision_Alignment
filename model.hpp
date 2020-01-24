#include <cmath>
#include <iostream>
#include <vector>
#include <string>

#include "layer.hpp"

#ifndef model_hpp
#define model_hpp

class Sequencial
{
public:
    Sequencial(uint32_t epochs, bool weight_range, bool print_to_cons, std::string, float);
    Sequencial(uint32_t epochs, uint32_t batches, bool weight_range, bool print_to_cons, std::string, float);
    void add(uint32_t num_neurons, std::string act_func);
    void bias(void);
    void run(void);
    void format_input_data(std::vector<std::vector<float> > &);
    void format_output_data(std::vector<std::vector<float> > &);
    void format_input_data(float **&, uint32_t num_sets);
    void format_output_data(float **&);

    void initialize_global_variables(void);

    void toCons(uint32_t);
    void toCons(uint32_t, Matrix **&);
private:
    uint32_t num_epochs;
    uint32_t num_batches;
    Matrix **y_data;
    Matrix **x_data;
    uint32_t *input_shape;
    bool weight_range;
    bool print_to_cons;
    std::string cost_name;
    float learning_rate;
    std::vector<uint32_t> lSize_arr;
    std::vector<bool> bias_layer_arr;
    uint32_t bias_iteration;
    std::vector<std::string> act_func_arr;
    Layer **network;
    Matrix **variable_history;
    Matrix **bias_history;

    void record_data(void);
    void update_stochatic(uint32_t y_index);
    void update_batch(float tot_cost);
    void free_mat_data(void);
};

#endif
