
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include "layer.hpp"

#ifndef model_hpp
#define model_hpp
#endif

template <typename T>
class Sequencial
{
public:
    Sequencial(uint32_t epochs, bool weight_range, bool print_to_cons);
    Sequencial(uint32_t epochs, uint32_t batches, bool weight_range, bool print_to_cons);
    void add(uint32_t num_neurons, std::string act_func);
    void run(void);
    void format_input_data(std::vector<std::vector<T> >);
    void format_output_data(std::vector<std::vector<T> >);
    void format_input_data(T **);
    void format_output_data(T **);

    void initialize_global_variables(void);

    void toCons(uint32_t);
private:
    uint32_t num_epochs;
    uint32_t num_batches;
    Matrix<T> **y_data;
    Matrix<T> **x_data;
    uint32_t *input_shape;
    bool weight_range;
    bool print_to_cons;
    std::vector<uint32_t> lSize_arr;
    std::vector<std::string> act_func_arr;
    Layer<T> **network;
    Matrix<T> **variable_history;

    void record_data(void);
    void update(void);
};
