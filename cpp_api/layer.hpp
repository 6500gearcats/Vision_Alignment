#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

#ifndef layer_hpp
#define layer_hpp

class Matrix  
{
public:
    Matrix(uint32_t num_rows);
    Matrix(uint32_t num_rows, uint32_t num_cols);
    //~Matrix(void);

    void init_matrix_variables(void);

    void point_edit(uint32_t row_idx, float new_val);
    void point_edit(uint32_t row_idx, uint32_t col_idx, float new_val);
    float get_value(uint32_t row_idx);
    float get_value(uint32_t row_idx, uint32_t col_idx);

    void set_matrix(Matrix *);
    float *& get_matrix(void);

    void push(float val);

    void random(bool weight_range);
    void bw_random(bool weight_range);

    uint32_t num_rows;
    uint32_t num_cols;
private:
    float *_matrix;
    uint32_t index;
};

class Layer
{
public:
    Layer(bool weight_range, uint32_t num_rows, uint32_t num_cols, std::string act_func, bool hasBias);
    Layer(uint32_t num_rows, std::string act_func);
    //~Layer(void);

    uint32_t get_row(void);
    uint32_t get_col(void);

    void set_variable_mat(Matrix *);
    Matrix * get_variable_mat(void);
    void set_neuron_mat(Matrix *);
    Matrix * get_neuron_mat(void);
    void set_bias_mat(Matrix *);
    Matrix * get_bias_mat(void);

    float get_point_weight_value(uint32_t row_idx, uint32_t col_idx);
    float get_point_neuron_value(uint32_t row_idx);
    float get_point_bias_value(uint32_t col_idx);

    void feed_forward(Layer *&next_layer);
    void activate_output_layer(void);
    void dealloc_variables(void);

    void toString(void);
private:
    Matrix *neuron_values;
    Matrix *weight_values;
    Matrix *bias_values;
    bool hasBias;
    std::string act_func;
};

#endif
