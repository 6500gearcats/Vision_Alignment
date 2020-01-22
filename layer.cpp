#include "layer.hpp"
#include "activation_data.hpp"
#include <iostream>
#include <cstdlib>
#include <string>

float random_weight(bool weight_range)
{
    unsigned set_range = arc4random();
    float rVal = arc4random() % 1000;
    if(weight_range){
        if(set_range % 2 == 0)
            return rVal;
        else
            return rVal * -1;
    }else{
        return rVal;
    }
}

Matrix :: Matrix(uint32_t num_rows)
{
    this->num_rows = num_rows;
    this->num_cols = 0;
    this->index = 0;

    this->init_matrix_variables();
}

Matrix :: Matrix(uint32_t num_rows, uint32_t num_cols)
{
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->index = 0;

    this->init_matrix_variables();
}

inline void Matrix :: set_matrix(Matrix *new_mat)
{
    this->_matrix = new_mat->get_matrix();
}

inline void Matrix :: push(float val)
{
    this->_matrix[this->index] = val;
    this->index++;
}

void Matrix :: init_matrix_variables()
{
    if(this->num_cols != 0){
        this->_matrix = (float *)calloc(this->num_rows, sizeof(float));
    }else{
        this->_matrix = (float *)calloc(this->num_cols * this->num_rows, sizeof(float));
    }
}

inline void Matrix :: random(bool weight_range)
{
    uint32_t i = 0;
    if(this->num_cols == 0){
        while(i < this->num_rows){
            this->_matrix[i] = random_weight(weight_range); i++;}
    }else{
        while(i < this->num_cols * this->num_rows){
            this->_matrix[i] = random_weight(weight_range); i++;}
    }
}

inline float Matrix :: get_value(uint32_t row_idx)
{
    return this->_matrix[row_idx];
}

inline void Matrix :: point_edit(uint32_t row_idx, float new_val)
{
    this->_matrix[row_idx] = new_val;
}

inline float Matrix :: get_value(uint32_t row_idx, uint32_t col_idx)
{
    return this->_matrix[((this->num_rows * this->num_cols) - (this->num_cols * (this->num_rows - row_idx))) + col_idx];
}

inline void Matrix :: point_edit(uint32_t row_idx, uint32_t col_idx, float new_val)
{
    this->_matrix[((this->num_rows * this->num_cols) - (this->num_cols * (this->num_rows - row_idx))) + col_idx] = new_val;
}

inline float *& Matrix :: get_matrix()
{
    return _matrix;
}

Layer :: Layer(bool weight_range, uint32_t num_rows, uint32_t num_cols, std::string act_func)
{
    this->neuron_values = new Matrix(num_rows);
    this->weight_values = new Matrix(num_rows, num_cols);
    this->act_func = act_func;

    this->weight_values->random(weight_range);
}

Layer :: Layer(uint32_t num_rows, std::string act_func)
{
    this->act_func = act_func;
    this->neuron_values = new Matrix(num_rows);
}

inline uint32_t Layer :: get_row()
{
    return this->neuron_values->num_rows;
}

inline uint32_t Layer :: get_col()
{
    return this->weight_values->num_cols;
}

inline void Layer :: set_variable_mat(Matrix *new_var_mat)
{
    this->weight_values->set_matrix(new_var_mat);
}

inline void Layer :: set_neuron_mat(Matrix *new_neuron_mat)
{
    this->neuron_values->set_matrix(new_neuron_mat);
}

inline Matrix * Layer :: get_variable_mat()
{
    return this->weight_values;
}

inline Matrix * Layer :: get_neuron_mat()
{
    return this->neuron_values;
}

inline float Layer :: get_point_weight_value(uint32_t row_idx, uint32_t col_idx)
{
    return this->weight_values->get_value(row_idx, col_idx);
}

void Layer :: feed_forward(Layer *&next_layer)
{
    uint32_t i = 0, j = 0;
    Matrix *temp = new Matrix(this->get_col());
    float node_input = 0;

    while(i < this->get_row())
    {
        this->neuron_values->point_edit(i, __activate_value__(this->neuron_values->get_value(i), this->act_func));
        i++;
    }i=0;

    while(i < this->get_col())
    {
        while(j < this->get_row())
        {
            node_input += this->neuron_values->get_value(j) * this->get_point_weight_value(j, i);
            j++;
        }j=0; node_input=0;
        temp->push(node_input);
        i++;
    }
    next_layer->set_neuron_mat(temp);
}

void Layer :: activate_output_layer()
{
    if(act_func == "softmax" || act_func == "hardmax"){
        __activate_value__(this->neuron_values->get_matrix(), this->get_row(), this->act_func);
    }
    else{
        uint32_t i = 0;
        while(i < this->get_row())
        {
            this->neuron_values->point_edit(i, __activate_value__(this->neuron_values->get_value(i), this->act_func));
            i++;
        }
    }
    
}

void Layer :: toString()
{
    uint32_t i = 0, j = 0;
    std::cout << "Neuron Values\n";
    while(i < this->get_row())
    {
        std::cout << this->neuron_values->get_value(i) << std::endl;
        i++;
    }i=0;

    std::cout << "\n\nWeight Values\n";
    while(i < this->get_row())
    {
        while(j < this->get_col())
        {
            std::cout << this->weight_values->get_value(i, j) << "    ";
            j++;
        }j=0;
        std::cout << std::endl;
        i++;
    }
}