#include "layer.hpp"
#include "activation_data.hpp"
#include <iostream>
#include <cstdlib>
#include <string>

template <class T> T 
random_weight(bool weight_range)
{
    unsigned set_range = arc4random();
    T rVal = arc4random() % 1000;
    if(weight_range){
        if(set_range % 2 == 0)
            return rVal;
        else
            return rVal * -1;
    }else{
        return rVal;
    }
}

template <class T> 
Matrix<T> :: Matrix(uint32_t num_rows)
{
    this->num_rows = num_rows;
    this->num_cols = 0;
    this->index = 0;

    this->init_matrix_variables();
}

template <class T> 
Matrix<T> :: Matrix(uint32_t num_rows, uint32_t num_cols)
{
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->index = 0;

    this->init_matrix_variables();
}

// template <class T> 
// Matrix<T> :: ~Matrix()
// {
//     free(this->_matrix);
// }

template <class T> inline void
Matrix<T> :: set_matrix(Matrix<T> *new_mat)
{
    this->_matrix = new_mat->get_matrix();
}

template <class T> inline void
Matrix<T> :: push(T val)
{
    this->_matrix[this->index] = val;
    this->index++;
}

template <class T> void
Matrix<T> :: init_matrix_variables()
{
    if(this->num_cols != 0){
        this->_matrix = (T *)calloc(this->num_rows, sizeof(T));
    }else{
        this->_matrix = (T *)calloc(this->num_cols * this->num_rows, sizeof(T));
    }
}

template <class T> inline void
Matrix<T> :: random(bool weight_range)
{
    uint32_t i = 0;
    if(this->num_cols == 0){
        while(i < this->num_rows){
            this->_matrix[i] = random_weight<T>(weight_range); i++;}
    }else{
        while(i < this->num_cols * this->num_rows){
            this->_matrix[i] = random_weight<T>(weight_range); i++;}
    }
}

template <class T> inline T
Matrix<T> :: get_value(uint32_t row_idx)
{
    return this->_matrix[row_idx];
}

template <class T> inline void
Matrix<T> :: point_edit(uint32_t row_idx, T new_val)
{
    this->_matrix[row_idx] = new_val;
}

template <class T> inline T
Matrix<T> :: get_value(uint32_t row_idx, uint32_t col_idx)
{
    return this->_matrix[((this->num_rows * this->num_cols) - (this->num_cols * (this->num_rows - row_idx))) + col_idx];
}

template <class T> inline void
Matrix<T> :: point_edit(uint32_t row_idx, uint32_t col_idx, T new_val)
{
    this->_matrix[((this->num_rows * this->num_cols) - (this->num_cols * (this->num_rows - row_idx))) + col_idx] = new_val;
}

template <class T> inline T *&
Matrix<T> :: get_matrix()
{
    return _matrix;
}

template <class T>
Layer<T> :: Layer(bool weight_range, uint32_t num_rows, uint32_t num_cols, std::string act_func)
{
    this->neuron_values = new Matrix<T>(num_rows);
    this->weight_values = new Matrix<T>(num_rows, num_cols);
    this->act_func = act_func;

    this->weight_values->random(weight_range);
}

template <class T>
Layer<T> :: Layer(uint32_t num_rows, std::string act_func)
{
    this->act_func = act_func;
    this->neuron_values = new Matrix<T>(num_rows);
}

// template <class T>
// Layer<T> :: ~Layer()
// {
//     delete this->neuron_values;
//     delete this->weight_values;
// }

template <class T> inline uint32_t
Layer<T> :: get_row()
{
    return this->neuron_values->num_rows;
}

template <class T> inline uint32_t
Layer<T> :: get_col()
{
    return this->weight_values->num_cols;
}

template <class T> inline void
Layer<T> :: set_variable_mat(Matrix<T> *new_var_mat)
{
    this->weight_values->set_matrix(new_var_mat);
}

template <class T> inline void
Layer<T> :: set_neuron_mat(Matrix<T> *new_neuron_mat)
{
    this->neuron_values->set_matrix(new_neuron_mat);
}

template <class T> inline Matrix<T> * 
Layer<T> :: get_variable_mat()
{
    return this->weight_values;
}

template <class T> inline Matrix<T> * 
Layer<T> :: get_neuron_mat()
{
    return this->neuron_values;
}

template <class T> inline T
Layer<T> :: get_point_weight_value(uint32_t row_idx, uint32_t col_idx)
{
    return this->weight_values->get_value(row_idx, col_idx);
}

template <class T> void
Layer<T> :: feed_forward(Layer<T> *&next_layer)
{
    uint32_t i = 0, j = 0;
    Matrix<T> *temp = new Matrix<T>(this->get_col());
    T node_input = 0;

    while(i < this->get_row())
    {
        this->neuron_values->point_edit(i, __activate_value__<T>(this->neuron_values->get_value(i), this->act_func));
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

template <class T> void
Layer<T> :: activate_output_layer()
{
    if(act_func == "softmax" || act_func == "hardmax"){
        __activate_value__<T>(this->neuron_values->get_matrix(), this->get_row(), this->act_func);
    }
    else{
        uint32_t i = 0;
        while(i < this->get_row())
        {
            this->neuron_values->point_edit(i, __activate_value__<T>(this->neuron_values->get_value(i), this->act_func));
            i++;
        }
    }
    
}

template <class T> void
Layer<T> :: toString()
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