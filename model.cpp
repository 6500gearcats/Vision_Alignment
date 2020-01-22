#include <iostream>
#include <vector>
#include <string>

#include "layer.hpp"
#include "model.hpp"

Sequencial:: Sequencial(uint32_t epochs, bool weight_range, bool print_to_cons)
{
    this->num_epochs = epochs;
    this->weight_range = weight_range;
    this->print_to_cons = print_to_cons;
    this->network = (Layer **)malloc(sizeof(Layer *));
    this->variable_history = (Matrix **)malloc(sizeof(Matrix *));
}

Sequencial :: Sequencial(uint32_t epochs, uint32_t batches, bool weight_range, bool print_to_cons)
{
    this->num_epochs = epochs;
    this->num_batches = batches;
    this->weight_range = weight_range;
    this->print_to_cons = print_to_cons;
    this->network = (Layer **)malloc(sizeof(Layer *));
    this->variable_history = (Matrix **)malloc(sizeof(Matrix *));
}

void Sequencial :: add(uint32_t num_neurons, std::string act_func)
{
    this->lSize_arr.push_back(num_neurons);
    this->act_func_arr.push_back(act_func);
    this->network = (Layer **)realloc(this->network, this->lSize_arr.size() * sizeof(Layer *));
    this->variable_history = (Matrix **)realloc(this->variable_history, this->lSize_arr.size() * sizeof(Matrix *));
}

void Sequencial :: initialize_global_variables()
{
    uint32_t i = 0;
    while(i < this->lSize_arr.size())
    {
        if(i < this->lSize_arr.size() - 1)
            this->variable_history[i] = new Matrix(this->lSize_arr[i] * this->lSize_arr[i+1]);
        
        this->network[i] = new Layer(this->weight_range, this->lSize_arr[i], this->lSize_arr[i+1], this->act_func_arr[i]);
        i++;
    }
    this->network[this->lSize_arr.size() - 1] = new Layer(this->lSize_arr[this->lSize_arr.size() - 1], this->act_func_arr[this->lSize_arr.size() - 1]);
}

void Sequencial :: run()
{
    uint32_t n = 0, i = 0, input_idx = 0;
    while(n < this->num_epochs)
    {
        if((n % this->input_shape[0]) == 0)
            input_idx = 0;
        else
            input_idx ++;
        
        this->network[0]->set_neuron_mat(this->x_data[input_idx]);
        while(i < this->lSize_arr.size() - 1)
        {
            this->network[i]->feed_forward(network[i+1]);
            i++;
        }i=0;
        this->network[this->lSize_arr.size() - 1]->activate_output_layer();
        
        if(this->print_to_cons)
            this->toCons(n+1);

        this->record_data();
        this->update();
        n++;
    }
}

void Sequencial :: record_data()
{
    uint32_t i = 0;
    while(i < this->lSize_arr.size() - 1)
    {
        this->variable_history[i]->set_matrix(this->network[i]->get_variable_mat());
        i++;
    }
}

/**
 * This is where the backprop happens
 * USE THE MATH
 * Everything is set to 1 as seen in the point_edit method
 * */
void Sequencial :: update()
{
    for(int i = 0; i < lSize_arr[1]; ++i)
    {
        for(int j = 0; j < lSize_arr[2]; ++j)
        {
            this->variable_history[1]->point_edit(i, j, 1);
        }
    }

    for(int i = 0; i < lSize_arr[0]; ++i)
    {
        for(int j = 0; j < lSize_arr[2]; ++j)
        {
            this->variable_history[0]->point_edit(i, j, 1);
        }
    }

    for(int i = 0; i < this->lSize_arr.size() - 1; ++i)
    {
        this->network[i]->set_variable_mat(this->variable_history[i]);
    }
}

void Sequencial :: format_input_data(std::vector< std::vector<float> > in)
{
    uint32_t i = 0, j = 0;
    this->x_data = (Matrix **)calloc(in.size(), sizeof(Matrix *));
    while(i < in.size())
    {
        this->x_data[i] = new Matrix(in[i].size());
        while(j < in[i].size())
        {
            this->x_data[i]->push(in[i][j]);
            j++;
        }j=0;
        i++;
    }
    this->input_shape = (uint32_t *)calloc(2, sizeof(uint32_t));
    this->input_shape[0] = in.size();
    this->input_shape[1] = in[0].size();
}

void Sequencial :: format_output_data(std::vector<std::vector<float> > out)
{
    uint32_t i = 0, j = 0;
    this->y_data = (Matrix **)calloc(out.size(), sizeof(Matrix *));
    while(i < out.size())
    {
        this->y_data[i] = new Matrix(out[i].size());
        while(j < out[i].size())
        {
            this->y_data[i]->push(out[i][j]);
            j++;
        }j=0;
        i++;
    }
}

void Sequencial :: toCons(uint32_t iteration)
{
    uint32_t i = 0, j = 0;
    std::cout << "\n============= " << iteration << " =============\n";
    while(i < this->lSize_arr.size() - 1)
    {
        this->network[i]->toString();
        i++;
    }
    std::cout <<  "==============================";
    std::cout << "\n\n\n";
}