#include <iostream>
#include <vector>
#include <string>

#include "layer.hpp"
#include "model.hpp"
#include "activation_data.hpp"

Sequencial:: Sequencial(uint32_t epochs, bool weight_range, bool print_to_cons, std::string cost_name, float learning_rate)
{
    this->num_epochs = epochs;
    this->num_batches = 0;
    this->weight_range = weight_range;
    this->print_to_cons = print_to_cons;
    this->network = (Layer **)malloc(sizeof(Layer *));
    this->variable_history = (Matrix **)malloc(sizeof(Matrix *));
    this->cost_name = cost_name;
    this->learning_rate = learning_rate;
    this->bias_iteration = 0;
}

Sequencial :: Sequencial(uint32_t epochs, uint32_t batches, bool weight_range, bool print_to_cons, std::string cost_name, float learning_rate)
{
    this->num_epochs = epochs;
    if(batches == 0) std::terminate();
    this->num_batches = batches;
    this->weight_range = weight_range;
    this->print_to_cons = print_to_cons;
    this->network = (Layer **)malloc(sizeof(Layer *));
    this->variable_history = (Matrix **)malloc(sizeof(Matrix *));
    this->cost_name = cost_name;
    this->learning_rate = learning_rate;
    this->bias_iteration = 0;
}

void Sequencial :: add(uint32_t num_neurons, std::string act_func)
{
    this->lSize_arr.push_back(num_neurons);
    this->act_func_arr.push_back(act_func);
    this->network = (Layer **)realloc(this->network, this->lSize_arr.size() * sizeof(Layer *));
    this->variable_history = (Matrix **)realloc(this->variable_history, this->lSize_arr.size() * sizeof(Matrix *));
    this->bias_layer_arr.push_back(false);
}

void Sequencial :: bias()
{
    this->bias_layer_arr.pop_back();
    this->bias_layer_arr.push_back(true);
}

void Sequencial :: initialize_global_variables()
{
    uint32_t i = 0;
    this->bias_history = (Matrix **)calloc(this->lSize_arr.size() - 1, sizeof(Matrix *));
    this->bias_layer_arr.pop_back();
    while(i < this->lSize_arr.size() - 1)
    {
        this->bias_history[i] = new Matrix(this->lSize_arr[i+1]);
        this->variable_history[i] = new Matrix(this->lSize_arr[i], this->lSize_arr[i+1]);
        this->network[i] = new Layer(this->weight_range, this->lSize_arr[i], this->lSize_arr[i+1], this->act_func_arr[i], this->bias_layer_arr[i]);
        i++;
    }
    this->network[this->lSize_arr.size() - 1] = new Layer(this->lSize_arr[this->lSize_arr.size() - 1], this->act_func_arr[this->lSize_arr.size() - 1]);
}

void Sequencial :: run()
{
    uint32_t n = 0, i = 0, input_idx = 0, batch_idx = 0;
    Matrix *x_input_disposable;
    Matrix **n_batch_history = (Matrix **)calloc(this->num_batches, sizeof(Matrix *));

    float tot_cost = 0;

    while(n < this->num_epochs)
    {
        if((n % this->input_shape[0]) == 0)
            input_idx = 0;
        else
            input_idx ++;

        x_input_disposable = new Matrix(this->input_shape[1]);
        while(i < this->input_shape[1]){
            x_input_disposable->push(this->x_data[input_idx]->get_value(i));
            i++;
        }i=0;
        
        this->network[0]->set_neuron_mat(x_input_disposable);
        delete x_input_disposable;
        while(i < this->lSize_arr.size() - 1)
        {
            this->network[i]->feed_forward(network[i+1]);
            i++;
        }i=0;
        this->network[this->lSize_arr.size() - 1]->activate_output_layer();

        if(this->num_batches != 0){
            n_batch_history[batch_idx] = new Matrix(this->lSize_arr[this->lSize_arr.size() - 1]);
            while(i < this->lSize_arr[this->lSize_arr.size() - 1]){
                n_batch_history[batch_idx]->point_edit(i, this->network[this->lSize_arr.size() - 1]->get_point_neuron_value(i));
                i++;
            }i=0;
            batch_idx++;
        }
        if(this->num_batches != 0){
            tot_cost += __cost_value_derivative__(n_batch_history[batch_idx], this->y_data[input_idx], this->lSize_arr[this->lSize_arr.size() - 1], this->cost_name);
            if(((n+1) % this->num_batches) == 0){
                this->record_data();
                this->update_batch(tot_cost);
                
                std::cout << n << std::endl;
                if(this->print_to_cons)
                    this->toCons(n+1,  n_batch_history);
                batch_idx = 0;
            }
        }else{
            this->record_data();
            this->update_stochatic(input_idx);

            if(this->print_to_cons)
                this->toCons(n+1);
        }
        
        n++;
    }

    this->free_mat_data();
}

void Sequencial :: record_data()
{
    uint32_t i = 0;
    while(i < this->lSize_arr.size() - 1){
        this->variable_history[i]->set_matrix(this->network[i]->get_variable_mat());
        if(this->bias_layer_arr[i]){
            this->bias_history[i]->set_matrix(this->network[i]->get_bias_mat());
        }
        i++;
    }
}

/**
 * This is where the backprop happens
 * USE THE MATH
 * Everything is set to 1 as seen in the point_edit method
 * */
void Sequencial :: update_stochatic(uint32_t y_index)
{
    float dirivative = 0;
    for(int i = 0; i < lSize_arr[1]; ++i)
    {
        for(int j = 0; j < lSize_arr[2]; ++j)
        {
            dirivative = -1 * __cost_value_derivative__(this->network[2]->get_point_neuron_value(j), this->y_data[y_index]->get_value(j), this->cost_name);
            dirivative *= __activation_dirivative__(this->network[2]->get_point_neuron_value(j), this->act_func_arr[2]);
            dirivative *= this->network[1]->get_point_neuron_value(i);
            dirivative *= this->learning_rate;
            this->variable_history[1]->point_edit(i, j, this->variable_history[1]->get_value(i, j) - dirivative); //this->network[i]->get_point_weight_value(i, j));
        }
    }dirivative=0;

    for(int i = 0; i < lSize_arr[0]; ++i)
    {
        for(int j = 0; j < lSize_arr[1]; ++j)
        {
            dirivative = -1 * __cost_value_derivative__(this->network[2]->get_point_neuron_value(j), this->y_data[y_index]->get_value(j), this->cost_name);
            dirivative *= __activation_dirivative__(this->network[2]->get_point_neuron_value(j), this->act_func_arr[2]);
            dirivative *= this->network[1]->get_point_neuron_value(i);
            dirivative *= this->learning_rate;
            this->variable_history[0]->point_edit(i, j, 1); //this->network[i]->get_point_weight_value(i, j));
        }
    }
    
    for(int i = 0; i < this->lSize_arr.size() - 1; ++i)
    {
        if(this->bias_layer_arr[i]){
            for(int j = 0; j < this->lSize_arr[i+1]; ++j)
            {
                this->bias_history[i]->point_edit(j, this->bias_history[i]->get_value(j));
            }
        }
    }

    for(int i = 0; i < this->lSize_arr.size() - 1; ++i)
    {
        this->network[i]->set_variable_mat(this->variable_history[i]);
        if(this->bias_layer_arr[i]){
            this->network[i]->set_bias_mat(this->bias_history[i]);
        }
    }
}

void Sequencial :: update_batch(float tot_cost)
{
    float dirivative = 0;
    for(int i = 0; i < lSize_arr[1]; ++i)
    {
        for(int j = 0; j < lSize_arr[2]; ++j)
        {
            dirivative = tot_cost;
            dirivative *= __activation_dirivative__(this->network[2]->get_point_neuron_value(j), this->act_func_arr[2]);
            dirivative *= this->network[1]->get_point_neuron_value(i);
            dirivative *= this->learning_rate;
            this->variable_history[1]->point_edit(i, j, this->variable_history[1]->get_value(i, j) - dirivative); //this->network[i]->get_point_weight_value(i, j));
        }
    }dirivative=0;

    for(int i = 0; i < lSize_arr[0]; ++i)
    {
        for(int j = 0; j < lSize_arr[1]; ++j)
        {
            dirivative = tot_cost;
            dirivative *= __activation_dirivative__(this->network[2]->get_point_neuron_value(j), this->act_func_arr[2]);
            dirivative *= this->network[1]->get_point_neuron_value(i);
            dirivative *= this->learning_rate;
            this->variable_history[0]->point_edit(i, j, 1); //this->network[i]->get_point_weight_value(i, j));
        }
    }
    
    for(int i = 0; i < this->lSize_arr.size() - 1; ++i)
    {
        if(this->bias_layer_arr[i]){
            for(int j = 0; j < this->lSize_arr[i+1]; ++j)
            {
                this->bias_history[i]->point_edit(j, this->bias_history[i]->get_value(j));
            }
        }
    }

    for(int i = 0; i < this->lSize_arr.size() - 1; ++i)
    {
        this->network[i]->set_variable_mat(this->variable_history[i]);
        if(this->bias_layer_arr[i]){
            this->network[i]->set_bias_mat(this->bias_history[i]);
        }
    }
}

void Sequencial :: free_mat_data()
{
    uint32_t i = 0;
    while(i < this->lSize_arr.size() - 1)
    {
        this->network[i]->dealloc_variables();
        i++;
    }
}

void Sequencial :: format_input_data(std::vector< std::vector<float> > &in)
{
    uint32_t i = 0, j = 0;
    this->x_data = (Matrix **)calloc(in.size(), sizeof(Matrix *));
    while(i < in.size()){
        this->x_data[i] = new Matrix(in[0].size());
        while(j < in[i].size()){
            this->x_data[i][j] = in[i][j];
            j++;
        }j=0;
        i++;
    }
    this->input_shape = (uint32_t *)calloc(2, sizeof(uint32_t));
    this->input_shape[0] = in.size();
    this->input_shape[1] = in[0].size();
    in.clear();
}

void Sequencial :: format_output_data(std::vector<std::vector<float> > &out)
{
    uint32_t i = 0, j = 0;
    this->y_data = (Matrix **)calloc(out.size(), sizeof(Matrix *));
    while(i < out.size()){
        this->y_data[i] = new Matrix(out[0].size());
        while(j < out[i].size()){
            this->y_data[i][j] = out[i][j];
            j++;
        }j=0;
        i++;
    }
    out.clear();
}

void Sequencial :: format_input_data(float **&in, uint32_t num_sets)
{
    uint32_t i = 0, j = 0;
    this->x_data = (Matrix **)calloc(num_sets, sizeof(Matrix *));

    while(i < num_sets){
        this->x_data[i] = new Matrix(this->lSize_arr[0]);
        while(j < this->lSize_arr[0]){
            this->x_data[i]->push(in[i][j]);
            j++;
        }j=0;
        i++;
    }

    this->input_shape = (uint32_t *)calloc(2, sizeof(uint32_t));
    this->input_shape[0] = num_sets;
    this->input_shape[1] = this->lSize_arr[0];
    free(in);
}

void Sequencial :: format_output_data(float **&out)
{
    uint32_t i = 0, j = 0;
    this->y_data = (Matrix **)calloc(this->input_shape[0], sizeof(Matrix *));
    while(i < input_shape[0]){
        this->y_data[i] = new Matrix(this->lSize_arr[this->lSize_arr.size() - 1]);
        while(j < this->lSize_arr[this->lSize_arr.size() - 1]){
            this->y_data[i]->push(out[i][j]);
            j++;
        }j=0;
        i++;
    }
    free(out);
}

uint32_t y_data_index = 0;
void Sequencial :: toCons(uint32_t iteration) // make another toCons method that takes in an array of neurons to output the history
//                                               this should be specific for batch output
{
    uint32_t i = 0;
    if(((iteration-1) % this->input_shape[0]) == 0)
        y_data_index = 0;
    else
        y_data_index++;

    std::cout << "\n============= " << iteration << " =============";
    while(i < this->lSize_arr.size() - 1)
    {
        this->network[i]->toString();
        i++;
    }i=0;
    std::cout << "\nOutput Neuron(s)\n";
    while(i < this->lSize_arr[this->lSize_arr.size() - 1])
    {
        std::cout << this->network[this->lSize_arr.size() - 1]->get_point_neuron_value(i) << std::endl;
        i++;
    }i=0;

    std::cout << "\nReal Output(s)\n";
    while(i < this->lSize_arr[this->lSize_arr.size() - 1])
    {
        std::cout << this->y_data[y_data_index]->get_value(i) << std::endl;
        i++;
    }
    std::cout <<  "==============================";
    std::cout << "\n\n\n";

}

void Sequencial :: toCons(uint32_t iteration, Matrix **&n_values)
{
    uint32_t j = 0, i = 0;
    std::cout << "\n============= " << iteration << " =============";
    std::cout << "\nNeuron Outputs: \n";
    while(i < this->num_batches)
    {
        while(j < this->lSize_arr[this->lSize_arr.size() - 1])
        {
            std::cout << n_values[i]->get_value(j) << "    Real Value: " << this->y_data[i]->get_value(j) << std::endl;
            j++;
        }j=0;
        i++;
    }
}