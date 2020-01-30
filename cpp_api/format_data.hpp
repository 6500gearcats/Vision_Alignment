//
//  model.hpp
//  AI_Backbone
//
//  Created by maxwell on 1/5/2020.
//  Copyright © 2019 organized-organization. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>

#ifndef format_data_hpp
#define format_data_hpp

#define NUM_COORIDINATES 8
#define COORIDINATES_DIMENTIONALITY 2
#define MAX_PIXEL_LEN 155

// network tables ntcore library

/**
 * Data should be normalized before inputed into the neural network
 *      i.e. [0, 1] or [-1, 1]
 * The neural network will run twice(like a batch size of 2) once with x values, once with y values, that is epoch 1
 * @param location of file
 * @return vector of type vector of type T
 * */



#endif