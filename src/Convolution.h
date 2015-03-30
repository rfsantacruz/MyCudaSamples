/*
 * Convolution.h
 *
 *  Created on: Mar 30, 2015
 *      Author: rfsantacruz
 */

#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

void conv1DHost(const float* input, const int input_width, const float* mask, const int mask_width, float* output);

void conv2DHost(const float* input, const int input_width, const float* mask, const int mask_width, float* output);

#endif /* CONVOLUTION_H_ */
