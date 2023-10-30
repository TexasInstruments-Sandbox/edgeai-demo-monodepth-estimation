#  Copyright (C) 2023 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import math
import numpy as np
import onnxruntime
import tflite_runtime.interpreter as tfl

import cv2 as cv
import yaml

onnxruntime.set_default_logger_severity(3) #suppress some warnings that the logger prints

TENSOR_TIOVX_ALIGN_BYTES = 128 # tensors allocated in min block size; start of next tensor will be aligned with this. Found in https://github.com/TexasInstruments/edgeai-gst-plugins/blob/8201082cf590473ecbd95c3f73225968adcdcd89/ext/ti/gsttidlinferer.cpp#L94


class ModelRunner():
    '''
    This class handles model information to help genereate the correct gstreamer string and decode the output tensors provided from gstreamer
    '''
    def __init__(self, modeldir, paramsfile=None, modelfile=None):
        self.modeldir = modeldir

        #params file contains all crucial information for running the model. tidlinferer plugin will leverage this. This file is generated during model compilation
        if not paramsfile or not os.path.exists(paramsfile):
            self.params = yaml.safe_load(open(os.path.join(modeldir, 'param.yaml'), 'r'))
        else:
            self.params = yaml.safe_load(open(paramsfile, 'r'))


        self.modelfile = os.path.join(modeldir, self.params['session']['model_path'])
        assert os.path.exists(self.modelfile), 'Could not find model file ' + self.modelfile

        if type(self.params['preprocess']['resize']) == list:
            self.model_width, self.model_height = self.params['preprocess']['resize']
        else: 
            self.model_width = self.model_height = self.params['preprocess']['resize']

    def calculate_output_tensor_sizes(self):
        '''
        Calculate the length and size of tensors within the buffer of data supplied from gstreamer. This will use the params and model input/output information to generate these values, which are static to the model, but vary from model to model

        return: self.tensor_offsets is 2d list that is returned and saved as part of the ModelRunner object.
            First list/index (outer) is for which tensor in the output,
            Second list/index (interior) is 2-element list. The first value is number of bytes of data composing the tensor, and second is length of the tensor's buffer data; the next buffer starts as the end of that alignment.
        return: self.tensor_types, which holds the data type for the elements in the list

        Currently only works for select few object detection models like mobv2SSD lite and YOLOX. This needs significantly more logic to capture the near-infinite number of configurations for arbitrary models. This function should be extended to account for those on an as-needed basis

        '''
        print('Calculating output tensor dimensions and offsets...')
        self.tensor_offsets = []
        self.tensor_types = []

        #define a few helper functions
        def bytes_from_type_and_elements(tensor_type, num_el):
            '''
            Use tensor type and number of elements in the tensor to calculate size in bytes. 
            return: size in bytes
            return: numpy type corresponding to this tensor. This is determined by the the name of the type as supplied by the runtime and/or params.yaml file
            '''
            if 'float' in tensor_type: 
                num_bytes = num_el * 4
                np_type = np.float32
            elif 'int8' in tensor_type: 
                num_bytes = num_el
                np_type = np.uint8
            elif 'int16' in tensor_type: 
                num_bytes = num_el * 2
                np_type = np.uint16
            elif 'int32' in tensor_type: 
                num_bytes = num_el * 4
                np_type = np.uint32
            elif 'int64' in tensor_type: 
                num_bytes = num_el * 8 
                np_type = np.uint64
            else: raise ValueError('Do not recognize type ' + tensor_type)
            return num_bytes, np_type
        
        def align(num_bytes):
            '''
            Multiple tensors are aligned to a consistent block size. The next tensor will start at the nearest block (ceiling)
            '''
            num_blocks = math.ceil(num_bytes / TENSOR_TIOVX_ALIGN_BYTES)
            num_bytes_aligned = num_blocks * TENSOR_TIOVX_ALIGN_BYTES
            return num_bytes_aligned
        
        def get_size_from_output_details_onnx(od):
            '''
            Retrieve tensor output information from ONNXruntime 'output_details' nodes
            '''
            t = od.type
            s = od.shape
            num_el = np.prod(s)
            num_bytes, np_type = bytes_from_type_and_elements(t, num_el)

            num_bytes_aligned = align(num_bytes)
            return num_bytes_aligned,  num_bytes, np_type

        if all([all([type(n) == int for n in od.shape]) for od in self.output_details]): 
            # condition is checking for dynamic axes, which are set as strings
            self.num_boxes = self.output_details[0].shape[1]
            for od in self.output_details:
                num_bytes_aligned,  num_bytes, np_type = get_size_from_output_details_onnx(od)
                self.tensor_offsets.append([num_bytes_aligned,  num_bytes])
                self.tensor_types.append(np_type)
        elif self.params['session']['runtime_options'].get('object_detection:top_k'):
            # if there is a 'top_k' parameter set, this is the number of boxes
            self.num_boxes = self.params['session']['runtime_options']['object_detection:top_k']

            if len(self.output_details) == 2:
                #rough assumption of only 2 tensors
                boxes_od = self.output_details[0]
                tensor_type = boxes_od.type

                num_el = 5 * self.num_boxes
                num_bytes, np_type = bytes_from_type_and_elements(tensor_type, num_el)
                num_bytes_aligned = align(num_bytes)
                self.tensor_offsets.append([num_bytes, num_bytes_aligned])
                self.tensor_types.append(np_type)

                classes_od = self.output_details[1]
                tensor_type = classes_od.type

                num_el = self.num_boxes
                num_bytes, np_type = bytes_from_type_and_elements(tensor_type, num_el)
                num_bytes_aligned = align(num_bytes)
                self.tensor_offsets.append([num_bytes, num_bytes_aligned])
                self.tensor_types.append(np_type)

            else: 
                sample_od = self.output_details[0]
                tensor_type = self.output_details.type

                num_el = 6 * self.num_boxes
                num_bytes, np_type = bytes_from_type_and_elements(tensor_type, num_el)
                num_bytes_aligned = align(num_bytes)
                self.tensor_offsets.append([num_bytes, num_bytes_aligned])
                self.tensor_types.append(np_type)


        elif self.params.get('output_details'):
            '''Explictly described in SDK 9.0... this will makes things much easier once ready'''
            raise NotImplementedError('9.0 style is not added yet..')

        else:
            #fallback to just run an image through using CPU and use the output tensors... may be slower
            t = self.input_details[0].type
            _, np_t = bytes_from_type_and_elements(0, t)
            fake_input = np.zeros(self.input_details[0].shape, dtype=np_t)
            result = self.run_onnx(fake_input)
            for r in result:
                s = r.shape
                t = r.dtype
                num_el = np.prod(s)
                num_bytes = np.dtype(t.itemsize) * num_el
                num_bytes_aligned = align(num_bytes)
                self.tensor_offsets.append([num_bytes_aligned,  num_bytes])
                self.tensor_types.append(t)

        return self.tensor_offsets

    def load_model_tidl(self):
        ext = self.modelfile.split('.')[-1]

        if 'onnx' in ext:
            self.model_type = 'onnx'

            sess_options = onnxruntime.SessionOptions()
            ep_list = ['TIDLExecutionProvider','CPUExecutionProvider']
            runtime_options = {
                'artifacts_folder': os.path.join(self.modeldir, self.params['session']['artifacts_folder']),
                'debug_level': 0,
                'advanced_options:output_feature_16bit_names_list': '511, 983 ',
            }

            provider_options = [runtime_options, {}]

            self.model = onnxruntime.InferenceSession(self.modelfile, providers=ep_list, provider_options=provider_options, sess_options=sess_options)
            self.input_details = self.model.get_inputs()
            self.output_details = self.model.get_outputs()

            i = self.input_details[0]
            self.input_type = i.type.split('(')[-1][:-1] #format of type is "tensor($TYPE)", e.g. "tensor(uint8)"
            if self.input_type == 'float':
                self.input_type = 'float32'

            self.calculate_output_tensor_sizes()

    def load_model(self):
        '''
        Load model to see input and output details. Will not be directly used for computation. 

        May be worth unloading this model after saving some IO information, since it will consume RAM otherwise
        '''
        ext = self.modelfile.split('.')[-1]

        
        if 'onnx' in ext:
            self.model_type = 'onnx'

            sess_options = onnxruntime.SessionOptions()
            ep_list = ['TIDLExecutionProvider','CPUExecutionProvider']
            runtime_options = {
                'artifacts_folder': os.path.join(self.modeldir, self.params['session']['artifacts_folder']),
                'debug_level': 2
            }
            # ep_list = ['TIDLExecutionProvider','CPUExecutionProvider']
            ep_list = ['CPUExecutionProvider']

            # provider_options = [runtime_options, {}]
            provider_options = [{}]

            self.model = onnxruntime.InferenceSession(self.modelfile, providers=ep_list, provider_options=provider_options, sess_options=sess_options)
            self.input_details = self.model.get_inputs()
            self.output_details = self.model.get_outputs()

            i = self.input_details[0]
            self.input_type = i.type.split('(')[-1][:-1] #format of type is "tensor($TYPE)", e.g. "tensor(uint8)"
            if self.input_type == 'float':
                self.input_type = 'float32'

            self.calculate_output_tensor_sizes()
            

        elif 'tflite' in ext:
            self.model_type = 'tflite'

            raise NotImplementedError('TFLite models are not implemented yet')

    def run_onnx(self, input_tensor):
        result = self.model.run(None, {self.input_details[0].name: input_tensor}) #format depends on model. mobilenetv2SSD

        return result


    def decode_output_tensor(self, tensor_buffer):
        '''
        model dependent... 
        '''
        # print(len(tensor_buffer))
        # print(self.output_details[0])
        # out_shape = self.params['session']['output_details'][0]['shape']
        # tensor = np.ndarray(self.output_details[0].shape, self.params['session']['output_details'][0]['type'], tensor_buffer)
        tensor = np.ndarray(self.output_details[0].shape, self.tensor_types[0], tensor_buffer)
        return tensor

    def decode_input_tensor(self, tensor_buffer):
        '''
        model dependent... 
        '''
        print(len(tensor_buffer))
        in_shape = self.params['session']['input_details'][0]['shape']
        print(in_shape)
        in_type = self.params['session']['input_details'][0]['type']
        if 'float' in in_type: 
            np_type = np.float32
        elif 'int8' in in_type: 
            np_type = np.uint8
        elif 'int16' in in_type: 
            np_type = np.uint16
        elif 'int32' in in_type: 
            np_type = np.uint32
        elif 'int64' in in_type: 
            np_type = np.uint64
        # tensor = np.ndarray(self.output_details[0].shape, self.params['session']['output_details'][0]['type'], tensor_buffer)
        tensor = np.ndarray(in_shape, np_type, tensor_buffer)
        return tensor

    def resize_boxes(self, boxes_tensor, image_height, image_width):
        '''
        Assumed that format is x1,y1,x2,y2,score,class for each box (row) in the tensor
        '''


        for box in boxes_tensor:
            if self.params['postprocess'].get('normalized_detections'):
                box[0] *= image_width
                box[1] *= image_height
                box[2] *= image_width
                box[3] *= image_height
            else:
                box[0] *= image_width / self.model_width
                box[1] *= image_height / self.model_height
                box[2] *= image_width / self.model_width
                box[3] *= image_height / self.model_height

        return boxes_tensor

