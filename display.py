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
'''
This class creates handles the display and visualization by modifying input frames and pushing an output frame to gstreamer. 

Intended output display:
+------------------------------+----------+
|                              |          |
|                              |          |
|                              |  stats   |
|                              |  &       |
|  frame w/ post process       |  app     |
|  & visualization             |  info    |
|                              |          |
|                              |          |
|                              |          |
+------------------------------+----------+
|                                         |
|        performance stats/load           |
|                                         |
+-----------------------------------------+

The top two components of the display shown above will be made in this file. The bottom portion is blank so that tiperfoverlay can add performance overlay
'''


import numpy as np
import cv2 as cv

import utils


import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib, GObject
Gst.init(None)

RECEIPT_FONT = cv.FONT_HERSHEY_SIMPLEX
HEADING_FONT = cv.FONT_HERSHEY_TRIPLEX


class DisplayDrawer():
    '''
    Class to the output display. This is written primarily for 1920 x 1080 display, but should also scale to other sizes

    Performance stats should take up 20% of the image at the bottom, but have hard limit of height between 50 and 250 pixels. See tiperfoverlay gst plugin for source of this.

    '''
    def __init__(self, display_width=1920, display_height=1080, image_scale=0.8):
        self.display_width = display_width
        self.display_height = display_height
        self.image_scale = image_scale

        self.image_width = int(display_width * image_scale)
        self.image_height = int(display_height * image_scale)
        self.list_width = display_width - self.image_width
        self.list_height = self.image_height
        self.perf_width = display_width
        self.perf_height = display_height - self.image_height


    def set_gst_info(self, app_out, gst_caps): 
        '''
        Set output caps and hold onto a reference for the appsrc plugin that interfaces from here to the final output sink (by default, kmssink.. see gst_configs.py)
        '''
        self.gst_app_out = app_out
        self.gst_caps = gst_caps
        self.gst_app_out.set_caps(self.gst_caps)


    def push_to_display(self, image):
        '''
        Push an image to the display through the appsrc

        param image: and image whose dimensions and pixel format matches self.gst_caps
        '''

        buffer = Gst.Buffer.new_wrapped(image.tobytes())

        ret = self.gst_app_out.push_buffer(buffer)
      
    def make_frame_init(self):
        '''
        Make an initial frame to push immediately. This is intentionally blank
        '''
        return np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
    
    def make_frame_passthrough(self, input_image):
        # processed_image = self.make_depth_map(input_image, infer_output)

        frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        frame[0:self.image_height, 0:self.image_width] = input_image
        # frame[0:self.image_height, self.image_width:] = 255

        return frame

    def make_frame(self, input_image, infer_output, model_obj):
        '''
        Use the output information from the tidlinferer plugin (after reformatting to convenient shape)
            to write some useful information onto various portions of the screen
        
        input image: HxWxC numpy array
        infer_output: tensor/2d array of shape num_boxes x 6, where the 6 values are x1,y1,x2,y2,score, label
        categories: in same format as dataset.yaml, a mapping of class labels to class names (strings)
        model_obj: the ModelRunner object associated with the model being run with tidlinferer
        '''

        processed_image = self.make_depth_map(input_image, infer_output)

        frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        frame[0:self.image_height, 0:self.image_width] = processed_image
        frame[0:self.image_height, self.image_width:] = 255

        return frame
    
    def draw_bounding_boxes(self, image, boxes_tensor, categories, viz_thres=0.6):
        '''
        Draw bounding boxes with classnames onto the 
        
        Each box in the tensor expected to be x1,y1,x2,y2,score,class-label.

        '''
        for box in boxes_tensor:
            score = box[4]
            label = box[5]

            if score > viz_thres:
                class_name = categories[int(label)]['name']
                x1,y1,x2,y2 = box[:4]
                cv.rectangle(image, (x1,y1), (x2,y2), color=(0, 255, 255), thickness=4)
                cv.putText(image, class_name, (x1,y1), cv.FONT_HERSHEY_SIMPLEX, 0.75, color=(0, 255, 255), thickness=2)

        return image

    def make_depth_map(self, input_image, infer_output):
        '''
        Convert depth values from the output of the model into a heatmap image. This will scale all values to 0-255
        '''
        depth_values = infer_output[0,0]
        mm = depth_values.min()
        mM = depth_values.max()
        # print('Depth min (%f) and max(%f)' % (mm, mM))

        depth_values -= mm
        depth_values *= 255/mM

        # depth_values = 255 - depth_values #if using disparity / reverse colors

        heatmap = cv.applyColorMap(depth_values.astype(np.uint8), cv.COLORMAP_RAINBOW)
        #resize is the slowest operation here.. experiment with other interpolation methods to enhance performance
        heatmap = cv.resize(heatmap, (input_image.shape[1], input_image.shape[0]))

        #disable for higher performance
        heatmap_weight = 0.75
        output_image = cv.addWeighted(heatmap, heatmap_weight, input_image, 1-heatmap_weight, 0)

        return output_image
    
        
