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
This file configures a gstreamer pipeline that will run this demo. 

It is assumed there is 1 camera (imx219 or usb cameras supporting 720p or 1080p), 1 display, and 1 model to run
'''
import numpy as np
import math
import time


import display, model_runner

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib, GObject
Gst.init(None)


QUEUE_STR = 'queue leaky=2 max-size-buffers=2'


class CamParams():
    '''
    Save some premade/known camera parameters
    '''
    def __init__(self, cam_name, device='/dev/video2'):
        if cam_name == 'imx219': 
            self.width = 1640
            self.height = 1232
            self.fps = '30/1'
            self.pixel_format = 'RGB'

            # the ISP binary paths will have to change based on the SDK version. This is for 8.6. 
            self.input_gst_str = f'v4l2src device={device}  io-mode=dmabuf-import ! video/x-bayer, width={self.width}, height={self.height}, format=rggb10 ! tiovxisp sink_0::device=/dev/v4l-subdev2 sensor-name=SENSOR_SONY_IMX219_RPI dcc-isp-file=/opt/imaging/imx219/dcc_viss_10b_1640x1232.bin sink_0::dcc-2a-file=/opt/imaging/imx219/dcc_2a_10b_1640x1232.bin format-msb=9 ' 

        # c270 webcam settings
        elif cam_name=='usb-720p':
            self.width = 1280
            self.height = 720
            self.fps = '30/1'
            self.pixel_format = 'RGB'

            self.input_gst_str = f'v4l2src device={device}  ! image/jpeg,width={self.width},height={self.height} ! jpegdec '
            
        elif cam_name=='usb-1080p':
            self.width = 1920
            self.height = 1080
            self.fps = '30/1'
            self.pixel_format = 'RGB'

            self.input_gst_str = f'v4l2src device={device}  ! image/jpeg,width={self.width},height={self.height} !  jpegdec '
        else: 
            raise ValueError('cam_name not recognized: ' + cam_name)


class GstBuilder():
    def __init__(self, model_params, camera_params, display_obj:display.DisplayDrawer, appsink_tensor_name='tensor_in', appsink_image_name='image_in', appsrc_name='out'):
        '''
        GST pipeline builder class. Requires information about the input, model, and output. 
        '''
        self.model_params = model_params
        self.camera_params = camera_params

        self.appsink_tensor_name = appsink_tensor_name
        self.appsink_image_name = appsink_image_name
        self.appsrc_name = appsrc_name

        self.appsrc_output_format = 'RGB'
        
        self.display = display_obj

    def generate_resize_string(self, in_height, in_width, model_height, model_width):
        '''
        Generate the tiovxmultiscaler gstreamer string that is used for the model input. 
            This can only downscale from in_h/w to model_h/w. 
            If ther eis a down-scale by more than 4x, then the multiscaler has to be called twice due to silicon limitations
        '''
        MAX_RESIZE_FACTOR = 4


        gst_string = f'   split_resize.  '
        # gst_string = f'   split_resize. ! {QUEUE_STR} '

        width_ratio = in_width / model_width
        height_ratio = in_height / model_height
        if width_ratio > MAX_RESIZE_FACTOR or height_ratio > MAX_RESIZE_FACTOR:
            common_ratio = min([width_ratio, height_ratio, MAX_RESIZE_FACTOR])
            resize_h = math.ceil(in_height / common_ratio)
            resize_w = math.ceil(in_width / common_ratio)
            if resize_h % 2 == 1: resize_h +=1 #cannot be an odd number!
            if resize_w % 2 == 1: resize_w +=1 
            
            return gst_string + f' ! video/x-raw, width={resize_w}, height={resize_h}, format=NV12  ! tiovxmultiscaler target=1 ! {QUEUE_STR} ! video/x-raw, width={model_width}, height={model_height} '

        else:
            return gst_string + f' ! video/x-raw, width={model_width}, height={model_height}, format=NV12  '


    def build_gst_strings(self, model_obj:model_runner.ModelRunner):
        '''
        Build a GST string that pulls input, preprocesses, runs inference, post 
        processing, exposes interface to application code, and merges app output 
        with postproc output for display.

        param model_obj: Holds information about the model object, primarily 
        data type. Most parameters instead come from self.model_params, 
        which is set in __init__

        Note that queues here often play a very important role! Then need to have max sizes and drop policies to prevent long latency and memory overflows
        '''
    
        video_conv = 'tiovxdlcolorconvert' # videoconvert # tiovxdlcolorconvert #tiovxdl are Neon optimized

        # input from camera and get ready to split into two 
        gst_str = self.camera_params.input_gst_str 
        gst_str+= f'! {video_conv}  !  video/x-raw, format=NV12 ! {QUEUE_STR}  ! tiovxmultiscaler name=split_resize  ' 
        # gst_str+= f' ! {video_conv}  !  video/x-raw, format=NV12  ! tiovxmultiscaler name=split_resize  ' 
        
        tensor_format=self.model_params['preprocess']['data_layout']
        data_type = model_obj.input_type
        print('model datatype : ' + str(data_type))
        
        # pipeline to do DL inference on. Requires preprocessing to match model
        gst_str += self.generate_resize_string(self.camera_params.height, self.camera_params.width, model_obj.model_height, model_obj.model_width)

        #do preprocessing
        tensor_format = 'BGR' if self.model_params['preprocess']['reverse_channels'] else 'RGB'
        gst_str += f' ! tiovxdlpreproc out-pool-size=4 data-type={data_type}   channel-order={self.model_params["session"]["input_data_layout"].lower()} tensor-format={tensor_format.lower()} '
        if self.model_params['session']['input_scale'] and self.model_params['session']['input_scale']:
            # subtract mean and multiply by scale in the tiovxdlpreproc
            params_mean = self.model_params['session']['input_mean']
            params_scale = self.model_params['session']['input_scale'] 
            preproc_param_str = ' mean-0=%f mean-1=%f mean-2=%f scale-0=%f scale-1=%f scale-2=%f ' % (params_mean[0], params_mean[1], params_mean[2], params_scale[0], params_scale[1], params_scale[2])
            gst_str += preproc_param_str
        gst_str += f' ! application/x-tensor-tiovx '
        #output from preproc is a tensor

        #run inference and push into application code via appsink
        gst_str += f' ! appsink name={self.appsink_tensor_name} max-buffers=2 drop=True '

        # another copy of the input image is resized and pushed to application code via appsink
        gst_str += f'   split_resize. ! {QUEUE_STR}  ! video/x-raw, width={self.display.image_width}, height={self.display.image_height}, format=NV12 ! {video_conv} out-pool-size=4 ! video/x-raw, format=RGB ! appsink name={self.appsink_image_name} max-buffers=2 drop=True'


        #### boundary between input gstreamer string and output gstreamer string. Application code (appsink and appsrc) sits between these two. The two gstreamer strings are their own unique pipelines. 
        

        out_gst_str = ''
        # Application output will come from appsrc and be converted to more usable format (NV12)
        out_gst_str += f' appsrc format=GST_FORMAT_TIME is-live=true  name={self.appsrc_name} ! video/x-raw,  format={self.appsrc_output_format}, width={self.display.display_width}, height={self.display.display_height} '
        out_gst_str += f' ! queue ! {video_conv} out-pool-size=4  ! video/x-raw, format=NV12  '

        # Create an overlay with performance information
        # out_gst_str += f' ! queue  '
        out_gst_str += f'!  tiperfoverlay main-title=\"\"  '
        # Push to the display via kmssink
        out_gst_str += f'! kmssink sync=false driver-name=tidss  plane-id=31 force-modesetting=True'
        
        self.gst_str = gst_str
        self.out_gst_str = out_gst_str

        # define output caps for the receipt image coming from appsrc
        gst_caps_str = "video/x-raw, " + \
            "width=%d, " % self.display.display_width + \
            "height=%d, " % self.display.display_height + \
            "format=%s, " % self.appsrc_output_format + \
            "framerate=%s" % '0/1'
        self.gst_caps = Gst.caps_from_string(gst_caps_str)
        print("caps: " + str(gst_caps_str))

        return gst_str, out_gst_str
    

    def setup_gst_appsrcsink(self):
        '''
        Parse the GST pipeline string and launch. 
        Retrieve application interfaces (appsink input, appsrc output) 
        '''
        print('Parsing GST pipeline: \ninput: %s\n\noutput: %s\n' % (self.gst_str, self.out_gst_str))

        self.pipe = Gst.parse_launch(self.gst_str)
        self.out_pipe = Gst.parse_launch(self.out_gst_str)

        self.app_in_tensor = self.pipe.get_by_name(self.appsink_tensor_name)
        self.app_in_image = self.pipe.get_by_name(self.appsink_image_name)
        self.app_out = self.out_pipe.get_by_name(self.appsrc_name)


    def start_gst(self):
        '''
        Set the GST pipeline to start playing
        '''
        print('Starting GST pipeline')
        s = self.pipe.set_state(Gst.State.PLAYING)
        s = self.out_pipe.set_state(Gst.State.PLAYING)

    def pull_sample(self, app, loop=True):
        '''
        Retrieve a sample from the appsink 'app' and return a buffer of data.
        The pipeline and appsink instance must be PLAYING state

        param app: The appsink obtained from a valid pipeline
        '''
        data = None
        struct = None
        sample = app.try_pull_sample(50000000)
        if type(sample) != Gst.Sample:
            # Poll endlessly for a sample
            if loop:
                while type(sample) != Gst.Sample:
                    sample = app.try_pull_sample(50000000)
            else: return data, struct # None, None

        buffer = sample.get_buffer()
        _, map_info = buffer.map(Gst.MapFlags.READ)
        # the buffer of data
        data = map_info.data
        # release the sample and map to GST memory
        buffer.unmap(map_info)
        
        # get caps to help learn about input structure
        appsink_caps = sample.get_caps()
        struct = appsink_caps.get_structure(0)

        return data, struct

    def format_image_from_sample(self, sample_image, struct):
        '''
        Reformat the image from an input byte buffer into a numpy ndarray with proper height, width, and channels

        Assuming that inputs are RGB images
        '''
        width = struct.get_value("width")
        height = struct.get_value("height")
        frame = np.ndarray((height, width, 3), np.uint8, sample_image)

        return frame
    