# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import argparse
import cv2
from edgeai_benchmark import *
import configs


def get_imagecls_dataset_loaders(settings, download=False):
    # this example uses the datasets.ImageClassification data loader
    # this data loader assumes that the split argument provided is a text file containing a list of images
    # that are inside the folder provided in the path argument.
    # the split file also should contain the class id after a space.
    # so the split file format should be (for example)
    # image3.png 10
    # image2.jpg 1
    # image_cat.png 3
    # etc.
    # all the images need not be inside the path, but can be inside subdirectories,
    # but path combined with the lines in split file should give the image path.
    dataset_calib_cfg = dict(
        path=f'{settings.datasets_path}/imagenet/val',
        split=f'{settings.datasets_path}/imagenet/val.txt',
        num_classes=1000,
        shuffle=True,
        num_frames=min(settings.calibration_frames,50000),
        name='imagenet'
    )

    # dataset parameters for actual inference
    dataset_val_cfg = dict(
        path=f'{settings.datasets_path}/imagenet/val',
        split=f'{settings.datasets_path}/imagenet/val.txt',
        num_classes=1000,
        shuffle=True,
        num_frames=min(settings.num_frames,50000),
        name='imagenet'
    )

    # you are free to use any other data loaded provided in datasets folder or write your own instead of this
    calib_dataset = datasets.ImageClassification(**dataset_calib_cfg, download=download)
    val_dataset = datasets.ImageClassification(**dataset_val_cfg, download=download)
    return calib_dataset, val_dataset


def create_configs(settings, work_dir):
    '''
    configs for each model pipeline
    - calibration_dataset: dataset to be used for import - should support __len__ and __getitem__.
    - input_dataset: dataset to be used for inference - should support __len__ and __getitem__
      Output of __getitem__ should be understood by the preprocess stage.
      For example, if the dataset returns image filenames, the first entry in the preprocess can be an image read class.
    - preprocess is just a list of transforms wrapped in utils.TransformsCompose.
      It depends on what the dataset class outputs and what the model expects.
      We have some default transforms defined in settings.
    - postprocess is also a list of transforms wrapped in utils.TransformsCompose
      It depends on what the model outputs and what the metric evaluation expects.
    - metric - evaluation metric (eg. accuracy). If metric is not defined in the pipeline,
      evaluate() function of the dataset will be called.

    parameters for calibration_dataset and input_dataset
    - path: folder containing images
    - split: provide a .txt file containing two entries in each line
      first entry in each line is image file name (starting from path above),
      for classification, second entry is class id (just set to 0 if you don't know what it is)
        example:
          image10.jpg 0
          tomato/image2.jpg 9
      for segmentation, second entry is the label image.
      for detection, second entry is not used right now in this script.
    '''

    # get dataset loaders
    # imagecls_cflib_dataset, imagedet_val_dataset = get_imagedet_dataset_loaders(settings)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    common_session_cfg = sessions.get_common_session_cfg(settings, work_dir=work_dir)
    onnx_session_cfg = sessions.get_onnx_session_cfg(settings, work_dir=work_dir)
    onnx_bgr_session_cfg = sessions.get_onnx_bgr_session_cfg(settings, work_dir=work_dir)
    onnx_quant_session_cfg = sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir)
    onnx_bgr_quant_session_cfg = sessions.get_onnx_bgr_quant_session_cfg(settings, work_dir=work_dir)
    jai_session_cfg = sessions.get_jai_session_cfg(settings, work_dir=work_dir)
    jai_quant_session_cfg = sessions.get_jai_quant_session_cfg(settings, work_dir=work_dir)
    mxnet_session_cfg = sessions.get_mxnet_session_cfg(settings, work_dir=work_dir)
    tflite_session_cfg = sessions.get_tflite_session_cfg(settings, work_dir=work_dir)
    tflite_quant_session_cfg = sessions.get_tflite_quant_session_cfg(settings, work_dir=work_dir)
    
    
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)
    postproc_depth_estimation_onnx = postproc_transforms.get_transform_depth_estimation_onnx()

    nyudepthv2_cfg = {
        'task_type': 'depth_estimation',
        'dataset_category': datasets.DATASET_CATEGORY_NYUDEPTHV2,
        'calibration_dataset': settings.dataset_cache['nyudepthv2']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['nyudepthv2']['input_dataset'],
    }

    print(nyudepthv2_cfg['calibration_dataset'])
    pipeline_configs = {
        # 'imagecls-1': dict(
        #     task_type='classification',
        #     calibration_dataset=imagecls_calib_dataset,
        #     input_dataset=imagecls_val_dataset,
        #     preprocess=preproc_transforms.get_transform_onnx(),
        #     session=sessions.ONNXRTSession(**onnx_session_cfg,
        #         runtime_options=settings.runtime_options_onnx_np2(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx'),
        #     postprocess=postproc_transforms.get_transform_classification(),
        #     model_info=dict(metric_reference={'accuracy_top1%':71.88})
        # ),
        'de-7310':utils.dict_update(nyudepthv2_cfg,
            preprocess=preproc_transforms.get_transform_jai((256,256), (256,256), backend='cv2', interpolation=cv2.INTER_CUBIC),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.28, 103.53), input_scale=(0.017125, 0.017507, 0.017429)),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {'advanced_options:output_feature_16bit_names_list':'511, 983'}),
                model_path=f'{settings.models_path}/vision/depth_estimation/nyudepthv2/MiDaS/midas-small.onnx'),
            postprocess=postproc_depth_estimation_onnx,
            metric=dict(disparity=True, scale_shift=True),
            model_info=dict(metric_reference={'accuracy_delta_1%':86.4}, model_shortlist=None)
        ),
    }
    return pipeline_configs


if __name__ == '__main__':
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', type=str)
    parser.add_argument('--model_selection', type=str, default=None, nargs='*')
    parser.add_argument('--target_device', type=str)

    # dataset_nyu = datasets.nyudepthv2.NYUDepthV2(path='./dependencies/datasets/nyudepthv2', download=True, split='val')
    cmds = parser.parse_args()
    settings = config_settings.ConfigSettings(cmds.settings_file, model_selection=cmds.model_selection,
                                              target_device=cmds.target_device)

    work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
    print(f'work_dir = {work_dir}')

    packaged_dir = os.path.join(f'{settings.modelartifacts_path}_package', f'{settings.tensor_bits}bits')
    print(f'packaged_dir = {packaged_dir}')

    # now run the actual pipeline
    print('Creating configs and modifying settings')
    print(settings)
    pipeline_configs_default = configs.get_configs(settings, work_dir=work_dir)
    pipeline_configs = create_configs(settings, work_dir)
    print('Configs created and settings modified')
    print(settings)
    # run the accuracy pipeline
    interfaces.run_accuracy(settings, work_dir, pipeline_configs)

    # package the artifacts
    interfaces.run_package(settings, work_dir, packaged_dir, custom_model=True)