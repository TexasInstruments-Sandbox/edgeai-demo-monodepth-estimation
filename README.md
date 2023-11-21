# Edge AI Monodepth Estimation Demo

This demo application shows a depth-estimation using a single camera and a deep learning CNN. 

Depth is crucial for understanding and navigating 3-D space. Typically, depth is estimated using time-of-flight or LIDAR systems, which are high resolution, high accuracy, and generally high cost/power. Radar and ultrasonic sensors are alternatives, but have resolution limitations. Cameras are more ubiquitous, but the sensor inherently offers no depth information in the 2-D image, thereby requiring sophisticated postprocessing to infer depth. Stereo depth is quite accurate, but requires careful calibration and processing. Single camera depth is less accurate, but has lower cost -- in some cases, like detecting an object in the foreground vs. the background, this is perfectly suitable.

The demo runs a depth-estimation neural network using a single camera. This produces a relative depth map - values may be somewhat related to real world measurements, but they are rarely accurate in general scenes/settings. With this in mind, the output depth map is postprocessed into a heatmap that displays distances using colors -- closer objects will be red or 'hot' whereas farther objects (e.g. the background) will be blue or 'cold'.

This demo has been validated on the [AM62A SoC](https://www.ti.com/product/AM62A7) running the 9.0.0 Edge AI Linux SDK. It is expected to run equivalently well on other AM6xA / Edge AI processors from TI, like the TDA4VM, AM68A, and AM69A.

## How to run this demo

Note: this demo borrows heavily from the [edgeai-gst-apps-retail-checkout](https://github.com/TexasInstruments/edgeai-gst-apps-retail-checkout) project for running constructing a image-processing gstreamer pipeline. The default model is MiDaS [1]

1. Obtain an EVM for the AM6xA processor of choice, e.g. the [AM62A Starter Kit](https://www.ti.com/tool/SK-AM62A-LP)
2. Flash an SD card with the Edge AI SDK (Linux) by following the quick start guide [(Quick start for AM62A)](https://dev.ti.com/tirex/explore/node?node=A__AQniYj7pI2aoPAFMxWtKDQ__am62ax-devtools__FUz-xrs__LATEST)
3. Login to the device over a serial or SSH connection. A network connection is required to setup this demo
4. Clone this repository to the device using git.  
  * If the EVM is behind a proxy, first set the HTTPS_PROXY environment variable and then add it to git: `git config --global https.proxy $HTTPS_PROXY`
5. Run the run_demo.sh script. 

### Recompiling the model for another device or SDK version than AM62A w/ 9.0 SDK

This model was compiled with [edgeai-benchmark](https://github.com/texasinstruments/edgeai-benchmark). Compilation for this model takes slightly more effort than others since depth is a less standard task. Edgeai-benchmark includes scripts to help with this task on a pre-optimized MiDaS model, but this requires some setup. This includes a benchmark .PY script, a settings.YAML file, and the dataset to use for calibration.

The settings YAML file should include the following (see [edgeai_setting_import_PC_depth.yaml](./edgeai_setting_import_PC_depth.yaml)): 
``` YAML
model_selection : ['de-7310_onnxrt']

dataset_type_dict:
  'depth_estimation': 'nyudepthv2'

dataset_selection : ['nyudepthv2']
```

This model (7310 desginates MiDaS) was trained on NYUDepthv2, so that dataset will be used for calibrating the model. The dataset should be downloaded as the script is running. See the main function for a line to uncomment if that does not complete successfully.

As an example for the calling script, the [edgeai_benchmark_depth.py](./edgeai_benchmark_depth.py) can be placed under edgeai-benchmark/scripts, and called as:
``` bash
python3 scripts/edgeai_benchmark_depth.py --target_device AM62A edgeai_setting_import_PC_depth.yaml
```

## Resources and Help

* [Support Forums](https://e2e.ti.com)
* [Edge AI Resources FAQ](https://e2e.ti.com/support/processors-group/processors/f/processors-forum/1236957/faq-edge-ai-studio-edge-ai-resources-for-am6xa-socs)
* [TI Edge AI Github](https://github.com/TexasInstruments/edgeai)
* [Edge AI Studio cloud-based resources](https://dev.ti.com/edgeaistudio/)

## References
[1] Rene Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun, Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020