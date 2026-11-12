# VAGeo: View-specific Attention for Cross-View Object Geo-Localization

### Abstract
Cross-view object geo-localization (CVOGL) aims to locate an object of interest in a captured ground- or drone view image within the satellite image. However, existing works treat ground-view and drone-view query images equivalently, overlooking their inherent viewpoint discrepancies and the spatial correlation between the query image and the satellite-view reference image. To this end, this paper proposes a novel View specific Attention Geo-localization method (VAGeo) for accurate CVOGL. Specifically, VAGeo contains two key modules: view specific positional encoding (VSPE) module and channel-spatial hybrid attention (CSHA) module. In object-level, according to the characteristics of different viewpoints of ground and drone query images, viewpoint-specific positional codings are designed to more accurately identify the click-point object of the query image in the VSPE module. In feature-level, a hybrid attention in the CSHA module is introduced by combining channel attention and spatial attention mechanisms simultaneously for learning discriminative features. Extensive experimental results demonstrate that the proposed VAGeo gains a significant performance improvement, i.e., improving acc@0.25/acc@0.5 on the CVOGL dataset from 45.43%/42.24% to 48.21%/45.22% for ground-view, and from 61.97%/57.66% to 66.19%/61.87% for drone-view.

Index Terms—Cross-view Object-level Localization, View specific Positional Encoding, Hybrid Attention Mechanism

# Dataset
We evaluate our method on the publicly available CVOGL dataset.

The download link is  [CVOGL](https://drive.google.com/file/d/1WCwnK_rrU--ZOIQtmaKdR0TXcmtzU4cf/view?usp=sharing) 

# Train

1. Firstly, download CVOGL and rename it to 'data', i.e., 'data/CVOGL_DroneAerial' and 'data/CVOGL_SVI'.
2. Secondly, download the pretrained YOLOv3 weights and `resnet18-f37072fd.pth`, and move both files into the `saved_models` directory (i.e., `./saved_models/yolov3.weights` and `./saved_models/resnet18-f37072fd.pth`).
3. Thirdly,Modify the related file paths.
4.  Fourthly，execute 'scripts/run_train_all.sh', 'scripts/run_train_droneaerial.sh', or 'scripts/run_train_svi.sh' to train the models.
```
sh scripts/run_train_all.sh
# sh scripts/run_train_droneaerial.sh
# sh scripts/run_train_svi.sh
```

# Test
```
sh scripts/run_test_all.sh
# sh scripts/run_test_droneaerial.sh
# sh scripts/run_test_svi.sh
```

# Citation

```
@INPROCEEDINGS{10888758,
  author={Li, Zhongyang and Yuan, Xin and Liu, Wei and Xu, Xin},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={VAGeo: View-specific Attention for Cross-View Object Geo-Localization}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Location awareness;Accuracy;Attention mechanisms;Image coding;Signal processing;Encoding;Satellite images;Object recognition;Speech processing;Drones;Cross-view Object-level Localization;View-specific Positional Encoding;Hybrid Attention Mechanism},
  doi={10.1109/ICASSP49660.2025.10888758}}

```



