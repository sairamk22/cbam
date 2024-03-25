The focus of my academic project in machine learning lies in enhancing interpretability and understanding within deep learning models, specifically through the integration of convolution block attention layers and self-attention-enabled learners. By augmenting classic architectures like ResNet18 with these innovative layers, I aimed to visually analyze the correlation between attention maps and segmentations.
The project represents an effort to systematically explore and validate the alignment between attention maps and actual segmentations, thus bridging a crucial gap in understanding. Unlike existing literature, which often provides anecdotal comparisons on a limited set of images, my work seeks to establish a comprehensive and rigorous analysis across various datasets.

By leveraging convolution block attention layers and self-attention mechanisms, I aimed to elucidate meaningful patterns and correlations between attention maps and segmentations, thereby enhancing interpretability and providing insightful explanations for model decisions. This project not only pushes the boundaries of interpretability in deep learning but also lays the foundation for future research endeavours aimed at unravelling the intricacies of neural network behaviour.
1. Implemented standalone CBAM layers that can be used in any convolutional architecture like ResNet
2. Acquired data from a semantic segmentation dataset (Pascal VOC 2012)
3. Modified ResNet18 architecture to include CBAM layers
4. The classification model (ResNet18) was trained with cross-entropy as a loss function.
5. Archived decent accuracy on this dataset and extracted generated channel maps and spatial maps from every layer
6. Visualized these attention maps in a grid(spatial and channel combined) alongside segmentations to relate them(layerwise).
